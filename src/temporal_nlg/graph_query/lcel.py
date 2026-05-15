from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from temporal_nlg.graph_query.index import TemporalGraphIndex
from temporal_nlg.graph_query.grounding import SemanticGrounder, ScoredEdge as GroundingScoredEdge
from temporal_nlg.graph_query.retrieval import GraphAnswer, GraphRetriever
from temporal_nlg.graph_query.semantic import EdgeSemanticIndex, ScoredEdge as SemanticScoredEdge
from temporal_nlg.graph_query.row_index import RowRetrievalIndex
from temporal_nlg.graph_query.visualization import answer_to_mermaid

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    StrOutputParser = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore
    ChatOpenAI = None  # type: ignore


class QueryPlan(BaseModel):
    query_type: str = Field(default="unsupported")
    entities: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    relation_hint: str | None = None
    temporal_operator: str | None = None
    year: int | None = None
    start_year: int | None = None
    end_year: int | None = None
    k: int = 5
    max_hops: int = 4
    confidence: float = 0.0
    rationale: str | None = None


_YEAR_RE = re.compile(r"\b(1\d{3}|2\d{3})\b")

# ── Self-contained temporal logic detection ───────────────────────────────────
# These questions provide ALL facts needed inline — no graph lookup required.
# Pattern: explicit multi-interval timeline described in the question itself.

_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember))\b",
    re.IGNORECASE,
)
_DAY_NAMES_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_FROM_TO_RE = re.compile(r"\bfrom\b.{2,50}?\bto\b", re.IGNORECASE | re.DOTALL)
# Also catches en-dash / hyphen intervals like "2018–2022" or "Feb-March"
_YEAR_DASH_YEAR_RE = re.compile(r"\b(1\d{3}|2\d{3})\s*[\u2013\u2014\-]\s*(1\d{3}|2\d{3})\b")

# Signals that the question is asking about status/ordering *within* the described timeline
_SELF_CONTAINED_ANSWER_KW = frozenset({
    "status", "phase", "active", "which phase", "what phase",
    "did it happen", "was it", "were they",
    "what came before", "what came after",
    "came before", "came after",
    "before delivery", "before arrival", "before testing",
    "before development", "before launch",
    "what was active", "what was their",
})


def _is_self_contained(question: str) -> bool:
    """Return True when the question provides ALL temporal facts needed to answer it.

    These questions are pure temporal-logic / scheduling puzzles — the full timeline
    is given inline and no external knowledge graph lookup is required.

    Examples that trigger this:
      - "A concert was postponed from May 2020 to May 2021. Did it happen in 2020?"  → No
      - "Planning Jan, development Feb–March, testing April. Which phase was active March 15?"
      - "A package left Monday, arrived customs Wednesday, delivered Friday. What came before delivery?"
    """
    low = question.lower()

    # Count independent temporal anchors embedded in the question text
    from_to_count  = len(_FROM_TO_RE.findall(question))
    dash_ivl_count = len(_YEAR_DASH_YEAR_RE.findall(question))  # "2018–2022"
    month_count    = len(_MONTH_RE.findall(question))
    day_count      = len(_DAY_NAMES_RE.findall(question))
    year_count     = len(_YEAR_RE.findall(question))

    # Two year values + a causal/conditional connector → strong self-contained signal
    has_two_years_with_conditional = (
        year_count >= 2
        and any(kw in low for kw in ("postponed", "rescheduled", "moved to", "delayed"))
    )

    # Multiple explicit date labels (months/days) describing a sequence of phases
    has_phase_sequence = (month_count + day_count) >= 3

    # from-X-to-Y construction describing an interval that the question then asks about
    has_from_to_with_query = from_to_count >= 1 and any(
        kw in low for kw in _SELF_CONTAINED_ANSWER_KW
    )

    # Two from..to constructs (e.g. "attended 2018–2022... started work 2023")
    has_dual_intervals = from_to_count >= 2

    # En-dash year range(s) + an in-text query year
    has_dash_interval_with_query = (
        dash_ivl_count >= 1 and year_count >= 3  # e.g. "2018–2022 ... 2023 ... 2021?"
        or dash_ivl_count >= 2
    )

    # Guard: questions that ask for a specific entity name (who/what/which/where/name)
    # are NEVER self-contained — they always require a graph lookup to find the answer.
    # e.g. "Which city hosted Expo 2020?" looks like it has inline dates but is really
    # asking for a factual entity that must come from the graph, not from the question text.
    is_entity_question = bool(re.match(r"^(which|what|who|where|name)\b", low))
    if is_entity_question:
        return False

    return (
        has_two_years_with_conditional
        or has_phase_sequence
        or has_from_to_with_query
        or has_dual_intervals
        or has_dash_interval_with_query
    )


def _parse_edge_year(date_str) -> int | None:
    """Extract the 4-digit year from a date string like '2016' or '2007-06-29'."""
    if not date_str:
        return None
    m = re.match(r"(\d{4})", str(date_str))
    return int(m.group(1)) if m else None


_STOP_WORDS: set[str] = {
    "in", "during", "from", "to", "before", "after", "between", "since",
    "until", "when", "did", "does", "was", "were", "is", "are", "the",
    "a", "an", "of", "and", "or", "that", "this", "by", "for", "with",
    "on", "at", "it", "its", "be", "been", "being", "has", "have", "had",
    "do", "not", "no", "yes", "if", "than", "then", "so",
}

_QUESTION_WORDS: set[str] = {
    "who", "what", "which", "where", "how", "whom", "whose",
    "why", "name", "list", "tell",
}


_NOISE: set[str] = _STOP_WORDS | _QUESTION_WORDS | {
    "came", "won", "began", "begin", "started", "ended", "end",
    "released", "launched", "published", "hosted", "became",
    "first", "last", "earlier", "later", "latest", "earliest",
    "becoming", "called", "known", "named", "about",
    "term", "begin",
}


def _extract_entities(text: str) -> list[str]:
    """Extract candidate entity phrases from a question.

    Approach: strip punctuation, remove stop/question words, split on common
    function words, keep the remaining meaningful chunks.  No brittle regex
    NER — the downstream embedding grounding resolves these to actual graph
    nodes via dot-product.
    """
    clean = re.sub(r"[?!.,;:]+", " ", text).strip()

    # 1. Pull out quoted strings first (e.g. album '1989')
    candidates: list[str] = []
    for m in re.finditer(r"""[\u2018\u2019'"]+(.+?)[\u2018\u2019'"]+""", clean):
        span = m.group(1).strip()
        if span and len(span) > 1:
            candidates.append(span)
    # Remove quotes so they don't interfere with chunking
    clean = re.sub(r"""[\u2018\u2019'"]+.+?[\u2018\u2019'"]+""", " ", clean)

    # 2. Split on function / stop words → keep noun-phrase-like chunks
    chunks = re.split(
        r"\b(?:" + "|".join(re.escape(w) for w in sorted(_NOISE, key=len, reverse=True)) + r")\b",
        clean,
        flags=re.IGNORECASE,
    )

    for chunk in chunks:
        token = re.sub(r"\s+", " ", chunk).strip(" ,;:-")
        if not token or len(token) < 3:
            continue
        # Drop bare years — they're parsed separately
        if _YEAR_RE.fullmatch(token):
            continue
        # Drop overly long fragments (likely leftover sentence pieces)
        if len(token.split()) > 6:
            continue
        if token.lower() in _NOISE:
            continue
        if token not in candidates:
            candidates.append(token)

    return candidates[:6]


def _heuristic_plan(question: str) -> QueryPlan:
    text = (question or "").strip()
    low = text.lower()
    years = [int(v) for v in _YEAR_RE.findall(text)]

    # Prefer the *last* year that isn't inside quotes (the contextual year).
    # E.g. "Which artist released the album '1989' in 2014?" → year=2014
    unquoted = re.sub(r"""[\u2018\u2019'"]+.+?[\u2018\u2019'"]+""", "", text)
    unquoted_years = [int(v) for v in _YEAR_RE.findall(unquoted)]
    year = unquoted_years[-1] if unquoted_years else (years[-1] if years else None)

    entities = _extract_entities(text)

    # Determine query type via simple keyword priority rules.
    # Ordered from most specific to least specific.
    query_type = "state_at_time"  # default

    if low.startswith("if ") and ("will" in low or "could" in low or "same happen" in low):
        query_type = "analogical_transfer"
    elif "reason" in low or ("cause" in low and not low.startswith("if ")):
        query_type = "reason_of"
    elif "start affecting" in low or "started affecting" in low:
        query_type = "start_affecting"
    elif "how many" in low or re.search(r"\bcount\b", low):
        query_type = "temporal_count"
    elif "path" in low or "reach" in low:
        query_type = "temporal_path_existence"
    # Comparative ordering: "came first", "came later", "earlier: A or B", "before"
    elif re.search(r"came (first|earlier|later|last)\b", low):
        query_type = "first_occurrence" if "first" in low or "earlier" in low else "last_occurrence"
    elif ("earlier" in low or "later" in low) and "or" in low:
        query_type = "first_occurrence" if "earlier" in low else "last_occurrence"
    elif ("was " in low or "did " in low) and "before" in low:
        query_type = "first_occurrence"
    # "when did X first ..." but not "when the iPhone was first released"
    elif low.startswith("when did") and ("first" in low or "start" in low):
        query_type = "first_occurrence"
    # Yes/no existence: "was X ...", "did X ..."
    elif low.startswith("was ") or low.startswith("did "):
        query_type = "existence_in_time"
    # Who/what/which with a year → state_at_time (default already)

    return QueryPlan(
        query_type=query_type,
        entities=entities,
        year=year,
        confidence=0.4,
        rationale="heuristic parser fallback",
    )


def _normalize_query_type(raw_value: str, question: str) -> str:
    value = (raw_value or "").strip().lower()
    q = (question or "").lower()

    if value in {
        "state_at_time",
        "state_during_interval",
        "existence_in_time",
        "first_occurrence",
        "last_occurrence",
        "reason_of",
        "start_affecting",
        "analogical_transfer",
        "temporal_count",
        "temporal_top_k",
        "temporal_path_existence",
        "self_contained",
        "unsupported",
    }:
        return value

    if "analog" in value or "counterfactual" in value or q.startswith("if "):
        return "analogical_transfer"
    if "start" in value or "affect" in value or "start affecting" in q:
        return "start_affecting"
    if "reason" in value or "cause" in value:
        return "reason_of"
    if "first" in value or "earliest" in value:
        return "first_occurrence"
    if "last" in value or "latest" in value or "recent" in value:
        return "last_occurrence"
    if "before" in q and ("was " in q or "did " in q):
        return "first_occurrence"
    if "exist" in value or q.startswith("was ") or q.startswith("did "):
        return "existence_in_time"
    if "count" in value or "how many" in q:
        return "temporal_count"
    if "top" in value or "rank" in value:
        return "temporal_top_k"
    if "path" in value or "reach" in value:
        return "temporal_path_existence"
    if "state" in value:
        return "state_at_time"

    return "unsupported"


class TemporalGraphLCELPipeline:
    """Simple LCEL pipeline for temporal graph QA and evidence visualization."""

    def __init__(
        self,
        output_dir: str | Path,
        embed_model_name: str | None = None,
        embed_server_url: str | None = None,
    ):
        self.index = TemporalGraphIndex(output_dir)
        self.retriever = GraphRetriever(self.index)
        self.semantic_index = EdgeSemanticIndex(
            self.index,
            embed_model_name=embed_model_name,
            embed_server_url=embed_server_url,
        )
        # Grounder uses the same embed server; LLM server stays global
        _grounder_embed_url = embed_server_url or (os.getenv("QWEN_EMBED_URL") or "")
        self.semantic_grounder = SemanticGrounder(
            self.index,
            qwen_server_url=(os.getenv("QWEN_SERVER_URL") or ""),
            qwen_embed_url=_grounder_embed_url,
        )
        self.row_index = RowRetrievalIndex(output_dir)

        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano")
        self.internal_llm_backend = (os.getenv("GRAPH_INTERNAL_LLM_BACKEND") or "qwen_server").strip().lower()
        self.internal_openai_model = (os.getenv("GRAPH_INTERNAL_LLM_MODEL") or self.model_name).strip()
        self._internal_openai_llm = None
        if self.internal_llm_backend == "openai" and ChatOpenAI is not None:
            try:
                self._internal_openai_llm = ChatOpenAI(
                    model=self.internal_openai_model,
                    temperature=0.0,
                    max_tokens=260,
                )
            except Exception:
                self._internal_openai_llm = None
        self.planner_backend = (os.getenv("GRAPH_PLANNER_BACKEND") or "openai").strip().lower()
        self.qwen_server_url = (os.getenv("QWEN_SERVER_URL") or "").rstrip("/")
        self.qwen_planner_url = (
            os.getenv("QWEN_PLANNER_URL")
            or os.getenv("QWEN_LLM_URL")
            or os.getenv("QWEN_SERVER_URL")
            or ""
        ).rstrip("/")
        # Use heuristic planner when backend is 'heuristic' or no dedicated
        # /plan endpoint exists (avoids 404 noise against llama.cpp)
        if self.planner_backend == "heuristic":
            self._planner_chain = None
        else:
            self._planner_chain = self._build_planner_chain()

        self.parse_chain = RunnableLambda(lambda x: self._to_payload(str(x.get("question") or "")))
        self.retrieve_chain = self.parse_chain | RunnableLambda(self._retrieve)
        self.answer_chain = self.retrieve_chain | RunnableLambda(self._format_answer)

        self.llm_chain = None
        if ChatOpenAI is not None and ChatPromptTemplate is not None and StrOutputParser is not None:
            prompt = ChatPromptTemplate.from_template(
                """
You are a temporal graph analyst.
Answer the question using only the provided retrieval output.

Question:
{question}

Retrieved answer:
{answer_text}

Confidence:
{confidence}

Evidence:
{evidence}

If evidence is empty, explicitly say no evidence was found.
""".strip()
            )
            try:
                llm = ChatOpenAI(model=self.model_name, temperature=0.0, max_tokens=240)
                self.llm_chain = self.retrieve_chain | prompt | llm | StrOutputParser()
            except Exception:
                self.llm_chain = None

    def _build_planner_chain(self):
        if self.planner_backend == "qwen_server":
            return "qwen_server"

        if ChatOpenAI is None or ChatPromptTemplate is None:
            return None

        prompt = ChatPromptTemplate.from_template(
            """
You are a temporal graph query planner.
Convert user questions into a JSON plan with fields:
- query_type: one of [state_at_time,state_during_interval,existence_in_time,first_occurrence,last_occurrence,reason_of,start_affecting,analogical_transfer,temporal_count,temporal_top_k,temporal_path_existence,unsupported]
- entities: ordered list of key entities in the question
- locations: list of geographical entities (countries, cities, regions) if any
- relation_hint: short relation keyword if present
- temporal_operator: at|during|before|after|between|since|until|overlap|first|latest
- year,start_year,end_year,k,max_hops,confidence,rationale

Question:
{question}

Return only the structured plan.
""".strip()
        )

        try:
            llm = ChatOpenAI(model=self.model_name, temperature=0.0, max_tokens=220)
            return prompt | llm.with_structured_output(QueryPlan)
        except Exception:
            return None

    def _plan_with_qwen_server(self, question: str) -> QueryPlan | None:
        if not self.qwen_planner_url:
            return None
        payload = json.dumps({"question": question}).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.qwen_planner_url}/plan",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if isinstance(body, dict):
                return QueryPlan(**body)
            return None
        except Exception:
            return None

    def _to_payload(self, question: str) -> Dict[str, Any]:
        plan = None
        if self._planner_chain == "qwen_server":
            plan = self._plan_with_qwen_server(question)
        elif self._planner_chain is not None:
            try:
                response = self._planner_chain.invoke({"question": question})
                if isinstance(response, QueryPlan):
                    plan = response
                elif hasattr(response, "model_dump"):
                    plan = QueryPlan(**response.model_dump())
                elif isinstance(response, dict):
                    plan = QueryPlan(**response)
            except Exception:
                plan = None

        if plan is None:
            plan = _heuristic_plan(question)

        plan.query_type = _normalize_query_type(plan.query_type, question)

        # Override: self-contained temporal logic problems bypass graph retrieval
        if _is_self_contained(question):
            plan.query_type = "self_contained"

        # Stash question in plan so grounding can use raw tokens for keyword fallback
        plan_dict = plan.model_dump()
        plan_dict["_question"] = question
        return {
            "question": question,
            "plan": plan_dict,
        }

    # ── LLM helpers for 2-stage retrieval ─────────────────────────────

    def _call_llm(self, prompt: str, system: str = "", max_tokens: int = 200) -> str:
        """Single LLM call via the configured internal backend.

        Backends:
        - openai: ChatOpenAI model configured by GRAPH_INTERNAL_LLM_MODEL
        - qwen_server (default): local /v1/chat/completions endpoint
        """
        if self.internal_llm_backend == "openai" and self._internal_openai_llm is not None:
            try:
                if system:
                    msg = self._internal_openai_llm.invoke([
                        ("system", system),
                        ("human", prompt),
                    ])
                else:
                    msg = self._internal_openai_llm.invoke(prompt)
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            text = item.get("text")
                            if text is not None:
                                parts.append(str(text))
                    return "\n".join(p for p in parts if p).strip()
                return str(content).strip()
            except Exception:
                return ""

        if self.internal_llm_backend == "none":
            return ""

        url = self.qwen_planner_url or self.qwen_server_url
        if not url:
            return ""
        model_name = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3.5-0.8B")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = json.dumps({
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            url=f"{url.rstrip('/')}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, str):
                    return content.strip()
        except Exception:
            pass
        return ""

    def _stage1_filter_edges(
        self,
        question: str,
        scored_edges: list,  # List[GroundingScoredEdge]
        year: int | None,
        query_type: str = "",
        max_show: int = 30,
        plan_entities: list | None = None,
    ) -> Dict[str, Any]:
        """Stage 1: show numbered edge triples to LLM, ask it to pick relevant ones.

        Returns dict with:
          selected_indices: list[int]  — 0-based indices into scored_edges
          need_more: bool
          stage1_prompt: str
          stage1_raw: str
        """
        # Show more edges for comparison questions so both entities appear
        if query_type in ("first_occurrence", "last_occurrence"):
            max_show = 40

        if not scored_edges:
            return {
                "selected_indices": [],
                "need_more": False,
                "stage1_prompt": "",
                "stage1_raw": "",
            }

        # ── Pre-filter: remove clearly wrong-era edges before showing to LLM ──
        # When target year is set e.g. 2016, exclude edges whose ONLY date is in a
        # completely different century (e.g. year 1816).  This prevents the 0.8B
        # model from being confused by unrelated historical edges.
        show_pool = scored_edges
        if year and year >= 1950:
            # Use a tight ±20-year window so that clearly off-era edges
            # (e.g. Seoul 1988 when looking for 2016, Adenauer 1949 when
            # looking for 2006) are excluded before the 0.8B LLM sees them.
            era_min = year - 20
            era_max = year + 20
            def _in_era(se) -> bool:
                ey = _parse_edge_year(se.edge.start)
                if ey is None:
                    ey = _parse_edge_year(se.edge.end)
                # No date on edge → keep (could be an entity match without dates)
                if ey is None:
                    return True
                return era_min <= ey <= era_max
            filtered_pool = [se for se in scored_edges if _in_era(se)]
            # Only apply if it doesn't remove too many candidates
            if len(filtered_pool) >= 3:
                show_pool = filtered_pool

        # Build numbered edge list (reduced to max_show for 0.8B model)
        show_edges = show_pool[:max_show]
        # Categories silenced in the edge list (structural / noise)
        _SILENT_CATS = frozenset({"date", "tag", "metric", "concept"})
        lines = []
        for i, se in enumerate(show_edges):
            e = se.edge
            src = self.index.node_label(e.source_uid)
            tgt = self.index.node_label(e.target_uid)
            src_cat = self.index.node_category(e.source_uid)
            tgt_cat = self.index.node_category(e.target_uid)
            src_part = f"{src} [{src_cat}]" if src_cat not in _SILENT_CATS else src
            tgt_part = f"{tgt} [{tgt_cat}]" if tgt_cat not in _SILENT_CATS else tgt
            t = ""
            if e.start:
                t = f" ({e.start}"
                if e.end:
                    t += f"–{e.end}"
                t += ")"
            lines.append(f"{i+1}. {src_part} → {e.relation} → {tgt_part}{t}")
        edge_list = "\n".join(lines)

        year_line = f"\nTarget year: {year}" if year else ""

        # Adaptive guidance based on question keywords
        low_q = question.lower()

        # For ordering/comparison questions extract the two compared entities
        # and build explicit guidance to pick dates for BOTH sides.
        if query_type in ("first_occurrence", "last_occurrence"):
            # Try to find "X or Y" pattern in the question
            or_match = re.search(r":\s*(.+?)\s+or\s+(.+?)(?:\?|$)", question, re.IGNORECASE)
            if or_match:
                entity_a = or_match.group(1).strip().rstrip(",;")
                entity_b = or_match.group(2).strip().rstrip(",;?")
                guidance = (
                    f"You need start/end dates for TWO events to answer this ordering question.\n"
                    f"Event A: '{entity_a}'\n"
                    f"Event B: '{entity_b}'\n"
                    "Pick ONLY edges whose source or target is directly one of these two events "
                    "AND that have a start or end date. Ignore edges about unrelated entities."
                )
            elif any(w in low_q for w in ("before", "after", "earlier", "later", "first", "began")):
                guidance = (
                    "This is a temporal ordering question. Pick edges that show WHEN each named "
                    "event or entity started, ended, or was active — especially edges with explicit "
                    "start/end dates (date ranges). Ignore edges about unrelated entities. "
                    "Ensure you pick at least one edge for EACH of the two compared events."
                )
            else:
                guidance = (
                    "Focus on edges that show WHEN events started, ended, or occurred "
                    "(look at the date ranges). Also pick edges showing direct ordering "
                    "(e.g. 'preceded', 'followed_by'). Pick ALL edges with relevant dates."
                )
        elif any(w in low_q for w in ("who", "whom", "whose")):
            # Try to add country/org context so small LLM picks the right person
            country_hint = ""
            for kw in ["united states", "united kingdom", "uk", "france", "germany",
                       "japan", "china", "russia", "india", "brazil", "canada",
                       "apple", "google", "microsoft", "openai"]:
                if kw in question.lower():
                    country_hint = (
                        f" ONLY select edges that specifically mention '{kw}' in the source, "
                        f"target, or relation. Ignore edges about other countries or organizations."
                    )
                    break
            guidance = (
                "Pick edges where a PERSON is the source or target and the relation "
                f"matches the role asked about (e.g. served_as, president_of, ceo_of).{country_hint}"
            )
        elif any(w in low_q for w in ("before", "after", "end", "start", "began")):
            guidance = (
                "Focus on edges that show WHEN events started, ended, or occurred "
                "(look at the date ranges). Also pick edges showing direct ordering "
                "(e.g. 'preceded', 'followed_by'). Pick ALL edges with relevant dates."
            )
        else:
            guidance = (
                "Pick edges whose source/target entities and date ranges "
                "directly answer the question. Prefer edges with specific entities over abstract ones."
            )

        # Append entity-specificity note: always remind LLM which exact entities
        # the question asks about so it doesn't pick edges for similar-but-wrong ones
        # (e.g. Fukushima vs Chernobyl, David Cameron vs Shinzo Abe).
        plan_ents_for_prompt = [e for e in (plan_entities or []) if len(e) > 2]
        if plan_ents_for_prompt:
            ent_list = ", ".join(f"'{e}'" for e in plan_ents_for_prompt[:4])
            guidance += (
                f"\nIMPORTANT: This question is specifically about {ent_list}. "
                "If you see edges about similar but DIFFERENT entities, do NOT pick them."
            )

        # One-shot example so the 0.8B model knows the exact expected format
        _ONESHOT = (
            "Example:\n"
            "Question: Who was the US president in 2009?\n"
            "1. Barack Obama [person] → president_of → United States [org] (2009–2017)\n"
            "2. George W. Bush [person] → president_of → United States [org] (2001–2009)\n"
            "3. United States → gdp_growth → 2.3% (2009)\n"
            '{"pick":[1],"need_more":false}\n\n'
        )

        prompt = (
            f"{_ONESHOT}"
            f"Question: {question}{year_line}\n\n"
            f"Graph edges:\n{edge_list}\n\n"
            f"{guidance}\n"
            "Select ALL edges that help answer the question.\n"
            "Reply ONLY with a JSON object:\n"
            '{"pick":[1,7],"need_more":false}'
        )

        raw = self._call_llm(
            prompt,
            system="You pick relevant graph edges. Output ONLY valid JSON with edge numbers.",
            max_tokens=100,
        )

        # Parse response
        selected: list[int] = []
        need_more = False
        parsed_json_ok = False
        try:
            m = re.search(r"\{[^}]*\}", raw, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                parsed_json_ok = True
                raw_picks = obj.get("pick") or obj.get("picks") or obj.get("selected") or []
                for v in raw_picks:
                    idx0 = int(v) - 1  # convert 1-based → 0-based
                    if 0 <= idx0 < len(show_edges):
                        selected.append(idx0)
                need_more = bool(obj.get("need_more", False))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Only fall back to top 5 when JSON parsing completely failed.
        # If LLM returned valid JSON with empty picks ({"pick":[]}),
        # respect that as intentional "nothing relevant" signal.
        if not selected and not parsed_json_ok:
            selected = list(range(min(5, len(show_edges))))

        return {
            "selected_indices": selected,
            "need_more": need_more,
            "stage1_prompt": prompt,
            "stage1_raw": raw,
        }

    def _stage2_expand(
        self,
        question: str,
        selected_edges: list,  # List[GroundingScoredEdge]
        year: int | None,
    ) -> Dict[str, Any]:
        """Continuation: show currently selected edges, ask if enough or need more.

        If LLM suggests more entities, expand from their nodes.
        """
        lines = []
        for i, se in enumerate(selected_edges[:10]):
            e = se.edge
            src = self.index.node_label(e.source_uid)
            tgt = self.index.node_label(e.target_uid)
            t = f" ({e.start})" if e.start else ""
            lines.append(f"  {src} → {e.relation} → {tgt}{t}")
        edges_text = "\n".join(lines) if lines else "  (none)"

        prompt = (
            f"Question: {question}\n"
            f"Year: {year}\n\n"
            f"Selected evidence:\n{edges_text}\n\n"
            "Is this enough to answer? If not, name specific entities to search for.\n\n"
            "Reply ONLY JSON:\n"
            '{"sufficient":true,"entities":[]}'
        )

        raw = self._call_llm(
            prompt,
            system="You review graph evidence. Output ONLY valid JSON.",
            max_tokens=120,
        )

        try:
            m = re.search(r"\{[^}]*\}", raw, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                return {
                    "sufficient": bool(obj.get("sufficient", True)),
                    "entities": [str(v) for v in (obj.get("entities") or [])],
                    "stage2_prompt": prompt,
                    "stage2_raw": raw,
                }
        except (json.JSONDecodeError, ValueError):
            pass

        return {"sufficient": True, "entities": [],
                "stage2_prompt": prompt, "stage2_raw": raw}

    # ── Main retrieval: 2-stage LLM-in-the-loop ───────────────────

    @staticmethod
    def _edge_to_evidence(index, edge):
        """Convert a GraphEdge into the evidence dict the benchmark expects."""
        return {
            "edge_uid": edge.edge_uid,
            "source": index.node_label(edge.source_uid),
            "source_category": index.node_category(edge.source_uid),
            "target": index.node_label(edge.target_uid),
            "target_category": index.node_category(edge.target_uid),
            "relation": edge.relation,
            "start": edge.start,
            "end": edge.end,
            "edge_type": edge.edge_type,
            "support_count": edge.support_count,
            "source_row_ids": edge.source_row_ids[:10],
        }

    def _build_answer_from_edges(
        self,
        question: str,
        plan: Dict[str, Any],
        final_edges: list,  # List[GroundingScoredEdge]
    ) -> GraphAnswer:
        """Build a GraphAnswer directly from curated edges.

        Bypasses the old entity-UID re-filtering in answer_from_plan().
        """
        year = plan.get("year")
        query_type = plan.get("query_type") or "state_at_time"

        if not final_edges:
            return GraphAnswer(
                question_type=query_type,
                answer_text="Retrieval did not return supporting graph edges.",
                evidence=[],
                confidence=0.1,
            )

        # Only filter purely structural metadata relations (tag links).
        # Keep spans_year, dated, etc. — they carry entity/temporal info.
        _DISPLAY_FILTER = frozenset({"tag_related_to", "inferred_tag"})

        # Separate structural from factual edges
        # gold_fact edges are explicitly validated triples — treat equally with base
        factual = [
            se for se in final_edges
            if se.edge.relation.lower() not in _DISPLAY_FILTER
            and se.edge.edge_type in ("base", "gold_fact")
        ]
        if not factual:
            factual = [
                se for se in final_edges
                if se.edge.relation.lower() not in _DISPLAY_FILTER
            ]
        display = factual if factual else final_edges

        # For ordering/comparison queries, balance evidence across plan entities
        # so both events get representation (not just the top-scoring entity).
        evidence_limit = 18
        if query_type in ("first_occurrence", "last_occurrence"):
            plan_entities = [e.lower() for e in (plan.get("entities") or []) if len(e) > 2]
            if len(plan_entities) >= 2:
                entity_groups: dict = {ent: [] for ent in plan_entities}
                other_edges = []
                for se in display:
                    lbl = (
                        self.index.node_label(se.edge.source_uid) + " "
                        + self.index.node_label(se.edge.target_uid)
                    ).lower()
                    matched = False
                    for ent in plan_entities:
                        if ent in lbl or any(w in lbl for w in ent.split() if len(w) > 3):
                            entity_groups[ent].append(se)
                            matched = True
                            break
                    if not matched:
                        other_edges.append(se)
                balanced: list = []
                for ent in plan_entities:
                    balanced.extend(entity_groups[ent][:4])
                remainder = evidence_limit - len(balanced)
                if remainder > 0:
                    balanced.extend(other_edges[:remainder])
                display = balanced[:evidence_limit] if balanced else display[:evidence_limit]

        # Build summary
        summary_parts = []
        for se in display[:5]:
            e = se.edge
            src = self.index.node_label(e.source_uid)
            tgt = self.index.node_label(e.target_uid)
            summary_parts.append(f"{src} {e.relation} {tgt}")

        # Canonical entities
        canonical: list[str] = []
        for se in display[:evidence_limit]:
            for uid in (se.edge.source_uid, se.edge.target_uid):
                label = self.index.node_label(uid)
                cat = self.index.node_category_by_uid.get(uid, "")
                if cat in ("date", "year") or label.isdigit():
                    continue
                if label not in canonical:
                    canonical.append(label)

        time_text = f" in {year}" if year is not None else ""
        canonical_hint = f" Key entities: {', '.join(canonical[:6])}." if canonical else ""
        answer_text = f"Evidence{time_text}: " + "; ".join(summary_parts) + f".{canonical_hint}"

        evidence = [self._edge_to_evidence(self.index, se.edge) for se in display[:evidence_limit]]
        confidence = min(0.9, 0.3 + 0.1 * len(factual))

        return GraphAnswer(
            question_type=query_type,
            answer_text=answer_text,
            evidence=evidence,
            confidence=confidence,
        )

    def _answer_self_contained(self, question: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Bypass graph retrieval: answer pure temporal-logic problems directly.

        The question provides its own facts (dates/phases/sequence) — we just need
        the LLM to reason over them.
        """
        prompt = (
            "You are a precise temporal reasoning engine.\n"
            "The following question describes a timeline or schedule fully within its own text.\n"
            "Do NOT use any external knowledge. Reason only from the facts stated in the question.\n\n"
            f"Question: {question}\n\n"
            "Rules:\n"
            "- Read every date, range, day-name, and month mentioned.\n"
            "- Map each event/phase to its time window.\n"
            "- Answer with the single shortest correct phrase (one word, a date, or a short noun phrase).\n"
            "- If yes/no, reply Yes or No.\n"
            "- Do NOT add explanations, bullet points, or markdown.\n"
            "Answer:"
        )
        raw = self._call_llm(
            prompt,
            system=(
                "You solve temporal logic puzzles. "
                "Answer with the minimal correct phrase. No explanations."
            ),
            max_tokens=40,
        )
        # Strip common preamble the model sometimes emits
        answer_text = re.sub(
            r"^(answer\s*:\s*|the answer is\s*|answer is\s*)",
            "",
            raw.strip(),
            flags=re.IGNORECASE,
        ).strip()

        return {
            "question": question,
            "intent": "self_contained",
            "plan": plan,
            "answer": GraphAnswer(
                question_type="self_contained",
                answer_text=answer_text,
                evidence=[],
                confidence=0.85,
            ),
            "mermaid": "",
            "grounding": {"enabled": False, "self_contained_bypass": True},
            "stage1_selection": None,
            "all_candidate_edges": [],
            "continuation": None,
        }

    def _retrieve(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        plan = dict(payload.get("plan") or {})
        question = str(payload.get("question") or "")

        low_mem_mode = str(os.getenv("GRAPH_LOW_MEM_MODE", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        default_paraphrase = "0" if low_mem_mode else "1"
        enable_paraphrase = str(os.getenv("GRAPH_ENABLE_PARAPHRASE", default_paraphrase) or default_paraphrase).strip().lower() in {"1", "true", "yes", "on"}

        # ── Self-contained bypass ──────────────────────────────────────────────
        if plan.get("query_type") == "self_contained":
            return self._answer_self_contained(question, plan)

        # ── Step 1: Full graph-native retrieval (embed → walk → score) ──
        if low_mem_mode:
            sr = self.semantic_grounder.retrieve_subgraph(
                question,
                plan,
                top_k_nodes=6,
                top_k_tags=4,
                top_k_relations=3,
                max_returned_edges=30,
            )
        else:
            sr = self.semantic_grounder.retrieve_subgraph(question, plan)
        
        # Ground locations
        plan_locations = plan.get("locations") or []
        location_uids = set()
        if plan_locations:
            try:
                for loc in plan_locations:
                    if low_mem_mode:
                        loc_results = self.semantic_grounder.retrieve_subgraph(
                            str(loc),
                            plan,
                            top_k_nodes=4,
                            top_k_tags=2,
                            top_k_relations=2,
                            max_returned_edges=20,
                        )
                    else:
                        loc_results = self.semantic_grounder.retrieve_subgraph(str(loc), plan)
                    for l_hit in loc_results.node_hits[:3]:
                        location_uids.add(l_hit["uid"])
            except Exception:
                pass

        # ── Step 1b: Hybrid candidate merge via Reciprocal Rank Fusion (RRF) ──
        # RRF is superior to max-score merge: edges appearing highly in BOTH Route A
        # (graph walk) and Route B (semantic edge search) receive a compounded boost.
        # This avoids the need for score normalization across the two different scales.
        try:
            _sem_scored = self.semantic_index.search(
                question,
                top_k=(24 if low_mem_mode else 60),
                year=plan.get("year"), 
                query_type=plan.get("query_type", "point_in_time")
            )
            _RRF_K = 60  # Standard RRF smoothing constant

            # Build rank tables for both lists (rank is 1-indexed)
            _route_a_ranks: dict[str, int] = {
                se.edge.edge_uid: i + 1
                for i, se in enumerate(sr.scored_edges)
            }
            _route_b_ranks: dict[str, int] = {
                se.edge.edge_uid: i + 1
                for i, se in enumerate(_sem_scored)
            }

            # Collect all edge UIDs from both routes
            _all_uids: set[str] = set(_route_a_ranks) | set(_route_b_ranks)

            # Build a lookup from uid → ScoredEdge (prefer Route A object structure)
            _uid_to_edge: dict[str, Any] = {}
            for se in sr.scored_edges:
                _uid_to_edge[se.edge.edge_uid] = se
            for se in _sem_scored:
                uid = se.edge.edge_uid
                if uid not in _uid_to_edge:
                    _uid_to_edge[uid] = GroundingScoredEdge(edge=se.edge, score=se.score)

            CLUSTER_CAP = {
                "state_at_time":    4,
                "point_in_time":    4,
                "first_occurrence": 6,
                "ordering":         99,  # By-pass cluster cap to allow wide candidate pools
                "node_path":        99,
                "summary":          6,
            }
            max_per_source = CLUSTER_CAP.get(plan.get("query_type", ""), 6)

            # Compute initial RRF score for each uid
            fused_scores = {}
            for uid in _all_uids:
                r_a = _route_a_ranks.get(uid, len(_route_a_ranks) + _RRF_K)
                r_b = _route_b_ranks.get(uid, len(_route_b_ranks) + _RRF_K)
                fused_scores[uid] = 1.0 / (_RRF_K + r_a) + 1.0 / (_RRF_K + r_b)

            # Post-RRF Boosts
            RELATION_WHITELIST = {
                "state_at_time": ["led_by", "head_of_state", "president_of", "governed_by", "served_as", "ruled_by"],
                "first_occurrence": ["founded", "established", "launched", "premiered", "series_premiere", "introduced"],
            }
            whitelist = RELATION_WHITELIST.get(plan.get("query_type", ""))

            for uid, score in fused_scores.items():
                edge = _uid_to_edge[uid].edge
                
                # Relation boost
                if whitelist and edge.relation in whitelist:
                    fused_scores[uid] += 0.15
                
                # Location boost
                if location_uids:
                    if edge.source_uid in location_uids or edge.target_uid in location_uids:
                        fused_scores[uid] += 0.05

            # Sort and apply cluster cap
            sorted_uids = sorted(fused_scores, key=fused_scores.get, reverse=True)
            from collections import defaultdict
            source_counts = defaultdict(int)
            capped_uids = []
            
            for uid in sorted_uids:
                edge = _uid_to_edge[uid].edge
                cluster_key = (edge.source_uid, edge.relation)
                if source_counts[cluster_key] < max_per_source:
                    capped_uids.append(uid)
                    source_counts[cluster_key] += 1

            # Build final scored edges with their newly boosted fused scores (scaling up for readability)
            sr.scored_edges = [
                GroundingScoredEdge(edge=_uid_to_edge[uid].edge, score=fused_scores[uid] * 100.0) 
                for uid in capped_uids
            ]
        except Exception as e:
            print("RRF hybrid merge failed:", e)

        # ── Step 1c: Paraphrase aggregation for ALL query types ──
        # Generate 2 LLM paraphrases of every question, retrieve subgraphs for
        # each, and merge all scored-edge pools (keeping the highest score when
        # the same edge appears in multiple results).  This gives 3 different
        # perspectives on the candidate pool before any filtering.
        _query_type_tmp = plan.get("query_type", "")
        if enable_paraphrase:
            _para_prompt = (
                f"Rewrite the following question in 2 different ways that preserve"
                f" the exact same meaning. Output ONLY a JSON array with exactly 2 strings.\n"
                f"Question: {question}"
            )
            _para_raw = self._call_llm(_para_prompt, max_tokens=150)
            _paraphrases: list[str] = []
            try:
                _arr = json.loads(_para_raw.strip())
                if isinstance(_arr, list):
                    _paraphrases = [str(p) for p in _arr[:2] if str(p).strip()]
            except Exception:
                pass
            _merged: dict[str, Any] = {se.edge.edge_uid: se for se in sr.scored_edges}
            for _pq in _paraphrases:
                try:
                    _psr = self.semantic_grounder.retrieve_subgraph(_pq, plan)
                    for _pse in _psr.scored_edges:
                        _uid = _pse.edge.edge_uid
                        if _uid not in _merged or _pse.score > _merged[_uid].score:
                            _merged[_uid] = _pse
                except Exception:
                    pass
            # Rebuild sr.scored_edges from the merged pool (sorted by score desc)
            sr.scored_edges = sorted(_merged.values(), key=lambda x: -x.score)

        grounding = {
            "enabled": sr.grounding_enabled,
            "entity_uids": sr.entity_uids,
            "relation_hint": sr.relation_hints[0] if sr.relation_hints else None,
            "node_hits": sr.node_hits,
            "tag_hits": sr.tag_hits,
            "relation_hits": sr.relation_hits,
        }

        # ── Step 1b: Entity / relation pre-filter ──
        # Keep edges whose src/tgt overlaps plan entities OR whose relation
        # matches the relation hint.  Falls back to all candidates if < 5 survive.
        _raw_ents = plan.get("entities") or []
        plan_entities_raw: list[str] = [
            e.get("name", "") if isinstance(e, dict) else str(e)
            for e in _raw_ents
        ]
        rel_hint_raw: str = (plan.get("relation_hint") or "").lower().replace("_", " ")
        query_type = plan.get("query_type", "")

        # ── LLM entity cleaning ──────────────────────────────────────────────
        # Strip role words, possessives, and noise tokens from extracted entity
        # names so they can be used for keyword / embedding lookup.
        # Examples:  "Barack Obama president" → "Barack Obama"
        #            "Donald Trump's"         → "Donald Trump"
        #            "release"                → (drop — not a real entity)
        # SKIP for ordering queries: the contrasting descriptive phrases
        # (e.g. "DVD rental service" vs "major streaming platform") are the
        # two events being compared and must be preserved for COMPUTED_ANSWER.
        _skip_cleaning = query_type in {"first_occurrence", "last_occurrence"}
        if plan_entities_raw and not _skip_cleaning:
            _clean_prompt = (
                "You are an entity name normalizer.  Given a list of entity strings "
                "extracted from a question, return a JSON array with the cleaned proper-noun "
                "entity names only.  Rules:\n"
                "- Remove trailing or leading role/action words (president, pandemic, film, book, release, hosting, etc.)\n"
                "- Remove possessive 's and apostrophes\n"
                "- Remove generic nouns that are NOT the actual entity (e.g. 'release', 'term', 'country', 'hosting')\n"
                "- Keep multi-word proper names intact (e.g. 'Harry Potter', 'Barack Obama', 'COVID-19')\n"
                "- Drop entries that after cleaning are empty, generic, or shorter than 3 chars\n"
                "- Output ONLY a JSON array of strings, no explanation\n\n"
                "Examples:\n"
                "Input: [\"Barack Obama president\", \"Donald Trump's\"]\n"
                "Output: [\"Barack Obama\", \"Donald Trump\"]\n\n"
                "Input: [\"COVID-19 pandemic\", \"release\", \"ChatGPT\"]\n"
                "Output: [\"COVID-19\", \"ChatGPT\"]\n\n"
                "Input: [\"Harry Potter book\", \"Harry Potter film\"]\n"
                "Output: [\"Harry Potter\"]\n\n"
                f"Input: {json.dumps(plan_entities_raw)}\n"
                "Output:"
            )
            try:
                _clean_raw = self._call_llm(_clean_prompt, max_tokens=120)
                _clean_arr = json.loads(_clean_raw.strip())
                if isinstance(_clean_arr, list) and _clean_arr:
                    _cleaned = [str(x).strip() for x in _clean_arr if str(x).strip() and len(str(x).strip()) >= 3]
                    if _cleaned:
                        plan_entities_raw = _cleaned
            except Exception:
                pass  # keep original if LLM/parse fails

        def _token_overlap(label: str, entity: str) -> bool:
            label_l = label.lower()
            ent_l = entity.lower()
            if ent_l in label_l or label_l in ent_l:
                return True
            ent_words = {w for w in ent_l.split() if len(w) > 2}
            lbl_words = set(label_l.split())
            return bool(ent_words & lbl_words)

        def _edge_passes_prefilter(se) -> bool:
            e = se.edge
            src_lbl = self.index.node_label(e.source_uid)
            tgt_lbl = self.index.node_label(e.target_uid)
            # Node match: any plan entity appears in src or tgt label
            for ent in plan_entities_raw:
                if _token_overlap(src_lbl, ent) or _token_overlap(tgt_lbl, ent):
                    return True
            # Relation match: relation overlaps with hint tokens
            if rel_hint_raw:
                hint_words = {w for w in rel_hint_raw.split("_") if len(w) > 2}
                rel_words = set(e.relation.lower().replace("_", " ").split())
                if hint_words & rel_words:
                    return True
            return False

        # ── Year extraction (MUST run before prefilter so date-rescue works) ──
        # Priority: plan.year → year in entity names → year in question text.
        # Guard: if a year is used as a title/reference (e.g. "1984 novel",
        # "2001 film", "1980 Olympics") it should NOT become the temporal anchor.
        target_year = plan.get("year")
        # query_type already assigned above (before entity cleaning)

        _TITLE_WORD_RE = re.compile(
            r'^(novel|book|poem|play|film|movie|song|album|game|series|show|'
            r'olympics|olympic games|world cup|championship|ep|single)',
            re.IGNORECASE,
        )
        _WRITTEN_RE = re.compile(
            r'\b(written|published|released|written by|based on)\b', re.IGNORECASE
        )

        def _is_title_year(text: str, m) -> bool:
            """Return True if this year match is a title label, not a temporal anchor."""
            if m is None:
                return False
            after = text[m.end():].strip()[:30]
            before = text[:m.start()].strip()[-30:]
            # Year immediately followed by a media/event descriptor → title
            if _TITLE_WORD_RE.match(after):
                return True
            # Question is about when something was written/published and year looks like title
            if _WRITTEN_RE.search(text) and (_TITLE_WORD_RE.search(before) or _TITLE_WORD_RE.search(after)):
                return True
            return False

        if target_year is None:
            for _ent_str in plan_entities_raw:
                _ym = re.search(r'\b(1\d{3}|2\d{3})\b', str(_ent_str))
                if _ym:
                    # Re-find in question text for title-word proximity check
                    _q_match = re.search(r'\b' + _ym.group(1) + r'\b', question)
                    if not _is_title_year(question, _q_match or _ym):
                        target_year = int(_ym.group(1))
                    break
        if target_year is None:
            _ym = re.search(r'\b(1\d{3}|2\d{3})\b', question)
            if _ym and not _is_title_year(question, _ym):
                target_year = int(_ym.group(1))

        # ── Date-exact rescue: edges whose start/end exactly spans target_year
        # bypass the entity prefilter — date is the #1 filter criterion.
        def _is_date_exact_match(se) -> bool:
            if target_year is None:
                return False
            e_start = _parse_edge_year(se.edge.start)
            e_end = _parse_edge_year(se.edge.end)
            if e_start is not None and e_end is not None:
                return e_start <= target_year <= e_end
            if e_start is not None:
                return e_start == target_year
            return False

        prefiltered = [
            se for se in sr.scored_edges
            if _edge_passes_prefilter(se) or _is_date_exact_match(se)
        ]
        # Fallback: if filter is too aggressive, use all candidates
        # Always a mutable list so entity-rescue can append to it.
        candidate_edges: list = list(prefiltered) if len(prefiltered) >= 3 else list(sr.scored_edges)

        # ── Entity coverage rescue ────────────────────────────────────────────
        # For each plan entity, count how many candidate edges mention it AND
        # have a date.  If an entity has fewer than 3 dated edges, re-query the
        # grounder using the entity name alone and inject up to 5 dated edges.
        # This fixes cases where the initial embedding query is dominated by
        # one entity (e.g. COVID) and misses others (e.g. ChatGPT).
        _seen_uids: set[str] = {se.edge.edge_uid for se in candidate_edges}

        def _edge_mentions_entity(se, ent: str) -> bool:
            src_lbl = self.index.node_label(se.edge.source_uid)
            tgt_lbl = self.index.node_label(se.edge.target_uid)
            return _token_overlap(src_lbl, ent) or _token_overlap(tgt_lbl, ent)

        def _has_date(se) -> bool:
            return _parse_edge_year(se.edge.start) is not None

        for _rescue_ent in plan_entities_raw:
            _rescue_ent = str(_rescue_ent)
            if not _rescue_ent or len(_rescue_ent) < 3:
                continue
            _dated_count = sum(
                1 for se in candidate_edges
                if _edge_mentions_entity(se, _rescue_ent) and _has_date(se)
            )
            if _dated_count < 3:
                # Re-query grounder focused on this entity
                _injected = 0
                try:
                    _rescue_sr = self.semantic_grounder.retrieve_subgraph(
                        _rescue_ent, plan
                    )
                    for _rse in sorted(_rescue_sr.scored_edges,
                                       key=lambda x: (
                                           0 if _has_date(x) else 1,
                                           -x.score
                                       )):
                        if _injected >= 5:
                            break
                        if _rse.edge.edge_uid in _seen_uids:
                            continue
                        if not (_edge_mentions_entity(_rse, _rescue_ent) and _has_date(_rse)):
                            continue
                        candidate_edges.append(_rse)
                        _seen_uids.add(_rse.edge.edge_uid)
                        _injected += 1
                except Exception:
                    pass

                # Keyword rescue: if embedding rescue still found no dated edges,
                # scan the index for substring matches directly.
                # This catches entities that rank poorly in embedding space (e.g. Harry Potter).
                if _injected == 0:
                    try:
                        _kw_node_uids = self.index.resolve_node_uids(_rescue_ent, max_hits=20)
                        _kw_injected = 0
                        for _kw_uid in _kw_node_uids:
                            _kw_edges = (
                                list(self.index.outgoing_edge_uids.get(_kw_uid, []))
                                + list(self.index.incoming_edge_uids.get(_kw_uid, []))
                            )
                            for _kw_eid in _kw_edges:
                                if _kw_injected >= 5:
                                    break
                                if _kw_eid in _seen_uids:
                                    continue
                                _kw_edge = self.index.edge_by_uid.get(_kw_eid)
                                if _kw_edge is None:
                                    continue
                                if _parse_edge_year(_kw_edge.start) is None:
                                    continue
                                candidate_edges.append(GroundingScoredEdge(edge=_kw_edge, score=0.35))
                                _seen_uids.add(_kw_eid)
                                _kw_injected += 1
                            if _kw_injected >= 5:
                                break
                    except Exception:
                        pass

        # ── Year-prioritised direct selection (no LLM stage-1 for dated lookups) ──
        # For state_at_time / simple entity lookups where we have a target year,
        # rank ALL candidates by how well their date range overlaps the query year
        # and take the top-5.  This avoids asking the 0.8B LLM to choose among
        # edges it can't interpret reliably and guarantees chronologically correct
        # evidence reaches the final prompt.
        #
        # Scoring priority (lower = better):
        #   0 – edge interval strictly contains the query year
        #   1 – edge start year == query year (point-event in correct year)
        #   2 – edge start within ±1 year of query year
        #   3 – edge has no date (may still be relevant entity edge)
        #   9 – edge date is outside the query year (deprioritised)

        _YR_LOOKUP_TYPES = {"state_at_time", "state_during_interval", ""}

        def _year_priority(se) -> int:
            """Priority 0 = best hit.
            0 – interval contains year  OR  start == year  (equally good)
            1 – no date at all         (uncertain but entity-relevant; beats near-miss)
            2 – start within ±1 year   (near miss)
            9 – date exists but clearly outside year (wrong era)
            """
            if target_year is None:
                return 0  # no year filter → don't reorder
            e_start = _parse_edge_year(se.edge.start)
            e_end = _parse_edge_year(se.edge.end)
            if e_start is not None and e_end is not None:
                return 0 if e_start <= target_year <= e_end else 9
            if e_start is not None:
                dist = abs(e_start - target_year)
                if dist == 0:
                    return 0   # exact start == same priority as interval containment
                if dist <= 1:
                    return 2   # near miss — worse than undated
                return 9       # clearly wrong era
            # No date: uncertain but keeps entity relevance — better than ±1 near miss
            return 1

        # ── Deduplicate candidate_edges after all rescue injections ─────────────
        import hashlib, unicodedata
        def _normalize_text(text):
            return unicodedata.normalize("NFKD", str(text)).lower().strip()
            
        _dedup_seen: set[str] = set()
        _deduped: list = []
        for _dse in candidate_edges:
            # Instead of UID grouping, dedup perfectly matching (source + relation + target) to dodge MMR over-filtering
            # GraphEdge does not have a .text attribute, so we reconstruct the semantic string
            src_lbl = self.index.node_label(_dse.edge.source_uid) or ""
            tgt_lbl = self.index.node_label(_dse.edge.target_uid) or ""
            pseudo_text = f"{src_lbl} {_dse.edge.relation} {tgt_lbl}"
            h = hashlib.md5(_normalize_text(pseudo_text).encode()).hexdigest()
            
            if h not in _dedup_seen:
                _dedup_seen.add(h)
                _deduped.append(_dse)
        candidate_edges = _deduped

        # ── Build step-by-step debug trace ────────────────────────────────────
        def _edge_dict(se, extra: dict | None = None) -> dict:
            """Convert a ScoredEdge to a compact debug dict."""
            d = {
                "src": self.index.node_label(se.edge.source_uid),
                "src_cat": self.index.node_category(se.edge.source_uid),
                "rel": se.edge.relation,
                "tgt": self.index.node_label(se.edge.target_uid),
                "tgt_cat": self.index.node_category(se.edge.target_uid),
                "start": se.edge.start,
                "end": se.edge.end,
                "score": round(se.score, 4),
                "yr_priority": _year_priority(se),
            }
            if extra:
                d.update(extra)
            return d

        debug_steps: list[dict] = []
        debug_steps.append({
            "step": 1,
            "name": "grounding",
            "plan_entities": plan_entities_raw,
            "relation_hint": rel_hint_raw,
            "target_year": target_year,
            "query_type": query_type,
            "node_hits": [
                {"label": h.get("normalized_label"), "score": round(float(h.get("score", 0)), 4)}
                for h in (grounding.get("node_hits") or [])[:8]
            ],
            "tag_hits": [
                {"tag": h.get("normalized_tag"), "score": round(float(h.get("score", 0)), 4)}
                for h in (grounding.get("tag_hits") or [])[:4]
            ],
            "raw_scored_edges_top5": [_edge_dict(se) for se in sr.scored_edges[:5]],
        })
        # Build per-entity prefilter breakdown for step 2
        _prefilter_by_entity: dict[str, list] = {}
        for _pse in prefiltered[:30]:
            src_l = self.index.node_label(_pse.edge.source_uid)
            tgt_l = self.index.node_label(_pse.edge.target_uid)
            for _pent in plan_entities_raw:
                if _token_overlap(src_l, _pent) or _token_overlap(tgt_l, _pent):
                    _prefilter_by_entity.setdefault(_pent, []).append(
                        {"src": src_l, "rel": _pse.edge.relation, "tgt": tgt_l,
                         "start": _pse.edge.start, "end": _pse.edge.end}
                    )
                    break
        debug_steps.append({
            "step": 2,
            "name": "candidate_pool",
            "total_from_embedding": len(sr.scored_edges),
            "after_entity_prefilter": len(prefiltered),
            "after_rescue": len(candidate_edges),
            "using_fallback": len(prefiltered) < 3,
            "prefilter_by_entity": {ent: hits[:4] for ent, hits in _prefilter_by_entity.items()},
            "top10_candidates": [_edge_dict(se) for se in candidate_edges[:10]],
        })

        stage1: Dict[str, Any]
        if query_type in _YR_LOOKUP_TYPES and target_year is not None:
            # Sort candidates by year-overlap priority (date is primary filter),
            # then by retrieval score.  Take top-10 so more date-relevant edges
            # are available to the answer builder.
            year_sorted = sorted(
                enumerate(candidate_edges),
                key=lambda x: (_year_priority(x[1]), -x[1].score),
            )
            # Priority 0 = exact year hit (interval or start==year)
            # Priority 1 = no date (entity-relevant but undated)
            # Priority 2 = ±1 near miss
            # Priority 9 = wrong era
            
            def _apply_mmr(pool: list, top_n: int, lambda_val: float = 0.5) -> list:
                if not pool: return []
                pool_copy = list(pool)
                selected = []
                max_s = max(x[1].score for x in pool_copy)
                min_s = min(x[1].score for x in pool_copy)
                val_range = max_s - min_s if max_s > min_s else 1.0

                def _sim(e1, e2) -> float:
                    s = 0.0
                    if e1.relation == e2.relation: s += 0.4
                    if e1.target_uid == e2.target_uid: s += 0.3
                    if e1.source_uid == e2.source_uid: s += 0.3
                    return s

                for _ in range(min(top_n, len(pool_copy))):
                    best_idx = 0
                    best_mmr = -9999.0
                    for i, cand in enumerate(pool_copy):
                        ns = (cand[1].score - min_s) / val_range
                        max_sim = 0.0
                        if selected:
                            max_sim = max(_sim(cand[1].edge, sel[1].edge) for sel in selected)
                        
                        mmr = lambda_val * ns - (1 - lambda_val) * max_sim
                        mmr -= _year_priority(cand[1]) * 10.0  # Heavily prioritize temporal bucket
                        
                        if mmr > best_mmr:
                            best_mmr = mmr
                            best_idx = i
                    selected.append(pool_copy.pop(best_idx))
                return selected

            # Adaptive MMR λ by query type:
            MMR_LAMBDA = {
                "state_at_time":    0.65,
                "point_in_time":    0.65,
                "first_occurrence": 0.40,
                "ordering":         0.40,
                "summary":          0.40,
            }
            _mmr_lambda = MMR_LAMBDA.get(query_type, 0.55)
            top_year = _apply_mmr(year_sorted, top_n=20, lambda_val=_mmr_lambda)

            selected_idx_set: set[int] = {i for i, _ in top_year}

            # Record year-ranking details in debug trace
            # (exact_year / undated / near_miss replaced by MMR; recompute bucket counts inline)
            _dbg_exact = sum(1 for _, se in year_sorted if _year_priority(se) == 0)
            _dbg_undated = sum(1 for _, se in year_sorted if _year_priority(se) == 1)
            _dbg_near = sum(1 for _, se in year_sorted if _year_priority(se) == 2)
            debug_steps.append({
                "step": 3,
                "name": "year_priority_selection",
                "target_year": target_year,
                "exact_hits": _dbg_exact,
                "undated_hits": _dbg_undated,
                "near_miss_hits": _dbg_near,
                "ranked_edges": [
                    dict(_edge_dict(se), rank=rank + 1, selected=(idx in selected_idx_set))
                    for rank, (idx, se) in enumerate(year_sorted[:15])
                ],
                "total_selected": len(top_year),
            })

            stage1 = {
                "selected_indices": sorted(selected_idx_set),
                "need_more": False,
                "stage1_prompt": "(year-priority direct selection — no LLM)",
                "stage1_raw": f"year={target_year}, top_year_hits={len(top_year)}",
            }
        else:
            # ── Step 2: LLM filters edges (Stage 1) for ordering/other queries ──
            # Reorder candidate_edges to guarantee each plan entity has representation
            # in the top-20 slice that the LLM sees.  Strategy: for every entity in
            # plan_entities_raw, bubble up its best 3 dated edges to the front of the
            # list, then append remaining edges deduplicated.
            _entity_priority_uids: list[str] = []
            _ep_seen: set[str] = set()
            for _ep_ent in plan_entities_raw:
                if not _ep_ent or len(_ep_ent) < 3:
                    continue
                _ep_candidates = [
                    se for se in candidate_edges
                    if _edge_mentions_entity(se, _ep_ent) and _has_date(se)
                ]
                _ep_candidates.sort(key=lambda x: -x.score)
                for _ep_se in _ep_candidates[:3]:
                    if _ep_se.edge.edge_uid not in _ep_seen:
                        _ep_seen.add(_ep_se.edge.edge_uid)
                        _entity_priority_uids.append(_ep_se.edge.edge_uid)
            _uid_to_se = {se.edge.edge_uid: se for se in candidate_edges}
            _interleaved: list = [
                _uid_to_se[uid] for uid in _entity_priority_uids if uid in _uid_to_se
            ]
            # Sort remaining (un-prioritised) edges by composite score:
            # (1) date proximity, (2) entity coverage count, (3) embedding score
            _remaining_unsorted = [ce for ce in candidate_edges if ce.edge.edge_uid not in _ep_seen]
            _remaining_sorted = sorted(
                _remaining_unsorted,
                key=lambda _x: (
                    _year_priority(_x),
                    -sum(1 for _en in plan_entities_raw if _edge_mentions_entity(_x, _en)),
                    -_x.score,
                ),
            )
            _interleaved.extend(_remaining_sorted)
            stage1 = self._stage1_filter_edges(
                question, _interleaved, sr.year,
                query_type=query_type,
                plan_entities=plan.get("entities") or [],
            )
            llm_indices: list[int] = stage1.get("selected_indices") or []
            # Map indices back to original candidate_edges positions
            _interleaved_to_orig: dict[int, int] = {}
            for _ii, _ise in enumerate(_interleaved):
                for _oi, _ose in enumerate(candidate_edges):
                    if _ise.edge.edge_uid == _ose.edge.edge_uid:
                        _interleaved_to_orig[_ii] = _oi
                        break
            selected_idx_set = {_interleaved_to_orig.get(i, i) for i in llm_indices}

            # ── Safety net: add edges for uncovered plan entities ──
            # Strictly require year-overlap to avoid injecting wrong-era edges.
            # Use plan_entities_raw (already extracted string list) — NOT plan.get("entities")
            # which returns raw dicts and causes len(dict)>2 to silently fail.
            plan_entities_lc = [e.lower() for e in plan_entities_raw if len(e) > 2]

            def _labels_cover(entity: str, label_text: str) -> bool:
                if entity in label_text:
                    return True
                ent_words = {w for w in entity.split() if len(w) > 2}
                return bool(ent_words & set(label_text.split()))

            covered: set[str] = set()
            for i in selected_idx_set:
                if i >= len(candidate_edges):
                    continue
                e = candidate_edges[i].edge
                combo = (self.index.node_label(e.source_uid).lower() + " "
                         + self.index.node_label(e.target_uid).lower())
                for ent in plan_entities_lc:
                    if _labels_cover(ent, combo):
                        covered.add(ent)

            uncovered = [ent for ent in plan_entities_lc if ent not in covered]
            if uncovered:
                safety_added = 0
                for i in range(len(candidate_edges)):
                    if i in selected_idx_set or safety_added >= 4:
                        continue
                    e = candidate_edges[i].edge
                    src_lbl = self.index.node_label(e.source_uid).lower()
                    tgt_lbl = self.index.node_label(e.target_uid).lower()
                    labels = src_lbl + " " + tgt_lbl
                    relevant = any(_labels_cover(ent, labels) for ent in uncovered)
                    if not relevant:
                        rel_hint = plan.get("relation_hint", "") or ""
                        if rel_hint and rel_hint.lower() in e.relation.lower():
                            relevant = True
                    # Strict year check for safety-net edges: must overlap target year
                    if relevant and target_year is not None:
                        e_start = _parse_edge_year(e.start)
                        e_end = _parse_edge_year(e.end)
                        if e_start is not None and e_end is not None:
                            relevant = e_start <= target_year <= e_end
                        elif e_start is not None:
                            relevant = abs(e_start - target_year) <= 2
                        # No date → keep (entity match without dates)
                    if relevant:
                        selected_idx_set.add(i)
                        safety_added += 1

            # For yes/no existence questions the LLM tends to fail when only
            # 1-2 edges survive Stage-1 filtering. Ensure a minimum evidence
            # set biased toward temporal relevance.
            if query_type == "existence_in_time" and len(selected_idx_set) < 4:
                fill_budget = 4 - len(selected_idx_set)
                ranked_fill = sorted(
                    range(len(candidate_edges)),
                    key=lambda idx: (_year_priority(candidate_edges[idx]), -candidate_edges[idx].score),
                )
                for i in ranked_fill:
                    if fill_budget <= 0:
                        break
                    if i in selected_idx_set:
                        continue
                    e = candidate_edges[i].edge
                    if target_year is not None:
                        e_start = _parse_edge_year(e.start)
                        e_end = _parse_edge_year(e.end)
                        if e_start is not None and e_end is not None:
                            year_ok = e_start <= target_year <= e_end
                        elif e_start is not None:
                            year_ok = abs(e_start - target_year) <= 2
                        else:
                            year_ok = True
                        if not year_ok:
                            continue
                    selected_idx_set.add(i)
                    fill_budget -= 1

        # Sort by original score order
        all_selected = sorted(selected_idx_set)
        filtered_edges = [candidate_edges[i] for i in all_selected if i < len(candidate_edges)]

        # Step 3b: record LLM stage1 selection details
        if not any(d["name"] == "year_priority_selection" for d in debug_steps):
            debug_steps.append({
                "step": 3,
                "name": "llm_stage1_selection",
                "interleaved_pool_top_k": [_edge_dict(se) for se in _interleaved[:30]],
                "llm_picked_indices": sorted(llm_indices),
                "safety_net_added": sorted(selected_idx_set - {_interleaved_to_orig.get(i, i) for i in llm_indices}),
                "stage1_raw": stage1.get("stage1_raw"),
            })
        debug_steps.append({
            "step": 4,
            "name": "final_edges",
            "count": len(filtered_edges),
            "edges": [_edge_dict(se) for se in filtered_edges],
        })

        # ── Optional continuation (Stage 2) ──
        continuation_info: Dict[str, Any] | None = None
        if stage1.get("need_more") and filtered_edges:
            cont = self._stage2_expand(question, filtered_edges, sr.year)
            continuation_info = cont
            if not cont.get("sufficient", True):
                extra_ents = cont.get("entities") or []
                if extra_ents:
                    # Retrieve more edges from suggested entities
                    sr2 = self.semantic_grounder.retrieve_edges_for_entities(
                        entities=extra_ents,
                        year=sr.year,
                        max_returned_edges=20,
                    )
                    existing_uids = {se.edge.edge_uid for se in filtered_edges}
                    for se2 in sr2.scored_edges:
                        if se2.edge.edge_uid not in existing_uids:
                            filtered_edges.append(se2)
                            existing_uids.add(se2.edge.edge_uid)

        # Feed relation hints
        if sr.relation_hints:
            plan["relation_hint"] = sr.relation_hints[0]

        # ── Build answer directly from curated edges ──
        # (Bypass answer_from_plan() entity-UID re-filtering)
        answer = self._build_answer_from_edges(question, plan, filtered_edges)

        # Step 5: answer
        debug_steps.append({
            "step": 5,
            "name": "answer",
            "answer_text": answer.answer_text,
            "confidence": answer.confidence,
        })

        return {
            "question": payload.get("question", ""),
            "intent": plan.get("query_type", "unsupported"),
            "plan": plan,
            "answer": answer,
            "mermaid": answer_to_mermaid(answer),
            "grounding": grounding,
            "stage1_selection": stage1,
            "all_candidate_edges": [
                {
                    "src": self.index.node_label(se.edge.source_uid),
                    "rel": se.edge.relation,
                    "tgt": self.index.node_label(se.edge.target_uid),
                    "start": se.edge.start,
                    "end": se.edge.end,
                    "score": round(se.score, 4),
                }
                for se in candidate_edges[:30]
            ],
            "continuation": continuation_info,
            "debug_steps": debug_steps,
        }

    @staticmethod
    def _format_answer(payload: Dict[str, Any]) -> Dict[str, Any]:
        answer: GraphAnswer = payload["answer"]
        return {
            "question": payload["question"],
            "intent": payload["intent"],
            "plan": payload.get("plan", {}),
            "answer_text": answer.answer_text,
            "confidence": answer.confidence,
            "evidence": answer.evidence,
            "mermaid": payload["mermaid"],
            "grounding": payload.get("grounding", {}),
            "stage1_selection": payload.get("stage1_selection"),
            "all_candidate_edges": payload.get("all_candidate_edges"),
            "continuation": payload.get("continuation"),
            "debug_steps": payload.get("debug_steps", []),
        }

    def invoke(self, question: str, use_llm: bool = False) -> Dict[str, Any]:
        base = self.answer_chain.invoke({"question": question})
        if use_llm and self.llm_chain is not None:
            base["llm_answer"] = self.llm_chain.invoke({"question": question})
        return base
