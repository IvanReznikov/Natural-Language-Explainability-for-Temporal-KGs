"""Stateful bridge between the temporal explanation system and MeTTa (M4).

``TemporalBridge`` is a session-scoped holder that exposes the capabilities
delivered in earlier milestones as plain Python methods, grouped by source:

* **M1** (natural-language generation)  -> :meth:`nlg_fact`, :meth:`nlg_readability`
* **M2** (truth maintenance / traces)   -> :meth:`add_belief`, :meth:`record_rule`,
  :meth:`explain_belief`, :meth:`contradictions`, :meth:`why_not`,
  :meth:`support_chain`, :meth:`retract`
* **M3** (temporal graph QA)            -> :meth:`answer`, :meth:`evidence`,
  :meth:`confidence`, :meth:`mermaid`, :meth:`explain_path`

The bridge holds *no* new logic of its own; it composes the existing M1/M2/M3
APIs. ``atoms.py`` thinly wraps these methods as grounded MeTTa operations.

The graph pipeline (M3) is loaded lazily so the bridge can be constructed in a
bare environment (no model servers, no graph artifacts) and used purely for the
M1/M2 surface, which keeps unit tests hermetic and fast.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from temporal_nlg.core.templates import TemplateType, TemporalFact
from temporal_nlg.evaluation import AccuracyEvaluator, calculate_flesch_score
from temporal_nlg.models import HybridGenerator
from temporal_nlg.tms.belief_store import Belief, BeliefStore
from temporal_nlg.tms import meta_query
from temporal_nlg.tms.trace import QueryTrace, TraceRecorder

from .config import MettaConfig

# Map the small vocabulary used by MeTTa callers to TemplateType members.
_FACT_TYPE_ALIASES: Dict[str, TemplateType] = {
    "point_in_time": TemplateType.POINT_IN_TIME,
    "point": TemplateType.POINT_IN_TIME,
    "interval": TemplateType.INTERVAL,
    "sequence": TemplateType.SEQUENCE,
    "causality": TemplateType.CAUSALITY,
    "overlap": TemplateType.OVERLAP,
}


def _parse_fact_type(fact_type: str) -> TemplateType:
    key = (fact_type or "").strip().lower()
    if key not in _FACT_TYPE_ALIASES:
        raise ValueError(
            f"Unknown fact type {fact_type!r}. Expected one of: "
            f"{', '.join(sorted(_FACT_TYPE_ALIASES))}"
        )
    return _FACT_TYPE_ALIASES[key]


def _normalize_strategy(strategy: Optional[str]) -> Optional[str]:
    if strategy is None:
        return None
    key = (strategy or "").strip().lower() or None
    if key is None:
        return None
    # Accept "polished" (HybridGenerator's internal label) under the "polish" alias too.
    if key == "polish":
        return "polished"
    return key


class TemporalBridge:
    """Session-scoped composition of M1/M2/M3 capabilities for MeTTa."""

    def __init__(
        self,
        config: Optional[MettaConfig] = None,
        *,
        generator: Optional[HybridGenerator] = None,
        belief_store: Optional[BeliefStore] = None,
        trace_recorder: Optional[TraceRecorder] = None,
        pipeline: Any = None,
    ):
        self.config = config or MettaConfig.from_env()
        # M1 — NLG. Constructed eagerly; HybridGenerator tolerates no-API-key
        # environments by falling back to a stub polisher.
        self.generator = generator or HybridGenerator(
            model=self.config.nlg_model,
            polish_threshold=self.config.polish_threshold,
        )
        self.evaluator = AccuracyEvaluator()
        # M2 — truth maintenance.
        self.belief_store = belief_store or BeliefStore()
        self.trace_recorder = trace_recorder or TraceRecorder(
            sampling_rate=self.config.trace_sampling_rate
        )
        self._active_trace: Optional[QueryTrace] = None
        # M3 — graph QA. Lazy: only built when a graph op is first called.
        self._pipeline = pipeline

    # ------------------------------------------------------------------
    # Lifecycle / session helpers
    # ------------------------------------------------------------------

    @property
    def pipeline(self) -> Any:
        """Lazily build the M3 graph pipeline on first use."""
        if self._pipeline is None:
            from temporal_nlg.graph_query import TemporalGraphLCELPipeline

            graph_dir = self.config.graph_dir
            if not (graph_dir / "nodes.jsonl").exists():
                raise FileNotFoundError(
                    f"Graph artifacts not found at {graph_dir}. "
                    "Set METTA_GRAPH_DIR to a directory containing nodes.jsonl "
                    "and edges.jsonl (see data/jsonls/temporal_graph_output_v3)."
                )
            self._pipeline = TemporalGraphLCELPipeline(graph_dir)
        return self._pipeline

    def reset(self) -> Dict[str, int]:
        """Clear M2 session state (beliefs + active trace).

        The M1 generator cache and the M3 pipeline are retained because they are
        expensive to rebuild and stateless w.r.t. the session.
        """
        n_beliefs = len(self.belief_store.beliefs)
        self.belief_store = BeliefStore()
        self._active_trace = None
        return {"cleared_beliefs": n_beliefs}

    def start_trace(self, query_id: Optional[str] = None) -> str:
        """Begin a traced reasoning session (M2). Returns the trace id."""
        trace = self.trace_recorder.start_query(query_id)
        self._active_trace = trace
        return trace.query_id

    def _require_trace(self) -> QueryTrace:
        if self._active_trace is None:
            self.start_trace()
        assert self._active_trace is not None
        return self._active_trace

    def trace_to_dict(self) -> Dict[str, Any]:
        """Serialize the active trace (empty dict if none started)."""
        if self._active_trace is None:
            return {}
        self.trace_recorder.complete_query(self._active_trace)
        return self._active_trace.to_dict()

    # ------------------------------------------------------------------
    # M1 — Natural-language generation
    # ------------------------------------------------------------------

    def nlg_fact(
        self,
        fact_type: str,
        content_json: str,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verbalize a single temporal fact (M1).

        Args:
            fact_type: one of point_in_time|interval|sequence|causality|overlap.
            content_json: JSON object with the fact's fields.
            strategy: optional "template"|"polish"|"llm" override.
        """
        fact = TemporalFact(
            fact_type=_parse_fact_type(fact_type),
            content=json.loads(content_json),
        )
        result = self.generator.generate(fact, force_strategy=_normalize_strategy(strategy))
        return {
            "text": result.text,
            "strategy": result.strategy,
            "confidence": result.confidence,
            "flesch_score": result.flesch_score,
            "template_id": result.template_id,
        }

    def nlg_readability(self, text: str) -> Dict[str, float]:
        """Compute readability metrics for a generated text (M1)."""
        flesch = calculate_flesch_score(text)
        try:
            from temporal_nlg.evaluation import calculate_information_density

            density = calculate_information_density(text)
        except Exception:
            density = None
        out: Dict[str, float] = {"flesch_score": flesch}
        if density is not None:
            out["information_density"] = density
        return out

    def nlg_evaluate(
        self, content_json: str, generated_text: str, fact_type: str = "point_in_time"
    ) -> Dict[str, Any]:
        """Score a generated text against its source fact (M1)."""
        fact = TemporalFact(
            fact_type=_parse_fact_type(fact_type),
            content=json.loads(content_json),
        )
        metrics = self.evaluator.evaluate(fact, generated_text)
        return {
            "date_preservation": metrics.date_preservation,
            "entity_preservation": metrics.entity_preservation,
            "relation_preservation": metrics.relation_preservation,
            "hallucination_detected": metrics.hallucination_detected,
            "overall_accuracy": metrics.overall_accuracy,
        }

    # ------------------------------------------------------------------
    # M2 — Truth maintenance / traces
    # ------------------------------------------------------------------

    def add_belief(
        self,
        belief_id: str,
        payload_json: str = "{}",
        supports_json: str = "[]",
        evidence_json: str = "[]",
    ) -> str:
        """Register a belief with its supports and evidence (M2)."""
        belief = Belief(
            belief_id=belief_id,
            payload=json.loads(payload_json) if payload_json else {},
            supports=json.loads(supports_json) if supports_json else [],
            evidence=json.loads(evidence_json) if evidence_json else [],
        )
        self.belief_store.add_belief(belief)
        return belief_id

    def record_rule(
        self,
        rule_id: str,
        rule_name: str,
        inputs_json: str = "[]",
        conclusion_json: str = "{}",
        confidence: float = 1.0,
    ) -> str:
        """Record a rule firing into the active trace (M2).

        Starts a trace automatically if none is active.
        """
        trace = self._require_trace()
        self.trace_recorder.record_rule_firing(
            trace,
            rule_id=rule_id,
            rule_name=rule_name,
            inputs=json.loads(inputs_json) if inputs_json else [],
            conclusion=json.loads(conclusion_json) if conclusion_json else {},
            confidence=confidence,
        )
        return rule_id

    def record_rule_facts(
        self,
        rule_id: str,
        rule_name: str,
        in_fact_id: str,
        in_value: str,
        out_fact_id: str,
        out_value: str,
        confidence: float = 1.0,
    ) -> str:
        """Record a rule firing from plain fact scalars instead of JSON (M2).

        Same as :meth:`record_rule` but takes the single input fact and the
        conclusion fact as individual values, so MeTTa programs can feed atoms
        they derived themselves (e.g. via ``match`` over evidence atoms)
        straight into the trace without hand-building JSON strings.
        """
        return self.record_rule(
            rule_id,
            rule_name,
            json.dumps([{"fact_id": in_fact_id, "value": in_value}]),
            json.dumps({"fact_id": out_fact_id, "value": out_value}),
            confidence,
        )

    def rules_fired(self) -> List[str]:
        """Return the ids of rules that fired in the active trace (M2)."""
        if self._active_trace is None:
            return []
        return meta_query.rules_fired(self._active_trace)

    def why_not(self, expected_json: str) -> Dict[str, str]:
        """Explain why expected rules did not fire (M2)."""
        trace = self._require_trace()
        return meta_query.why_not_fired(trace, json.loads(expected_json) if expected_json else [])

    def explain_belief_trace(self, fact_id: str) -> str:
        """Explain how a fact id was derived within the trace (M2)."""
        trace = self._require_trace()
        return meta_query.explain_fact(trace, fact_id)

    def justification_paths(self, fact_id: str, max_depth: int = 6) -> List[Dict[str, Any]]:
        """Extract the full multi-hop justification paths for a conclusion (M2).

        Unlike :meth:`explain_belief_trace` (which renders a single textual
        rationale), this returns the structured chains of rules that produced a
        conclusion fact, reconstructed by DFS over the recorded trace. This is
        the substrate for "show me the full chain of reasoning" debugging.
        """
        from temporal_nlg.tms.trace_explain import TraceJustifier

        trace = self._require_trace()
        justifier = TraceJustifier(trace)
        paths = justifier.paths_for_fact(fact_id, max_depth=max_depth)
        return [
            {
                "conclusion_fact": p.conclusion_fact,
                "rule_sequence": [
                    {
                        "rule_id": rt.rule_id,
                        "rule_name": rt.rule_name,
                        "inputs": rt.inputs,
                        "conclusion": rt.conclusion,
                    }
                    for rt in p.rule_sequence
                ],
                "as_text": p.as_text(),
            }
            for p in paths
        ]

    def contradictions(self) -> List[Dict[str, Any]]:
        """Detect contradictions among trace conclusions (M2)."""
        if self._active_trace is None:
            return []
        return meta_query.contradictions(self._active_trace)

    def influential_facts(self, top_k: int = 5) -> List[List[Any]]:
        """Rank facts by how often they fed a rule (M2)."""
        trace = self._require_trace()
        # Tuples -> lists so they serialize cleanly through MeTTa atoms.
        return [list(pair) for pair in meta_query.influential_facts(trace, top_k=top_k)]

    def support_chain(self, belief_id: str) -> List[Dict[str, Any]]:
        """Return the breadth-first support chain for a belief (M2)."""
        chain = self.belief_store.get_support_chain(belief_id)
        return [self._belief_to_dict(b) for b in chain]

    def explain_belief(self, belief_id: str) -> str:
        """Render a human-readable justification for a belief (M2)."""
        from temporal_nlg.tms.justification import JustificationBuilder

        belief = self.belief_store.get_belief(belief_id)
        if belief is None:
            return f"Belief {belief_id} not found."
        supports = [
            self.belief_store.get_belief(s)
            for s in belief.supports
            if self.belief_store.get_belief(s) is not None
        ]
        return JustificationBuilder().build(belief, supports)

    def retract(self, belief_id: str) -> Dict[str, Any]:
        """Retract a belief and cascade dirty marks to dependents (M2)."""
        before = {b.belief_id: b.status for b in self.belief_store.beliefs.values()}
        self.belief_store.retract(belief_id)
        after = {b.belief_id: b.status for b in self.belief_store.beliefs.values()}
        affected = [bid for bid in after if after[bid] != before.get(bid)]
        return {"retracted": belief_id, "affected": affected}

    def active_beliefs(self) -> List[str]:
        return [b.belief_id for b in self.belief_store.get_active_beliefs()]

    def dirty_beliefs(self) -> List[str]:
        return [b.belief_id for b in self.belief_store.get_dirty_beliefs()]

    def counterfactual_shift_time(self, belief_id: str, delta: str) -> Dict[str, Any]:
        """Pose a temporal counterfactual over a recorded belief (M2 + M1).

        Uses the TMS CounterfactualEngine to ask "if the timeframe of this
        belief shifted by ``delta``, what would it become?" The counterfactual
        is registered as a new belief that *supports* the original, preserving
        the dependency chain.
        """
        from temporal_nlg.tms.counterfactual import CounterfactualEngine

        belief = self.belief_store.get_belief(belief_id)
        if belief is None:
            return {"error": f"Belief {belief_id} not found."}
        result = CounterfactualEngine().shift_time(belief, delta)
        self.belief_store.add_belief(result.new_belief)
        return {
            "original_id": result.original_id,
            "new_belief_id": result.new_belief.belief_id,
            "description": result.description,
            "payload": result.new_belief.payload,
        }

    # ------------------------------------------------------------------
    # M3 — Temporal graph QA + path explanation
    # ------------------------------------------------------------------

    def answer(self, question: str) -> Dict[str, Any]:
        """Answer a temporal question over the graph (M3)."""
        result = self.pipeline.invoke(question, use_llm=self.config.use_llm)
        # Cache the last answer for the cheap accessor ops below.
        self._last_answer = result
        return self._answer_view(result)

    def _answer_view(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": result.get("question", ""),
            "answer_text": result.get("answer_text", ""),
            "confidence": result.get("confidence", 0.0),
            "intent": result.get("intent", ""),
            "plan": result.get("plan", {}),
            "evidence": result.get("evidence", []),
            "mermaid": result.get("mermaid", ""),
        }

    def evidence(self) -> List[Dict[str, Any]]:
        last = getattr(self, "_last_answer", None)
        if last is None:
            return []
        return list(last.get("evidence", []))

    def evidence_edge_tuples(self) -> List[List[str]]:
        """Last answer's evidence edges as ``[src, rel, tgt, time]`` quads (M3).

        This is the atom-friendly view of :meth:`evidence`: plain string tuples
        that ``atoms.py`` turns into matchable ``(edge ...)`` expression atoms,
        so a MeTTa program can pattern-match over the graph evidence itself.
        """
        tuples: List[List[str]] = []
        for edge in self.evidence():
            when = edge.get("start") or edge.get("year") or ""
            tuples.append(
                [
                    str(edge.get("source") or edge.get("src") or ""),
                    str(edge.get("relation") or edge.get("rel") or ""),
                    str(edge.get("target") or edge.get("tgt") or ""),
                    str(when),
                ]
            )
        return tuples

    def confidence(self) -> float:
        last = getattr(self, "_last_answer", None)
        if last is None:
            return 0.0
        return float(last.get("confidence", 0.0))

    def mermaid(self) -> str:
        last = getattr(self, "_last_answer", None)
        if last is None:
            return 'graph TD\n    empty["No answer queried"]'
        return last.get("mermaid", "")

    def explain_path(
        self,
        adjacency_json: str,
        start: str,
        end: str,
        belief_id: Optional[str] = None,
        supports_json: str = "[]",
        evidence_json: str = "[]",
    ) -> Dict[str, str]:
        """Explain a graph path as a natural-language narrative (M3 + M1 + M2).

        Combines M3 path extraction with M1 narrative rendering and M2
        justification: the resulting narrative is recorded as a justified belief.
        """
        from temporal_nlg.explain.path_pipeline import path_to_narrative

        adj = json.loads(adjacency_json) if adjacency_json else {}
        bid = belief_id or f"belief_{uuid.uuid4().hex[:8]}"
        narrative = path_to_narrative(
            adj=adj,
            start=start,
            end=end,
            belief_store=self.belief_store,
            belief_id=bid,
            supports=json.loads(supports_json) if supports_json else None,
            evidence=json.loads(evidence_json) if evidence_json else None,
            style=self.config.narrative_style,
            domain=self.config.narrative_domain,
        )
        return narrative

    def explain_path_styled(
        self,
        adjacency_json: str,
        start: str,
        end: str,
        style: str = "neutral",
        domain: str = "general",
    ) -> Dict[str, str]:
        """Explain a graph path with an explicit style/domain (M3 + M1).

        Like :meth:`explain_path` but renders the *same* path in a chosen
        register (``novice``/``neutral``/``expert``) and domain
        (``general``/``medical``/``finance``) without recording a belief —
        useful for showing the same inference explained to different audiences.
        """
        from temporal_nlg.explain.graph_extract import extract_path
        from temporal_nlg.explain.narratives import PathNarrativeRenderer

        adj = json.loads(adjacency_json) if adjacency_json else {}
        nodes, edges = extract_path(adj, start, end)
        return PathNarrativeRenderer(style=style, domain=domain).render(nodes, edges)

    def counterfactual(
        self,
        subject: str,
        predicate: str,
        obj: str,
        alt_subject: str,
        alt_predicate: str,
        alt_obj: str,
        timeframe: Optional[str] = None,
        alt_timeframe: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate a fact-level counterfactual explanation (M1 + M3).

        Pairs a factual temporal fact against an alternative and verbalizes the
        divergence: "if instead <alternative>, then the outcome would diverge
        from the factual path." This is the hallmark of causal explainability.
        """
        from temporal_nlg.explain.counterfactuals import CounterfactualGenerator, Fact

        factual = Fact(subject=subject, predicate=predicate, obj=obj, timeframe=timeframe)
        alternative = Fact(
            subject=alt_subject, predicate=alt_predicate, obj=alt_obj, timeframe=alt_timeframe
        )
        return CounterfactualGenerator().generate(factual, alternative)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _belief_to_dict(belief: Belief) -> Dict[str, Any]:
        return {
            "belief_id": belief.belief_id,
            "payload": belief.payload,
            "supports": list(belief.supports),
            "evidence": list(belief.evidence),
            "status": belief.status,
        }


__all__ = ["TemporalBridge", "MettaConfig"]
