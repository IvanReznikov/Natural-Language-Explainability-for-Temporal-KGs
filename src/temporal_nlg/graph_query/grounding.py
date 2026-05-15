from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from temporal_nlg.graph_query.index import GraphEdge, TemporalGraphIndex

# ── Relations that are structural / temporal bookkeeping ─────────────
STRUCTURAL_RELATIONS: frozenset[str] = frozenset({
    "spans_year", "within_year", "dated", "has_year",
    "occurred_on", "start_date", "end_date",
    "tag_related_to", "inferred_tag",
})

_YEAR_RE = re.compile(r"\b(1\d{3}|2\d{3})\b")
_DATE_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?\b")
_QUOTED_RE = re.compile(r"[\u2018\u2019'\"]+(.+?)[\u2018\u2019'\"]+")
_TITLE_ENTITY_RE = re.compile(r"\b([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3})\b")

_FALLBACK_STOPWORDS: frozenset[str] = frozenset({
    "who", "what", "which", "where", "when", "why", "how",
    "was", "were", "is", "are", "did", "does", "do",
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to",
    "before", "after", "during", "between", "from", "with", "for",
    "first", "last", "year", "time", "country", "city", "company",
})


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def _norm(text: str) -> str:
    return str(text or "").strip().lower()


_LEXICAL_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "in", "on", "at", "by", "for", "to", "from", "of",
    "and", "or", "with", "when", "what", "which", "who", "was", "were", "is",
    "are", "did", "does", "do", "before", "after", "during", "between", "into",
    "over", "under", "how", "why", "where", "whose", "whom",
})


def _question_lexical_candidates(question: str, max_terms: int = 10) -> List[str]:
    candidates: List[str] = []

    for m in re.finditer(r"[\"'\u2018\u2019\u201c\u201d]([^\"'\u2018\u2019\u201c\u201d]{2,80})[\"'\u2018\u2019\u201c\u201d]", question):
        term = m.group(1).strip()
        if term and term not in candidates:
            candidates.append(term)

    for m in re.finditer(r"\b(?:[A-Z][\w\-']+(?:\s+[A-Z][\w\-']+){0,3})\b", question):
        phrase = m.group(0).strip()
        if phrase and phrase.lower() not in _LEXICAL_STOP_WORDS and phrase not in candidates:
            candidates.append(phrase)

    clean = re.sub(r"[^\w\s\-']", " ", question.lower())
    words = [w for w in clean.split() if len(w) >= 4 and w not in _LEXICAL_STOP_WORDS and not _YEAR_RE.fullmatch(w)]
    for w in words:
        if w not in candidates:
            candidates.append(w)
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        if bg not in candidates:
            candidates.append(bg)

    return candidates[:max_terms]


def _top_k(scores: np.ndarray, k: int) -> List[int]:
    if scores.size == 0 or k <= 0:
        return []
    kk = min(int(k), int(scores.size))
    idx = np.argpartition(scores, -kk)[-kk:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [int(v) for v in idx]


def _find_latest(emb_dir: Path, prefix: str) -> Optional[Path]:
    candidates = sorted(emb_dir.glob(f"{prefix}_*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_string_meta(path: Path, field: str) -> List[str]:
    values: List[str] = []
    for row in _iter_jsonl(path):
        values.append(str(row.get(field) or ""))
    return values


def _extract_question_fallback_entities(question: str, *, max_entities: int = 4) -> List[str]:
    """Extract conservative fallback entity phrases from question text.

    This is used only when planner entities are sparse/unresolved. It avoids
    broad raw-keyword fallback by preferring quoted spans and title-case phrases.
    """
    text = str(question or "").strip()
    if not text:
        return []

    candidates: List[str] = []

    for m in _QUOTED_RE.finditer(text):
        span = m.group(1).strip()
        if span and span not in candidates and len(span.split()) <= 5:
            candidates.append(span)

    for m in _TITLE_ENTITY_RE.finditer(text):
        span = m.group(1).strip(" ,;:?!")
        if not span:
            continue
        low = span.lower()
        if _YEAR_RE.fullmatch(low) or _DATE_RE.fullmatch(low):
            continue
        if low in _FALLBACK_STOPWORDS:
            continue
        if len(low) < 3 or len(span.split()) > 5:
            continue
        if span not in candidates:
            candidates.append(span)
        if len(candidates) >= max_entities:
            break

    return candidates[:max_entities]


@dataclass
class GroundingArtifacts:
    node_matrix: Optional[np.ndarray]
    node_labels: List[str]
    tag_matrix: Optional[np.ndarray]
    tags: List[str]
    relation_matrix: Optional[np.ndarray]
    relations: List[str]


@dataclass
class ScoredEdge:
    """Lightweight scored edge used throughout the new retrieval pipeline."""
    edge: GraphEdge
    score: float


@dataclass
class SubgraphResult:
    """Full result from the graph-native retrieval pipeline."""
    scored_edges: List[ScoredEdge]
    entity_uids: List[str]
    relation_hints: List[str]
    year: Optional[int]
    node_hits: List[Dict[str, Any]]
    tag_hits: List[Dict[str, Any]]
    relation_hits: List[Dict[str, Any]]
    grounding_enabled: bool


class SemanticGrounder:
    """Primary graph retrieval engine using precomputed node/tag/relation embeddings.

    Pipeline:
    1. Parse question → extract entities + year
    2. Embed question → dot-product vs precomputed node/tag/relation matrices
    3. Resolve top-K nodes → walk graph edges
    4. Score edges by relation similarity + date overlap
    5. Return ranked edge triples for LLM consumption
    """

    def __init__(self, index: TemporalGraphIndex, qwen_server_url: str = "", qwen_embed_url: str = ""):
        self.index = index
        self.qwen_server_url = (
            qwen_embed_url or qwen_server_url or ""
        ).rstrip("/")
        self.low_mem_mode = str(os.getenv("GRAPH_LOW_MEM_MODE", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        self.disable_vector_grounding = str(os.getenv("GRAPH_DISABLE_GROUNDING_VECTORS", "1" if self.low_mem_mode else "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        self.emb_dir = self.index.output_dir / "embeddings"
        self._server_mode: Optional[str] = None

        self.node_label_to_uids: Dict[str, List[str]] = {}
        self.tag_to_uid: Dict[str, str] = {}
        self._load_uid_maps()

        # Build reverse map: node_uid ↦ normalized_label (for date detection)
        self._uid_to_norm_label: Dict[str, str] = {}
        for label, uids in self.node_label_to_uids.items():
            for uid in uids:
                self._uid_to_norm_label[uid] = label

        self.artifacts = self._load_artifacts()
        self.enabled = any(
            part is not None
            for part in [
                self.artifacts.node_matrix,
                self.artifacts.tag_matrix,
                self.artifacts.relation_matrix,
            ]
        )

    def _load_uid_maps(self) -> None:
        nodes_path = self.index.output_dir / "nodes.jsonl"
        for row in _iter_jsonl(nodes_path):
            node_uid = str(row.get("node_uid") or "")
            normalized_label = _norm(str(row.get("normalized_label") or ""))
            if not node_uid or not normalized_label:
                continue
            self.node_label_to_uids.setdefault(normalized_label, []).append(node_uid)

        tags_path = self.index.output_dir / "tags.jsonl"
        for row in _iter_jsonl(tags_path):
            tag_uid = str(row.get("tag_uid") or "")
            normalized_tag = _norm(str(row.get("normalized_tag") or ""))
            if tag_uid and normalized_tag:
                self.tag_to_uid[normalized_tag] = tag_uid

    def _load_artifacts(self) -> GroundingArtifacts:
        if self.disable_vector_grounding:
            return GroundingArtifacts(
                node_matrix=None,
                node_labels=[],
                tag_matrix=None,
                tags=[],
                relation_matrix=None,
                relations=[],
            )

        node_matrix = None
        node_labels: List[str] = []
        tag_matrix = None
        tags: List[str] = []
        relation_matrix = None
        relations: List[str] = []

        node_npy = _find_latest(self.emb_dir, "node_normalized_label_embeddings")
        if node_npy is not None:
            node_meta = Path(str(node_npy).replace(".npy", ".meta.jsonl"))
            if node_meta.exists():
                labels = _load_string_meta(node_meta, "normalized_label")
                arr = np.load(node_npy)
                if arr.ndim == 2 and arr.shape[0] == len(labels):
                    node_matrix = np.asarray(arr, dtype=np.float32)
                    node_labels = labels

        tag_npy = _find_latest(self.emb_dir, "tag_normalized_tag_embeddings")
        if tag_npy is not None:
            tag_meta = Path(str(tag_npy).replace(".npy", ".meta.jsonl"))
            if tag_meta.exists():
                vals = _load_string_meta(tag_meta, "normalized_tag")
                arr = np.load(tag_npy)
                if arr.ndim == 2 and arr.shape[0] == len(vals):
                    tag_matrix = np.asarray(arr, dtype=np.float32)
                    tags = vals

        relation_npy = _find_latest(self.emb_dir, "edge_relation_embeddings")
        if relation_npy is not None:
            relation_meta = Path(str(relation_npy).replace(".npy", ".meta.jsonl"))
            if relation_meta.exists():
                vals = _load_string_meta(relation_meta, "relation")
                arr = np.load(relation_npy)
                if arr.ndim == 2 and arr.shape[0] == len(vals):
                    relation_matrix = np.asarray(arr, dtype=np.float32)
                    relations = vals

        return GroundingArtifacts(
            node_matrix=node_matrix,
            node_labels=node_labels,
            tag_matrix=tag_matrix,
            tags=tags,
            relation_matrix=relation_matrix,
            relations=relations,
        )

    def _post_json(self, url: str, payload: dict, timeout_sec: int = 120) -> dict:
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _resolve_server_mode(self) -> Optional[str]:
        if self._server_mode:
            return self._server_mode
        if not self.qwen_server_url:
            return None

        model_name = os.getenv("LOCAL_EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
        probe = ["probe"]
        try:
            body = self._post_json(
                f"{self.qwen_server_url}/v1/embeddings",
                {"model": model_name, "input": probe},
            )
            data = body.get("data")
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) and isinstance(data[0].get("embedding"), list):
                self._server_mode = "v1_embeddings"
                return self._server_mode
        except urllib.error.HTTPError as exc:
            if exc.code not in {404, 405}:
                return None
        except Exception:
            return None

        try:
            body = self._post_json(
                f"{self.qwen_server_url}/embed",
                {"texts": probe, "is_query": True},
            )
            vectors = body.get("vectors")
            if isinstance(vectors, list) and len(vectors) == 1 and isinstance(vectors[0], list):
                self._server_mode = "embed"
                return self._server_mode
        except Exception:
            return None

        return None

    def _embed_query(self, text: str) -> Optional[np.ndarray]:
        q = str(text or "").strip()
        if not q:
            return None

        mode = self._resolve_server_mode()
        if mode is not None:
            try:
                if mode == "v1_embeddings":
                    model_name = os.getenv("LOCAL_EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
                    body = self._post_json(
                        f"{self.qwen_server_url}/v1/embeddings",
                        {"model": model_name, "input": [q]},
                    )
                    data = body.get("data")
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        emb = data[0].get("embedding")
                        if isinstance(emb, list):
                            arr = np.asarray(emb, dtype=np.float32)
                            n = float(np.linalg.norm(arr))
                            return arr / n if n > 0 else arr
                else:
                    body = self._post_json(
                        f"{self.qwen_server_url}/embed",
                        {"texts": [q], "is_query": True},
                    )
                    vectors = body.get("vectors")
                    if isinstance(vectors, list) and vectors and isinstance(vectors[0], list):
                        arr = np.asarray(vectors[0], dtype=np.float32)
                        n = float(np.linalg.norm(arr))
                        return arr / n if n > 0 else arr
            except Exception:
                return None

        try:
            from temporal_nlg.models import QwenEmbeddingModel

            if not hasattr(self, '_cached_embed_model'):
                self._cached_embed_model = QwenEmbeddingModel()
            model = self._cached_embed_model
            if not model.available:
                return None
            arr = np.asarray(model.embed_query(q), dtype=np.float32)
            n = float(np.linalg.norm(arr))
            return arr / n if n > 0 else arr
        except Exception:
            return None

    def ground(
        self,
        question: str,
        plan: Dict[str, Any],
        top_k_nodes: int = 10,
        top_k_tags: int = 8,
        top_k_relations: int = 3,
    ) -> Dict[str, Any]:
        """Compatibility API — thin wrapper around retrieve_subgraph()."""
        sr = self.retrieve_subgraph(question, plan,
                                     top_k_nodes=top_k_nodes,
                                     top_k_tags=top_k_tags,
                                     top_k_relations=top_k_relations)
        return {
            "enabled": sr.grounding_enabled,
            "entity_uids": sr.entity_uids,
            "relation_hint": sr.relation_hints[0] if sr.relation_hints else None,
            "node_hits": sr.node_hits,
            "tag_hits": sr.tag_hits,
            "relation_hits": sr.relation_hits,
        }

    # ── Public vector grounding (Stage 1 of 2-stage pipeline) ────────
    def ground_vectors(
        self,
        question: str,
        plan: Dict[str, Any],
        top_k_nodes: int = 10,
        top_k_tags: int = 8,
        top_k_relations: int = 5,
    ) -> Tuple[
        List[Dict[str, Any]],  # node_hits
        List[Dict[str, Any]],  # tag_hits
        List[Dict[str, Any]],  # relation_hits
        List[str],             # entity_uids
    ]:
        """Public API: embedding-only candidate discovery (no edge walking)."""
        return self._ground_vectors(
            question, plan,
            top_k_nodes=top_k_nodes,
            top_k_tags=top_k_tags,
            top_k_relations=top_k_relations,
        )

    # ── Targeted edge retrieval (Stage 2 of 2-stage pipeline) ────────
    def retrieve_edges_for_entities(
        self,
        entities: List[str],
        relation_filter: Optional[List[str]] = None,
        year: Optional[int] = None,
        *,
        max_returned_edges: int = 60,
    ) -> SubgraphResult:
        """Walk edges from explicitly named entities, filtering by relations & year.

        Used after the LLM has selected specific entities/relations from the
        embedding candidate list (Stage 1).
        """
        import math as _math  # noqa: F811

        all_uids: List[str] = []
        seen_uids: Set[str] = set()
        for ent in entities:
            ent_str = str(ent).strip()
            if not ent_str:
                continue
            # exact label lookup
            label_norm = _norm(ent_str)
            uids_from_label = self.node_label_to_uids.get(label_norm, [])
            # substring / fuzzy lookup
            uids_from_resolve = self.index.resolve_node_uids(ent_str, max_hits=5)
            for uid in (uids_from_label + uids_from_resolve):
                if uid and uid not in seen_uids:
                    seen_uids.add(uid)
                    all_uids.append(uid)

        # Filter out date-only nodes
        non_date_seeds: List[str] = []
        for uid in all_uids:
            cat = self.index.node_category_by_uid.get(uid, "")
            label = self._uid_to_norm_label.get(uid, "")
            if cat == "date" or _YEAR_RE.fullmatch(label) or _DATE_RE.fullmatch(label):
                continue
            non_date_seeds.append(uid)

        seed_uids = non_date_seeds if non_date_seeds else all_uids[:10]

        # Walk 1-hop edges
        edge_pool = self._collect_edges_from_nodes(
            seed_uids, max_edges_per_node=60, max_total=600,
        )

        # Build relation weight map from the filter list
        rel_filter_set: Set[str] = set()
        relation_hits_synthetic: List[Dict[str, Any]] = []
        if relation_filter:
            for rel in relation_filter:
                rn = _norm(str(rel))
                if rn:
                    rel_filter_set.add(rn)
                    relation_hits_synthetic.append({"relation": rn, "score": 0.80})

        # Score edges
        seed_set = set(seed_uids)
        scored = self._score_edges(edge_pool, relation_hits_synthetic, year, seed_set)

        # Boost edges whose relation matches the LLM-selected filter
        if rel_filter_set:
            for se in scored:
                if _norm(se.edge.relation) in rel_filter_set:
                    se.score += 0.30
            scored.sort(key=lambda s: (s.score, s.edge.support_count), reverse=True)

        # 2-hop expansion
        if scored:
            hop2 = self._expand_2hop(
                scored[:8], edge_pool, seed_set, year,
                max_expand=5, max_fanout=20,
            )
            if hop2:
                edge_pool.update(hop2)
                scored = self._score_edges(edge_pool, relation_hits_synthetic, year, seed_set)
                if rel_filter_set:
                    for se in scored:
                        if _norm(se.edge.relation) in rel_filter_set:
                            se.score += 0.30
                    scored.sort(key=lambda s: (s.score, s.edge.support_count), reverse=True)

        return SubgraphResult(
            scored_edges=scored[:max_returned_edges],
            entity_uids=all_uids,
            relation_hints=list(rel_filter_set),
            year=year,
            node_hits=[],
            tag_hits=[],
            relation_hits=relation_hits_synthetic,
            grounding_enabled=True,
        )

    # ── Step 1: Parse dates from question ────────────────────────────
    @staticmethod
    def _parse_year_from_question(question: str) -> Optional[int]:
        """Extract the most likely target year from the question text."""
        # Try full date patterns first  (e.g. "2008-12-15")
        dm = _DATE_RE.search(question)
        if dm:
            return int(dm.group(1))
        # Bare years
        years = [int(v) for v in _YEAR_RE.findall(question)]
        return years[0] if years else None

    # ── Step 2: Embed question and find top nodes/tags/relations ─────
    def _ground_vectors(
        self,
        question: str,
        plan: Dict[str, Any],
        top_k_nodes: int = 10,
        top_k_tags: int = 8,
        top_k_relations: int = 5,
    ) -> Tuple[
        List[Dict[str, Any]],  # node_hits
        List[Dict[str, Any]],  # tag_hits
        List[Dict[str, Any]],  # relation_hits
        List[str],             # entity_uids (from embedding search)
    ]:
        """Pure vector search over precomputed matrices -> top node/tag/relation hits."""

        context_plan = {k: v for k, v in plan.items() if k != "_question"}
        context = f"{question} {json.dumps(context_plan, ensure_ascii=False)}"
        qvec = self._embed_query(context)

        node_hits: List[Dict[str, Any]] = []
        tag_hits: List[Dict[str, Any]] = []
        relation_hits: List[Dict[str, Any]] = []
        entity_uids: List[str] = []

        if qvec is None:
            return node_hits, tag_hits, relation_hits, entity_uids

        # Node scoring
        if self.artifacts.node_matrix is not None and self.artifacts.node_labels:
            scores = np.dot(self.artifacts.node_matrix, qvec)
            for idx in _top_k(scores, top_k_nodes):
                label = _norm(self.artifacts.node_labels[idx])
                uids = self.node_label_to_uids.get(label, [])
                node_hits.append({
                    "normalized_label": label,
                    "score": float(scores[idx]),
                    "uids": uids[:5],
                })
                entity_uids.extend(uids)

        # Tag scoring
        if self.artifacts.tag_matrix is not None and self.artifacts.tags:
            scores = np.dot(self.artifacts.tag_matrix, qvec)
            for idx in _top_k(scores, top_k_tags):
                normalized_tag = _norm(self.artifacts.tags[idx])
                tag_uid = self.tag_to_uid.get(normalized_tag)
                tag_hits.append({
                    "normalized_tag": normalized_tag,
                    "score": float(scores[idx]),
                    "uid": tag_uid,
                })
                if tag_uid:
                    entity_uids.append(tag_uid)

        # Relation scoring
        if self.artifacts.relation_matrix is not None and self.artifacts.relations:
            scores = np.dot(self.artifacts.relation_matrix, qvec)
            for idx in _top_k(scores, top_k_relations):
                relation_hits.append({
                    "relation": self.artifacts.relations[idx],
                    "score": float(scores[idx]),
                })

        return node_hits, tag_hits, relation_hits, entity_uids

    # ── Step 3: Walk graph edges from resolved nodes ─────────────────
    def _collect_edges_from_nodes(
        self,
        seed_uids: List[str],
        *,
        max_edges_per_node: int = 60,
        max_total: int = 500,
    ) -> Dict[str, GraphEdge]:
        """1-hop edge collection from seed nodes."""
        collected: Dict[str, GraphEdge] = {}
        for uid in seed_uids:
            edges = self.index.outgoing_edges(uid) + self.index.incoming_edges(uid)
            added = 0
            for edge in edges:
                if edge.edge_uid in collected:
                    continue
                collected[edge.edge_uid] = edge
                added += 1
                if added >= max_edges_per_node:
                    break
            if len(collected) >= max_total:
                break
        return collected

    # ── Step 4: Score edges by relation similarity + date overlap ────
    def _score_edges(
        self,
        edges: Dict[str, GraphEdge],
        relation_hits: List[Dict[str, Any]],
        year: Optional[int],
        seed_uid_set: Set[str],
    ) -> List[ScoredEdge]:
        """Score collected edges using relation affinity, temporal overlap, and structural weighting."""

        # Build relation score map from grounding
        top_relations: Dict[str, float] = {}
        for rh in relation_hits:
            rel = _norm(str(rh.get("relation") or ""))
            if rel:
                top_relations[rel] = float(rh.get("score") or 0.0)

        scored: List[ScoredEdge] = []
        for edge in edges.values():
            score = 0.0
            rel_norm = _norm(edge.relation)

            # (a) Relation affinity: if edge.relation is among top embedding hits
            if rel_norm in top_relations:
                score += 0.35 * top_relations[rel_norm]

            # (b) Structural penalty: spans_year etc. are poor evidence
            if rel_norm in STRUCTURAL_RELATIONS:
                score -= 0.25

            # (c) Base / gold_fact edge bonus
            if edge.edge_type == "gold_fact":
                score += 0.25  # gold_fact = explicitly validated triple
            elif edge.edge_type == "base":
                score += 0.15

            # (d) Support count bonus (log-scaled)
            import math
            score += 0.05 * math.log1p(edge.support_count)

            # (e) Continuous temporal overlap score
            # Instead of a binary +0.25, we compute inverse interval-length weighting:
            # a 1-year exact match scores +0.40; a 50-year interval gets < +0.01.
            if year is not None:
                s_year = edge.start_year
                e_year = edge.end_year
                if edge.overlaps_year(year):
                    if s_year is not None and e_year is not None:
                        span = max(abs(e_year - s_year), 0) + 1
                    elif s_year is not None:
                        span = 1  # point-in-time edge — exactly the target year
                    else:
                        span = 1
                    # Proportional to 1/span so narrower intervals get a larger boost
                    import math as _math
                    score += 0.40 / _math.log1p(span)
                elif s_year is not None or e_year is not None:
                    # Penalise wrong-year dated edges proportional to distance
                    closest = min(
                        abs((s_year or year) - year),
                        abs((e_year or year) - year)
                    )
                    score -= min(0.20, 0.02 * closest)

            # (f) Seed proximity: at least one endpoint is a seed
            if edge.source_uid in seed_uid_set or edge.target_uid in seed_uid_set:
                score += 0.10

            scored.append(ScoredEdge(edge=edge, score=score))

        scored.sort(key=lambda s: (s.score, s.edge.support_count), reverse=True)
        return scored

    # ── Step 5: Optional 2-hop expansion ─────────────────────────────
    def _expand_2hop(
        self,
        top_edges: List[ScoredEdge],
        existing: Dict[str, GraphEdge],
        seed_uids: Set[str],
        year: Optional[int],
        max_expand: int = 5,
        max_fanout: int = 20,
    ) -> Dict[str, GraphEdge]:
        """Expand 1 additional hop from endpoints of best edges."""
        new_edges: Dict[str, GraphEdge] = {}
        for se in top_edges[:max_expand]:
            for ep_uid in (se.edge.source_uid, se.edge.target_uid):
                if ep_uid in seed_uids:
                    continue
                neighbours = self.index.outgoing_edges(ep_uid) + self.index.incoming_edges(ep_uid)
                added = 0
                for nb in neighbours:
                    if nb.edge_uid in existing or nb.edge_uid in new_edges:
                        continue
                    if nb.edge_type not in ("base", "gold_fact"):
                        continue
                    if year is not None and not nb.overlaps_year(year):
                        continue
                    new_edges[nb.edge_uid] = nb
                    added += 1
                    if added >= max_fanout:
                        break
        return new_edges

    # ── Main entry point ─────────────────────────────────────────────
    def retrieve_subgraph(
        self,
        question: str,
        plan: Dict[str, Any],
        *,
        top_k_nodes: int = 10,
        top_k_tags: int = 8,
        top_k_relations: int = 5,
        max_returned_edges: int = 60,
    ) -> SubgraphResult:
        """Full graph-native retrieval pipeline.

        1. Parse year from question
        2. Embed question → top-K nodes / tags / relations via precomputed matrices
        3. Resolve entity names from plan → node UIDs (lexical)
        4. Walk 1-hop edges from all seed nodes
        5. Score edges (relation affinity × date overlap × structural weight)
        6. Expand 2-hop from best edges
        7. Re-score, rank, return
        """

        # ─── 1. Parse year ───────────────────────────────────────────
        year: Optional[int] = plan.get("year")
        if year is None:
            year = self._parse_year_from_question(question)

        # ─── 2. Embedding-based grounding ────────────────────────────
        node_hits, tag_hits, relation_hits, embed_uids = self._ground_vectors(
            question, plan,
            top_k_nodes=top_k_nodes,
            top_k_tags=top_k_tags,
            top_k_relations=top_k_relations,
        )
        grounding_enabled = bool(node_hits or tag_hits or relation_hits)

        # ─── 3. Lexical entity resolution from plan.entities ────────
        # Use only plan entities (from the query planner / NER) for lexical
        # lookup.  Keyword-substring fallback over raw question text was removed
        # because it injected noise (matching unrelated nodes that happen to share
        # common words) and hurt retrieval precision.  When graph coverage is
        # missing the pipeline falls back to query-embedding + QA retriever.
        plan_entities = [str(v) for v in (plan.get("entities") or [])]
        lexical_uids: List[str] = []
        for ent in plan_entities:
            lexical_uids.extend(self.index.resolve_node_uids(ent, max_hits=5))

        # Guarded lexical backstop: when planner entities are sparse, derive
        # a few lexical candidates directly from the question.
        if len(lexical_uids) < 2:
            for cand in _question_lexical_candidates(question, max_terms=10):
                cand_uids = self.index.resolve_node_uids(cand, max_hits=2)
                if not cand_uids:
                    continue
                cand_tokens = {t for t in _norm(cand).split() if len(t) > 2}
                for uid in cand_uids:
                    label = self._uid_to_norm_label.get(uid, "")
                    label_tokens = set(label.split())
                    if cand_tokens and cand_tokens & label_tokens:
                        lexical_uids.append(uid)

        # Guarded lexical fallback: when planner entities are sparse or fail to
        # resolve, use conservative phrase extraction from the question text.
        # This restores recall without reintroducing broad noisy keyword matching.
        non_empty_plan_entities = [e for e in plan_entities if _norm(e)]
        if len(non_empty_plan_entities) < 2 or not lexical_uids:
            plan_entity_set = {_norm(e) for e in non_empty_plan_entities}
            fallback_entities = _extract_question_fallback_entities(question, max_entities=4)
            for ent in fallback_entities:
                if _norm(ent) in plan_entity_set:
                    continue
                lexical_uids.extend(self.index.resolve_node_uids(ent, max_hits=3))

        # Also resolve date nodes from the year
        if year is not None:
            date_uids = self.index.resolve_node_uids(str(year), max_hits=3)
            lexical_uids.extend(date_uids)

        # Merge all entity UIDs (dedup)
        all_uids: List[str] = []
        seen: Set[str] = set()
        for uid in (lexical_uids + embed_uids):
            if uid and uid not in seen:
                seen.add(uid)
                all_uids.append(uid)

        # Filter out date-only nodes from the "seed entities for walking"
        # because date nodes (e.g. "1995") have thousands of edges.
        # Instead, we use year for filtering.
        non_date_seeds: List[str] = []
        date_seeds: List[str] = []
        for uid in all_uids:
            cat = self.index.node_category_by_uid.get(uid, "")
            label = self._uid_to_norm_label.get(uid, "")
            if cat == "date" or _YEAR_RE.fullmatch(label) or _DATE_RE.fullmatch(label):
                date_seeds.append(uid)
            else:
                non_date_seeds.append(uid)

        seed_uids_for_walk = non_date_seeds if non_date_seeds else all_uids[:10]

        # ─── 4. Walk 1-hop edges from seeds ──────────────────────────
        edge_pool = self._collect_edges_from_nodes(
            seed_uids_for_walk,
            max_edges_per_node=60,
            max_total=600,
        )

        # ─── 5. Score edges ──────────────────────────────────────────
        seed_set = set(seed_uids_for_walk)
        scored = self._score_edges(edge_pool, relation_hits, year, seed_set)

        # ─── 6. 2-hop expansion from best hits ──────────────────────
        if scored:
            hop2_edges = self._expand_2hop(
                scored[:8], edge_pool, seed_set, year,
                max_expand=5, max_fanout=20,
            )
            if hop2_edges:
                edge_pool.update(hop2_edges)
                scored = self._score_edges(edge_pool, relation_hits, year, seed_set)

        # ─── 7. Final ranking ────────────────────────────────────────
        final = scored[:max_returned_edges]

        relation_hint_list = [str(rh.get("relation") or "") for rh in relation_hits if rh.get("relation")]

        return SubgraphResult(
            scored_edges=final,
            entity_uids=all_uids,
            relation_hints=relation_hint_list,
            year=year,
            node_hits=node_hits,
            tag_hits=tag_hits,
            relation_hits=relation_hits,
            grounding_enabled=grounding_enabled,
        )
