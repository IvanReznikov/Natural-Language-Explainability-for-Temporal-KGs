from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from temporal_nlg.graph_query.index import GraphEdge, TemporalGraphIndex
from temporal_nlg.graph_query.semantic import ScoredEdge


# Relations that are structural / temporal bookkeeping only — useless as factual answer evidence
STRUCTURAL_RELATIONS: frozenset[str] = frozenset({
    "spans_year",
    "within_year",
    "dated",
    "has_year",
    "occurred_on",
    "start_date",
    "end_date",
    "tag_related_to",
    "inferred_tag",
})

CAUSAL_RELATIONS = {
    "caused",
    "led_to",
    "fueled",
    "compelled",
    "triggered",
    "drove",
    "influenced",
    "enabled",
}

AFFECT_RELATIONS = CAUSAL_RELATIONS.union({"affected", "impact", "impacted"})


@dataclass
class GraphAnswer:
    question_type: str
    answer_text: str
    evidence: List[Dict[str, object]]
    confidence: float


def _edge_to_evidence(index: TemporalGraphIndex, edge: GraphEdge) -> Dict[str, object]:
    return {
        "edge_uid": edge.edge_uid,
        "source": index.node_label(edge.source_uid),
        "target": index.node_label(edge.target_uid),
        "relation": edge.relation,
        "start": edge.start,
        "end": edge.end,
        "edge_type": edge.edge_type,
        "support_count": edge.support_count,
        "source_row_ids": edge.source_row_ids[:10],
    }


class GraphRetriever:
    """Simple retrieval layer for temporal graph QA patterns."""

    def __init__(self, index: TemporalGraphIndex):
        self.index = index

    def _semantic_backoff(
        self,
        *,
        question_type: str,
        semantic_hits: Sequence[ScoredEdge],
        detail: str,
        max_edges: int = 5,
    ) -> GraphAnswer:
        top_edges = [hit.edge for hit in semantic_hits[: max(1, int(max_edges))]]
        if top_edges:
            return GraphAnswer(
                question_type=question_type,
                answer_text=f"{detail} Using best-effort semantic evidence from nearby graph edges.",
                evidence=[_edge_to_evidence(self.index, edge) for edge in top_edges],
                confidence=min(0.45, 0.18 + 0.06 * len(top_edges)),
            )
        return GraphAnswer(
            question_type=question_type,
            answer_text=f"{detail} Retrieval did not return supporting graph edges.",
            evidence=[],
            confidence=0.1,
        )

    def answer_from_plan(
        self,
        plan: Dict[str, Any],
        semantic_hits: Sequence[ScoredEdge],
        question: str,
    ) -> GraphAnswer:
        query_type = str(plan.get("query_type") or "unsupported")

        if query_type == "reason_of":
            entities = plan.get("entities") or []
            year = plan.get("year")
            if entities and year is not None:
                return self.reason_of_in_year(str(entities[0]), int(year))

        if query_type == "start_affecting":
            entities = plan.get("entities") or []
            if len(entities) >= 2:
                return self.when_started_affecting(str(entities[0]), str(entities[1]))

        if query_type == "analogical_transfer":
            entities = plan.get("entities") or []
            year = plan.get("year")
            if len(entities) >= 3 and year is not None:
                return self.analogical_outcome(
                    str(entities[0]),
                    str(entities[1]),
                    str(entities[2]),
                    int(year),
                )

        if query_type in {"state_at_time", "state_during_interval", "within_interval", "overlap_query"}:
            return self._answer_state(plan, semantic_hits)

        if query_type == "existence_in_time":
            return self._answer_existence(plan, semantic_hits)

        if query_type in {"first_occurrence", "earliest_among_set"}:
            return self._answer_first_last(plan, semantic_hits, earliest=True)

        if query_type in {"last_occurrence", "latest_among_set"}:
            return self._answer_first_last(plan, semantic_hits, earliest=False)

        if query_type in {"temporal_count", "frequency_query"}:
            return self._answer_count(plan, semantic_hits)

        if query_type in {"temporal_top_k", "trend_evolution"}:
            return self._answer_top_k(plan, semantic_hits)

        if query_type in {"temporal_path_existence", "time_respecting_path", "earliest_arrival_path"}:
            return self._answer_path_existence(plan)

        # Planner fallback for natural phrasing that should route to start-affecting.
        qlow = (question or "").strip().lower()
        if "start affecting" in qlow or "started affecting" in qlow:
            entities = [str(v) for v in (plan.get("entities") or [])]
            if len(entities) >= 2:
                return self.when_started_affecting(entities[0], entities[1])

            match = re.search(r"when\s+did\s+(.+?)\s+start(?:ed)?\s+affecting\s+(.+?)\??$", question, re.IGNORECASE)
            if match:
                return self.when_started_affecting(match.group(1).strip(), match.group(2).strip())

        if semantic_hits:
            top = [hit.edge for hit in semantic_hits[:5]]
            return GraphAnswer(
                question_type="semantic_fallback",
                answer_text=f"Top graph evidence for question '{question}' retrieved semantically.",
                evidence=[_edge_to_evidence(self.index, edge) for edge in top],
                confidence=min(0.8, 0.4 + 0.08 * len(top)),
            )

        return GraphAnswer(
            question_type="unsupported",
            answer_text=(
                "I could not map this question to a supported temporal operator yet. "
                "Try adding explicit entities or time constraints."
            ),
            evidence=[],
            confidence=0.0,
        )

    def _answer_state(self, plan: Dict[str, Any], semantic_hits: Sequence[ScoredEdge]) -> GraphAnswer:
        entities = [str(v) for v in (plan.get("entities") or [])]
        year = plan.get("year")

        filtered = list(semantic_hits)
        if entities:
            entity_uids = set()
            for entity in entities:
                entity_uids.update(self.index.resolve_node_uids(entity))
            if entity_uids:
                filtered = [h for h in filtered if h.edge.source_uid in entity_uids or h.edge.target_uid in entity_uids]

        if year is not None:
            filtered = [h for h in filtered if h.edge.overlaps_year(int(year))]

        if not filtered:
            return self._semantic_backoff(
                question_type="state_at_time",
                semantic_hits=semantic_hits,
                detail="Could not find a strict temporal/entity state match.",
            )

        # Prefer meaningful (base) edges over structural span/tag bookkeeping edges
        meaningful = [
            h for h in filtered
            if h.edge.relation.lower() not in STRUCTURAL_RELATIONS
            and h.edge.edge_type in ("base", "gold_fact")
        ]
        # Also accept non-base but non-structural edges if base is empty
        if not meaningful:
            meaningful = [
                h for h in filtered
                if h.edge.relation.lower() not in STRUCTURAL_RELATIONS
            ]

        has_factual = bool(meaningful)
        display_hits = meaningful if meaningful else filtered

        top = [h.edge for h in display_hits[:6]]

        # §6: Canonical answer entity projection — extract entity names from
        # evidence so the LLM can quote a canonical name.
        canonical_entities: List[str] = []
        for edge in ([h.edge for h in meaningful[:6]] if meaningful else [h.edge for h in filtered[:10]]):
            for uid in (edge.source_uid, edge.target_uid):
                label = self.index.node_label(uid)
                cat = self.index.node_category_by_uid.get(uid, "")
                # Skip year/date-like nodes from canonical list
                if cat in ("date", "year") or label.isdigit():
                    continue
                if label not in canonical_entities:
                    canonical_entities.append(label)

        time_text = f" in {year}" if year is not None else ""

        if has_factual:
            summary = []
            for edge in top[:3]:
                summary.append(
                    f"{self.index.node_label(edge.source_uid)} {edge.relation} {self.index.node_label(edge.target_uid)}"
                )
            canonical_hint = ""
            if canonical_entities:
                canonical_hint = f" Key entities: {', '.join(canonical_entities[:5])}."
            return GraphAnswer(
                question_type="state_at_time",
                answer_text=f"State evidence{time_text}: " + "; ".join(summary) + f".{canonical_hint}",
                evidence=[_edge_to_evidence(self.index, edge) for edge in [h.edge for h in filtered[:6]]],
                confidence=min(0.9, 0.35 + 0.1 * len(top)),
            )

        # Only structural/temporal links found — provide entity hints but flag weakness
        canonical_hint = ""
        if canonical_entities:
            canonical_hint = f" Entities present in graph: {', '.join(canonical_entities[:6])}."
        return GraphAnswer(
            question_type="state_at_time",
            answer_text=(
                f"Only structural/temporal links found{time_text} (no factual relations)."
                f"{canonical_hint}"
                f" The model should rely on its own knowledge."
            ),
            evidence=[_edge_to_evidence(self.index, edge) for edge in [h.edge for h in filtered[:6]]],
            confidence=0.25,
        )

    def _answer_existence(self, plan: Dict[str, Any], semantic_hits: Sequence[ScoredEdge]) -> GraphAnswer:
        entities = [str(v) for v in (plan.get("entities") or [])]
        if len(entities) < 2:
            return GraphAnswer(
                question_type="existence_in_time",
                answer_text="Need at least two entities to evaluate existence.",
                evidence=[],
                confidence=0.0,
            )

        left = set(self.index.resolve_node_uids(entities[0]))
        right = set(self.index.resolve_node_uids(entities[1]))
        year = plan.get("year")

        matches: List[GraphEdge] = []
        for edge in [h.edge for h in semantic_hits]:
            if (edge.source_uid in left and edge.target_uid in right) or (edge.source_uid in right and edge.target_uid in left):
                if year is None or edge.overlaps_year(int(year)):
                    matches.append(edge)

        if not matches:
            period = f" in {year}" if year is not None else ""
            return self._semantic_backoff(
                question_type="existence_in_time",
                semantic_hits=semantic_hits,
                detail=f"Could not confirm a direct edge/path connection between the entities{period}.",
            )

        period = f" in {year}" if year is not None else ""

        # §6: Canonical entity projection for existence answers
        canonical_entities: List[str] = []
        for edge in matches[:5]:
            if edge.relation.lower() in STRUCTURAL_RELATIONS:
                continue
            for uid in (edge.source_uid, edge.target_uid):
                label = self.index.node_label(uid)
                cat = self.index.node_category_by_uid.get(uid, "")
                if cat in ("date", "year") or label.isdigit():
                    continue
                if label not in canonical_entities:
                    canonical_entities.append(label)

        detail_parts: List[str] = []
        for edge in matches[:3]:
            detail_parts.append(
                f"{self.index.node_label(edge.source_uid)} {edge.relation} "
                f"{self.index.node_label(edge.target_uid)}"
            )

        canonical_hint = ""
        if canonical_entities:
            canonical_hint = f" Key entities: {', '.join(canonical_entities[:5])}."
        detail_text = "; ".join(detail_parts) if detail_parts else ""

        return GraphAnswer(
            question_type="existence_in_time",
            answer_text=(
                f"Yes, there is graph evidence connecting the entities{period}."
                f" Evidence: {detail_text}.{canonical_hint}"
            ),
            evidence=[_edge_to_evidence(self.index, edge) for edge in matches[:5]],
            confidence=min(0.95, 0.5 + 0.08 * len(matches[:5])),
        )

    def _answer_first_last(
        self,
        plan: Dict[str, Any],
        semantic_hits: Sequence[ScoredEdge],
        earliest: bool,
    ) -> GraphAnswer:
        entities = [str(v) for v in (plan.get("entities") or [])]
        qtype = "first_occurrence" if earliest else "last_occurrence"

        def _edge_year(edge: GraphEdge) -> int:
            y = edge.start_year if edge.start_year is not None else edge.end_year
            return y if y is not None else (0 if earliest else 999999)

        # --- Two-entity comparative ordering ("what came first: A or B?") ---
        if len(entities) >= 2:
            per_entity: Dict[str, List[ScoredEdge]] = {}
            for entity in entities[:2]:
                e_uids = set(self.index.resolve_node_uids(entity, max_hits=8))
                if not e_uids:
                    continue
                anchored = [
                    h for h in semantic_hits
                    if (h.edge.start_year is not None or h.edge.end_year is not None)
                    and (h.edge.source_uid in e_uids or h.edge.target_uid in e_uids)
                    and h.edge.relation.lower() not in STRUCTURAL_RELATIONS
                ]
                # fall back to structural if nothing meaningful
                if not anchored:
                    anchored = [
                        h for h in semantic_hits
                        if (h.edge.start_year is not None or h.edge.end_year is not None)
                        and (h.edge.source_uid in e_uids or h.edge.target_uid in e_uids)
                    ]
                if anchored:
                    # ── KEY FIX: sort by retrieval score (descending) not by year ──
                    # The best-scored edge is most semantically relevant to the
                    # question; its date is the representative anchor.  Sorting by
                    # extreme year picked up unrelated edges (e.g. "president" node
                    # linking to 1847 edges for a question about Barack Obama).
                    anchored.sort(key=lambda h: h.score, reverse=True)
                    per_entity[entity] = anchored

            if len(per_entity) == 2:
                names = list(per_entity.keys())
                a_best = per_entity[names[0]][0]
                b_best = per_entity[names[1]][0]
                a_year = _edge_year(a_best.edge)
                b_year = _edge_year(b_best.edge)
                a_edge = a_best.edge
                b_edge = b_best.edge
                if a_year != b_year:
                    winner = names[0] if a_year < b_year else names[1]
                    winner_year = min(a_year, b_year)
                    loser_year = max(a_year, b_year)
                    answer_text = (
                        f"Based on graph evidence, '{names[0]}' is anchored at {a_year} "
                        f"and '{names[1]}' at {b_year}. "
                        f"{'Earlier' if earliest else 'Later'}: '{winner}' (around {winner_year})."
                    )
                else:
                    answer_text = (
                        f"Both '{names[0]}' and '{names[1]}' share the same earliest year ({a_year}) in graph evidence."
                    )
                evidence = [
                    _edge_to_evidence(self.index, a_edge),
                    _edge_to_evidence(self.index, b_edge),
                ]
                return GraphAnswer(
                    question_type=qtype,
                    answer_text=answer_text,
                    evidence=evidence,
                    confidence=0.72,
                )

        # --- Single-entity or no entities: find earliest/latest anchored hit ---
        # Take the top-scored hits first, then find earliest/latest among those.
        scored_with_dates = [
            hit for hit in semantic_hits
            if hit.edge.start_year is not None or hit.edge.end_year is not None
        ]
        if not scored_with_dates:
            return self._semantic_backoff(
                question_type=qtype,
                semantic_hits=semantic_hits,
                detail="Could not find time-anchored edges for this query.",
            )

        # Narrow to top-10 by retrieval score, then pick earliest/latest
        scored_with_dates.sort(key=lambda h: h.score, reverse=True)
        top_pool = scored_with_dates[:10]
        top_pool.sort(key=lambda h: _edge_year(h.edge), reverse=not earliest)
        best = top_pool[0].edge
        t = best.start or best.end or "unknown"
        source = self.index.node_label(best.source_uid)
        target = self.index.node_label(best.target_uid)
        prefix = "Earliest" if earliest else "Latest"
        return GraphAnswer(
            question_type=qtype,
            answer_text=f"{prefix} matching event appears at {t}: {source} {best.relation} {target}.",
            evidence=[_edge_to_evidence(self.index, h.edge) for h in top_pool[:5]],
            confidence=0.75,
        )

    def _answer_count(self, plan: Dict[str, Any], semantic_hits: Sequence[ScoredEdge]) -> GraphAnswer:
        entities = [str(v) for v in (plan.get("entities") or [])]
        year = plan.get("year")

        edges = [h.edge for h in semantic_hits]
        if entities:
            seed = set(self.index.resolve_node_uids(entities[0]))
            if seed:
                edges = [e for e in edges if e.source_uid in seed or e.target_uid in seed]
        if year is not None:
            edges = [e for e in edges if e.overlaps_year(int(year))]

        unique_pairs = {(edge.source_uid, edge.target_uid, edge.relation) for edge in edges}
        count = len(unique_pairs)
        period = f" in {year}" if year is not None else ""
        return GraphAnswer(
            question_type="temporal_count",
            answer_text=f"Count result{period}: {count} matching relation instances.",
            evidence=[_edge_to_evidence(self.index, edge) for edge in edges[:5]],
            confidence=0.7 if count > 0 else 0.2,
        )

    def _answer_top_k(self, plan: Dict[str, Any], semantic_hits: Sequence[ScoredEdge]) -> GraphAnswer:
        k = int(plan.get("k") or 5)
        year = plan.get("year")
        edges = [h.edge for h in semantic_hits]
        if year is not None:
            edges = [e for e in edges if e.overlaps_year(int(year))]

        score_by_node: Dict[str, int] = {}
        for edge in edges:
            score_by_node[edge.source_uid] = score_by_node.get(edge.source_uid, 0) + 1
            score_by_node[edge.target_uid] = score_by_node.get(edge.target_uid, 0) + 1

        ranking = sorted(score_by_node.items(), key=lambda kv: kv[1], reverse=True)[: max(1, k)]
        if not ranking:
            return GraphAnswer(
                question_type="temporal_top_k",
                answer_text="No candidates found for top-k ranking.",
                evidence=[],
                confidence=0.2,
            )

        labels = [f"{self.index.node_label(uid)} ({score})" for uid, score in ranking]
        return GraphAnswer(
            question_type="temporal_top_k",
            answer_text=f"Top-{len(ranking)} entities: " + ", ".join(labels) + ".",
            evidence=[_edge_to_evidence(self.index, edge) for edge in edges[:5]],
            confidence=0.7,
        )

    def _answer_path_existence(self, plan: Dict[str, Any]) -> GraphAnswer:
        entities = [str(v) for v in (plan.get("entities") or [])]
        if len(entities) < 2:
            return GraphAnswer(
                question_type="temporal_path_existence",
                answer_text="Need source and target entities to test path existence.",
                evidence=[],
                confidence=0.0,
            )

        src_uids = self.index.resolve_node_uids(entities[0])
        dst_uids = set(self.index.resolve_node_uids(entities[1]))
        if not src_uids or not dst_uids:
            return GraphAnswer(
                question_type="temporal_path_existence",
                answer_text="Could not resolve one or both path endpoints.",
                evidence=[],
                confidence=0.0,
            )

        year = plan.get("year")
        max_hops = int(plan.get("max_hops") or 4)

        visited = set(src_uids)
        frontier = list(src_uids)
        depth = 0
        witness: List[GraphEdge] = []
        while frontier and depth < max_hops:
            next_frontier: List[str] = []
            for node_uid in frontier:
                for edge in self.index.outgoing_edges(node_uid):
                    if year is not None and not edge.overlaps_year(int(year)):
                        continue
                    nxt = edge.target_uid
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    next_frontier.append(nxt)
                    witness.append(edge)
            if any(node in dst_uids for node in next_frontier):
                period = f" in {year}" if year is not None else ""
                return GraphAnswer(
                    question_type="temporal_path_existence",
                    answer_text=f"A path exists between the entities{period} within {depth + 1} hops.",
                    evidence=[_edge_to_evidence(self.index, edge) for edge in witness[:8]],
                    confidence=0.75,
                )
            frontier = next_frontier
            depth += 1

        period = f" in {year}" if year is not None else ""
        return GraphAnswer(
            question_type="temporal_path_existence",
            answer_text=f"No path evidence found between the entities{period} within {max_hops} hops.",
            evidence=[_edge_to_evidence(self.index, edge) for edge in witness[:8]],
            confidence=0.3,
        )

    def reason_of_in_year(self, entity: str, year: int) -> GraphAnswer:
        target_uids = self.index.resolve_node_uids(entity)
        if not target_uids:
            return GraphAnswer(
                question_type="reason_of",
                answer_text=f"No node found for '{entity}'.",
                evidence=[],
                confidence=0.0,
            )

        matches: List[GraphEdge] = []
        for target_uid in target_uids:
            for edge in self.index.incoming_edges(target_uid):
                if edge.edge_type not in ("base", "gold_fact"):
                    continue
                if edge.relation.lower() not in CAUSAL_RELATIONS:
                    continue
                if not edge.overlaps_year(year):
                    continue
                matches.append(edge)

        if not matches:
            return GraphAnswer(
                question_type="reason_of",
                answer_text=f"No causal evidence found for '{entity}' in {year}.",
                evidence=[],
                confidence=0.15,
            )

        matches.sort(key=lambda e: e.support_count, reverse=True)
        top = matches[:5]
        reasons = [self.index.node_label(edge.source_uid) for edge in top]
        reasons_text = ", ".join(dict.fromkeys(reasons))
        text = f"Likely reasons for '{entity}' in {year}: {reasons_text}."
        return GraphAnswer(
            question_type="reason_of",
            answer_text=text,
            evidence=[_edge_to_evidence(self.index, edge) for edge in top],
            confidence=min(0.95, 0.45 + 0.1 * len(top)),
        )

    def analogical_outcome(self, a: str, b: str, c: str, year: int) -> GraphAnswer:
        a_uids = self.index.resolve_node_uids(a)
        b_uids = self.index.resolve_node_uids(b)
        c_uids = self.index.resolve_node_uids(c)
        if not a_uids or not b_uids or not c_uids:
            return GraphAnswer(
                question_type="analogical_transfer",
                answer_text="Could not resolve one or more entities (A, B, C).",
                evidence=[],
                confidence=0.0,
            )

        reference: Optional[GraphEdge] = None
        for a_uid in a_uids:
            for edge in self.index.outgoing_edges(a_uid):
                if edge.edge_type not in ("base", "gold_fact"):
                    continue
                if edge.target_uid not in b_uids:
                    continue
                if not edge.overlaps_year(year):
                    continue
                reference = edge
                break
            if reference is not None:
                break

        if reference is None:
            return GraphAnswer(
                question_type="analogical_transfer",
                answer_text=f"No baseline event found for '{a}' -> '{b}' in {year}.",
                evidence=[],
                confidence=0.1,
            )

        comparable: List[GraphEdge] = []
        relation = reference.relation.lower()
        for a_uid in a_uids:
            for edge in self.index.outgoing_edges(a_uid):
                if edge.edge_type not in ("base", "gold_fact"):
                    continue
                if edge.target_uid not in c_uids:
                    continue
                if edge.relation.lower() != relation:
                    continue
                comparable.append(edge)

        if comparable:
            comparable.sort(key=lambda e: e.support_count, reverse=True)
            best = comparable[0]
            when = best.start or best.end or "unknown time"
            answer_text = (
                f"There is evidence that a similar event happened: '{a}' {best.relation} '{c}' "
                f"(time: {when})."
            )
            confidence = 0.8
            evidence = [_edge_to_evidence(self.index, reference), _edge_to_evidence(self.index, best)]
        else:
            answer_text = (
                f"No direct evidence that '{a}' causing '{b}' in {year} also happened to '{c}'."
            )
            confidence = 0.3
            evidence = [_edge_to_evidence(self.index, reference)]

        return GraphAnswer(
            question_type="analogical_transfer",
            answer_text=answer_text,
            evidence=evidence,
            confidence=confidence,
        )

    def when_started_affecting(self, a: str, b: str) -> GraphAnswer:
        a_uids = self.index.resolve_node_uids(a)
        b_uids = self.index.resolve_node_uids(b)
        if not a_uids or not b_uids:
            return GraphAnswer(
                question_type="start_affecting",
                answer_text="Could not resolve source or target entity.",
                evidence=[],
                confidence=0.0,
            )

        candidate_edges: List[GraphEdge] = []
        for a_uid in a_uids:
            for edge in self.index.outgoing_edges(a_uid):
                if edge.edge_type not in ("base", "gold_fact"):
                    continue
                if edge.target_uid not in b_uids:
                    continue
                if edge.relation.lower() not in AFFECT_RELATIONS:
                    continue
                candidate_edges.append(edge)

        if not candidate_edges:
            return GraphAnswer(
                question_type="start_affecting",
                answer_text=f"No affecting edge found between '{a}' and '{b}'.",
                evidence=[],
                confidence=0.2,
            )

        candidate_edges.sort(
            key=lambda e: (
                e.start_year if e.start_year is not None else 999999,
                -(e.support_count or 0),
            )
        )
        best = candidate_edges[0]
        start_text = best.start or best.end or "unknown"
        return GraphAnswer(
            question_type="start_affecting",
            answer_text=f"'{a}' appears to start affecting '{b}' around {start_text}.",
            evidence=[_edge_to_evidence(self.index, edge) for edge in candidate_edges[:5]],
            confidence=0.75 if best.start else 0.45,
        )
