from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


_YEAR_RE = re.compile(r"^-?\d{1,12}$")
_DATE_YEAR_RE = re.compile(r"^(-?\d{1,12})-\d{2}-\d{2}$")

# Canonical category set — normalises 40+ raw category strings to 10 stable labels.
# Anything not listed falls back to "concept".
_CATEGORY_CANONICAL: Dict[str, str] = {
    "person":         "person",
    "org":            "org",
    "group":          "org",
    "location":       "location",
    "event":          "event",
    "phase":          "event",
    "period":         "event",
    "interval":       "event",
    "product":        "product",
    "technology":     "product",
    "infrastructure": "product",
    "structure":      "product",
    "instrument":     "product",
    "drug":           "product",
    "treatment":      "product",
    "work":           "work",
    "style":          "work",
    "literature":     "work",
    "policy":         "policy",
    "strategy":       "policy",
    "process":        "policy",
    "metric":         "metric",
    "value":          "metric",
    "attribute":      "metric",
    "feature":        "metric",
    "capability":     "metric",
    # silenced in prompts — keep as-is so callers can filter them out
    "date":           "date",
    "tag":            "tag",
    # everything else → concept
}


def _norm(value: str) -> str:
    return value.strip().lower()


def _parse_year(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    if _YEAR_RE.match(value):
        return int(value)
    match = _DATE_YEAR_RE.match(value)
    if match:
        return int(match.group(1))
    return None


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            yield json.loads(payload)


@dataclass(frozen=True)
class GraphEdge:
    edge_uid: str
    source_uid: str
    target_uid: str
    relation: str
    start: Optional[str]
    end: Optional[str]
    edge_type: str
    support_count: int
    source_row_ids: List[str]

    @property
    def start_year(self) -> Optional[int]:
        return _parse_year(self.start)

    @property
    def end_year(self) -> Optional[int]:
        return _parse_year(self.end)

    def overlaps_year(self, year: int) -> bool:
        s_year = self.start_year
        e_year = self.end_year
        if s_year is None and e_year is None:
            return False
        if s_year is not None and e_year is not None:
            low, high = (s_year, e_year) if s_year <= e_year else (e_year, s_year)
            return low <= year <= high
        if s_year is not None:
            return s_year == year
        return bool(e_year == year)


class TemporalGraphIndex:
    """In-memory index over temporal graph output JSONL artifacts."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.nodes_path = self.output_dir / "nodes.jsonl"
        self.edges_path = self.output_dir / "edges.jsonl"
        self.tags_path = self.output_dir / "tags.jsonl"
        self.processed_graph_path = self.output_dir / "processed_graph.jsonl"

        self.node_label_by_uid: Dict[str, str] = {}
        self.node_category_by_uid: Dict[str, str] = {}
        self.node_uids_by_norm_label: Dict[str, List[str]] = {}

        self.edge_by_uid: Dict[str, GraphEdge] = {}
        self.outgoing_edge_uids: Dict[str, List[str]] = {}
        self.incoming_edge_uids: Dict[str, List[str]] = {}
        self.edge_uids_by_relation: Dict[str, List[str]] = {}

        self.all_years: Set[int] = set()

        self._load()

    def _load(self) -> None:
        if not self.nodes_path.exists() or not self.edges_path.exists():
            raise FileNotFoundError(
                f"Missing graph artifacts in {self.output_dir}. Expected nodes.jsonl and edges.jsonl."
            )

        for row in _iter_jsonl(self.nodes_path):
            node_uid = str(row.get("node_uid") or "")
            label = str(row.get("label") or "")
            raw_cat = str(row.get("category") or "unknown")
            category = _CATEGORY_CANONICAL.get(raw_cat, "concept")
            if not node_uid or not label:
                continue
            self.node_label_by_uid[node_uid] = label
            self.node_category_by_uid[node_uid] = category
            key = _norm(label)
            self.node_uids_by_norm_label.setdefault(key, []).append(node_uid)

        for row in _iter_jsonl(self.edges_path):
            edge = GraphEdge(
                edge_uid=str(row.get("edge_uid") or ""),
                source_uid=str(row.get("source_uid") or ""),
                target_uid=str(row.get("target_uid") or ""),
                relation=str(row.get("relation") or "related_to"),
                start=row.get("start"),
                end=row.get("end"),
                edge_type=str(row.get("edge_type") or "base"),
                support_count=int(row.get("support_count") or 1),
                source_row_ids=[str(v) for v in (row.get("source_row_ids") or [])],
            )
            if not edge.edge_uid or not edge.source_uid or not edge.target_uid:
                continue
            self.edge_by_uid[edge.edge_uid] = edge
            self.outgoing_edge_uids.setdefault(edge.source_uid, []).append(edge.edge_uid)
            self.incoming_edge_uids.setdefault(edge.target_uid, []).append(edge.edge_uid)
            self.edge_uids_by_relation.setdefault(_norm(edge.relation), []).append(edge.edge_uid)

            if edge.start_year is not None:
                self.all_years.add(edge.start_year)
            if edge.end_year is not None:
                self.all_years.add(edge.end_year)

    def resolve_node_uids(self, entity_name: str, max_hits: int = 10) -> List[str]:
        """Resolve entity names using exact and substring matching over node labels."""

        probe = _norm(entity_name)
        if not probe:
            return []

        exact = self.node_uids_by_norm_label.get(probe, [])
        if exact:
            return exact[:max_hits]

        hits_with_len = []
        for node_uid, label in self.node_label_by_uid.items():
            lnorm = _norm(label)
            if probe in lnorm or lnorm in probe:
                hits_with_len.append((node_uid, len(lnorm)))
                
        hits_with_len.sort(key=lambda item: item[1])
        return [uid for uid, _ in hits_with_len[:max_hits]]

    def node_label(self, node_uid: str) -> str:
        return self.node_label_by_uid.get(node_uid, node_uid)

    def node_category(self, node_uid: str) -> str:
        """Return the canonical category for *node_uid* (one of the 10 stable labels)."""
        return self.node_category_by_uid.get(node_uid, "concept")

    def outgoing_edges(self, node_uid: str) -> List[GraphEdge]:
        return [self.edge_by_uid[eid] for eid in self.outgoing_edge_uids.get(node_uid, [])]

    def incoming_edges(self, node_uid: str) -> List[GraphEdge]:
        return [self.edge_by_uid[eid] for eid in self.incoming_edge_uids.get(node_uid, [])]
