from __future__ import annotations

import re
from typing import Dict, List

from temporal_nlg.graph_query.retrieval import GraphAnswer


def _safe_id(label: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9_]+", "_", label).strip("_")
    if not key:
        key = "node"
    if key[0].isdigit():
        key = f"n_{key}"
    return key


def answer_to_mermaid(answer: GraphAnswer, max_edges: int = 12) -> str:
    """Render answer evidence as Mermaid graph text (built-in, no external DB)."""

    lines: List[str] = ["graph TD"]
    nodes_seen: Dict[str, str] = {}

    for ev in answer.evidence[:max_edges]:
        source = str(ev.get("source") or "source")
        target = str(ev.get("target") or "target")
        relation = str(ev.get("relation") or "related_to")
        start = ev.get("start")
        end = ev.get("end")

        source_id = nodes_seen.setdefault(source, _safe_id(source))
        target_id = nodes_seen.setdefault(target, _safe_id(target))

        edge_label = relation
        if start or end:
            edge_label = f"{relation} ({start or '?'} → {end or '?'})"

        lines.append(f"    {source_id}[\"{source}\"] -->|\"{edge_label}\"| {target_id}[\"{target}\"]")

    if len(lines) == 1:
        lines.append("    empty[\"No evidence edges\"]")

    return "\n".join(lines)
