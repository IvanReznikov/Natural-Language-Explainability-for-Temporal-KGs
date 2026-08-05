"""Graph path explanation utilities.

This module produces human-readable explanations for linear paths through a graph.
It is intentionally lightweight so it can be reused by evaluators without
pulling in a graph DB dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GraphNode:
    """Lightweight graph node container."""

    node_id: str
    label: str
    attrs: Optional[Dict[str, str]] = None

    def short_label(self) -> str:
        base = self.label or self.node_id
        return base.strip()


@dataclass
class GraphEdge:
    """Lightweight graph edge container."""

    source: str
    target: str
    label: str
    timestamp: Optional[str] = None
    attrs: Optional[Dict[str, str]] = None

    def describe(self) -> str:
        parts = [self.label.strip()]
        if self.timestamp:
            parts.append(f"@{self.timestamp}")
        return " ".join(parts)


class GraphPathExplanation:
    """Generate readable explanations for graph paths."""

    def __init__(self, max_hops: int = 8):
        self.max_hops = max_hops

    def explain(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> Dict[str, str]:
        if not nodes:
            raise ValueError("Path must include at least one node")
        if len(nodes) < 2 or not edges:
            raise ValueError("Path must include at least two nodes and one edge")
        if len(nodes) - 1 != len(edges):
            raise ValueError("Edges count must be one less than nodes count for a linear path")
        if len(edges) > self.max_hops:
            raise ValueError("Path exceeds configured hop limit")

        steps = []
        for idx, edge in enumerate(edges):
            src = nodes[idx]
            dst = nodes[idx + 1]
            edge_text = edge.describe()
            steps.append(f"{src.short_label()} -[{edge_text}]-> {dst.short_label()}")

        summary = self._build_summary(nodes, edges)
        justification = self._build_justification(nodes, edges)

        return {
            "summary": summary,
            "path": " | ".join(steps),
            "justification": justification,
        }

    def _build_summary(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> str:
        if not edges:
            return nodes[0].short_label()
        start = nodes[0].short_label()
        end = nodes[-1].short_label()
        rels = {edge.label.strip() for edge in edges if edge.label}
        rel_text = ", ".join(sorted(rels)) or "related"
        return f"Connection from {start} to {end} via {rel_text}."

    def _build_justification(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> str:
        facts = []
        for edge in edges:
            timestamp = f" at {edge.timestamp}" if edge.timestamp else ""
            facts.append(f"{edge.source} {edge.label} {edge.target}{timestamp}")
        return "; ".join(facts)

    def render_text(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> str:
        explanation = self.explain(nodes, edges)
        return (
            f"Summary: {explanation['summary']}\n"
            f"Path: {explanation['path']}\n"
            f"Justification: {explanation['justification']}"
        )
