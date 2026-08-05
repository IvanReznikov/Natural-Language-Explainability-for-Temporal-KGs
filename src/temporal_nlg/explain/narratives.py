"""Narrative rendering for graph paths with style adaptation."""

from __future__ import annotations

from typing import List, Dict, Optional
from .graph_paths import GraphNode, GraphEdge, GraphPathExplanation


class PathNarrativeRenderer:
    """Builds human-readable narratives from graph paths."""

    def __init__(self, style: str = "neutral", domain: str = "general"):
        self.style = style
        self.domain = domain
        self.explainer = GraphPathExplanation()

    def render(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        evidence_ids: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        base = self.explainer.explain(nodes, edges)
        narrative = self._compose(base["path"], base["summary"], evidence_ids)
        return {
            "summary": base["summary"],
            "narrative": narrative,
            "justification": base["justification"],
        }

    def _compose(self, path: str, summary: str, evidence_ids: Optional[List[str]]) -> str:
        intro = self._intro()
        body = self._body(path, summary)
        outro = self._outro()
        evidence = f" Evidence chain: {', '.join(evidence_ids)}." if evidence_ids else ""
        return " ".join(filter(None, [intro, body, outro])) + evidence

    def _intro(self) -> str:
        if self.style == "novice":
            return "Here is the timeline in plain terms."
        if self.style == "expert":
            return "Tracing the dependency path:"
        return "Timeline overview:"

    def _body(self, path: str, summary: str) -> str:
        if self.domain == "medical":
            return f"Clinical sequence: {path}. Interpretation: {summary}"
        if self.domain == "finance":
            return f"Event flow: {path}. Impact: {summary}"
        if self.style == "novice":
            return f"Step by step: {path}. It means {summary}"
        if self.style == "expert":
            return f"Path {path}. Implication: {summary}"
        return f"Path {path}. {summary}"

    def _outro(self) -> str:
        if self.style == "novice":
            return "That is the key takeaway."
        if self.style == "expert":
            return "Summary complete."
        return ""
