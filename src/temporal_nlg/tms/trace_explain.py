"""Justification path extraction from QueryTrace events (M2-E5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .trace import QueryTrace, RuleTrace


@dataclass
class JustificationPath:
    """Represents a chain of rules leading to a conclusion fact."""

    conclusion_fact: str
    rule_sequence: List[RuleTrace]

    def as_text(self) -> str:
        parts = []
        for rt in self.rule_sequence:
            inputs_text = ", ".join(_format_fact(inp) for inp in rt.inputs)
            conclusion_text = _format_fact(rt.conclusion)
            parts.append(f"{rt.rule_name}: {inputs_text} -> {conclusion_text}")
        return " | ".join(parts)


class TraceJustifier:
    """Extracts justification paths and textual rationales from a trace."""

    def __init__(self, trace: QueryTrace):
        self.trace = trace
        self.fact_producers: Dict[str, List[RuleTrace]] = self._build_fact_index(trace.rule_traces)

    def list_conclusions(self) -> List[str]:
        return [
            self._fact_id(rt.conclusion)
            for rt in self.trace.rule_traces
            if self._fact_id(rt.conclusion)
        ]

    def paths_for_fact(self, fact_id: str, max_depth: int = 6) -> List[JustificationPath]:
        paths: List[JustificationPath] = []
        rule_seen: set = set()
        fact_seen: set = set()

        def dfs(current_fact: str, chain: List[RuleTrace], depth: int):
            if depth > max_depth:
                return
            producers = self.fact_producers.get(current_fact, [])
            if not producers:
                if chain:
                    paths.append(
                        JustificationPath(conclusion_fact=fact_id, rule_sequence=list(chain))
                    )
                return
            for rt in producers:
                if rt.rule_id in rule_seen:
                    continue
                rule_seen.add(rt.rule_id)
                inputs = [self._fact_id(inp) for inp in rt.inputs if self._fact_id(inp)]
                if not inputs:
                    paths.append(
                        JustificationPath(conclusion_fact=fact_id, rule_sequence=list(chain + [rt]))
                    )
                else:
                    for upstream in inputs:
                        if upstream in fact_seen:
                            continue
                        fact_seen.add(upstream)
                        dfs(upstream, chain + [rt], depth + 1)
                        fact_seen.remove(upstream)
                rule_seen.remove(rt.rule_id)

        dfs(fact_id, [], 0)
        # Deduplicate by rule_id sequence order
        unique = {}
        for p in paths:
            key = tuple(r.rule_id for r in p.rule_sequence)
            if key not in unique:
                unique[key] = p
        return list(unique.values())

    def explain_fact(self, fact_id: str) -> str:
        paths = self.paths_for_fact(fact_id)
        if not paths:
            return f"No justification paths found for {fact_id}."
        textual = [p.as_text() for p in paths]
        return " || ".join(textual)

    def _build_fact_index(self, rule_traces: List[RuleTrace]) -> Dict[str, List[RuleTrace]]:
        fact_index: Dict[str, List[RuleTrace]] = {}
        for rt in rule_traces:
            fid = self._fact_id(rt.conclusion)
            if fid:
                fact_index.setdefault(fid, []).append(rt)
        return fact_index

    @staticmethod
    def _fact_id(fact: Dict) -> Optional[str]:
        if not isinstance(fact, dict):
            return None
        return fact.get("fact_id")


def _format_fact(fact: Dict) -> str:
    if not isinstance(fact, dict):
        return str(fact)
    fid = fact.get("fact_id", "fact")
    value = fact.get("value")
    if value is None:
        return fid
    return f"{fid}={value}"


__all__ = ["TraceJustifier", "JustificationPath"]
