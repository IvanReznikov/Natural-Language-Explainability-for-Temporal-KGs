"""Meta-queries over rule firing traces (M2-E5)."""
from __future__ import annotations

from typing import Dict, List, Tuple

from .trace import QueryTrace
from .trace_explain import TraceJustifier
from .contradiction import ContradictionDetector


def rules_fired(trace: QueryTrace) -> List[str]:
    return [rt.rule_id for rt in trace.rule_traces if not trace.dropped]


def why_not_fired(trace: QueryTrace, expected_rule_ids: List[str]) -> Dict[str, str]:
    fired = set(rules_fired(trace))
    result = {}
    for rid in expected_rule_ids:
        if rid in fired:
            result[rid] = "fired"
        else:
            result[rid] = "absent in trace"
    return result


def influential_facts(trace: QueryTrace, top_k: int = 5) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for rt in trace.rule_traces:
        for inp in rt.inputs:
            fid = inp.get("fact_id") if isinstance(inp, dict) else None
            if fid:
                counts[fid] = counts.get(fid, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]


def explain_fact(trace: QueryTrace, fact_id: str) -> str:
    return TraceJustifier(trace).explain_fact(fact_id)


def contradictions(trace: QueryTrace):
    return [c.to_dict() for c in ContradictionDetector().detect(trace)]


__all__ = [
    "rules_fired",
    "why_not_fired",
    "influential_facts",
    "explain_fact",
    "contradictions",
]
