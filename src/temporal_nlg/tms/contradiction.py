"""Contradiction detection and root-cause surfacing for traces (M2-E5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .trace import QueryTrace, RuleTrace


@dataclass
class Contradiction:
    fact_id: str
    values: List
    rule_ids: List[str]
    reason: str

    def to_dict(self) -> Dict:
        return {
            "fact_id": self.fact_id,
            "values": self.values,
            "rule_ids": self.rule_ids,
            "reason": self.reason,
        }


class ContradictionDetector:
    """Detects value conflicts for facts concluded by different rules."""

    def detect(self, trace: QueryTrace) -> List[Contradiction]:
        fact_to_conclusions: Dict[str, List[RuleTrace]] = {}
        for rt in trace.rule_traces:
            fid = rt.conclusion.get("fact_id") if isinstance(rt.conclusion, dict) else None
            if not fid:
                continue
            fact_to_conclusions.setdefault(fid, []).append(rt)

        contradictions: List[Contradiction] = []
        for fid, rules in fact_to_conclusions.items():
            values = self._collect_values(rules)
            if len(values) <= 1:
                continue
            unique_vals = list({v for v in values if v is not None})
            if len(unique_vals) <= 1:
                continue
            rule_ids = [r.rule_id for r in rules]
            contradictions.append(
                Contradiction(
                    fact_id=fid,
                    values=unique_vals,
                    rule_ids=rule_ids,
                    reason=f"Conflicting values for {fid}: {unique_vals}",
                )
            )
        return contradictions

    @staticmethod
    def _collect_values(rules: List[RuleTrace]) -> List:
        vals = []
        for r in rules:
            if isinstance(r.conclusion, dict) and "value" in r.conclusion:
                vals.append(r.conclusion.get("value"))
        return vals


__all__ = ["Contradiction", "ContradictionDetector"]
