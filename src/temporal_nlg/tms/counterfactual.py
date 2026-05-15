"""Counterfactual transformations over beliefs with dependency carry-over."""

from typing import List, Dict
from dataclasses import dataclass
from .belief_store import Belief


@dataclass
class CounterfactualResult:
    original_id: str
    new_belief: Belief
    description: str


class CounterfactualEngine:
    def shift_time(self, belief: Belief, delta: str) -> CounterfactualResult:
        payload = dict(belief.payload)
        payload["counterfactual_time_shift"] = delta
        new_belief = Belief(
            belief_id=f"cf_{belief.belief_id}",
            payload=payload,
            supports=[belief.belief_id],
            evidence=list(belief.evidence),
        )
        desc = f"If time shifted by {delta}, belief {belief.belief_id} would become {new_belief.belief_id}."
        return CounterfactualResult(original_id=belief.belief_id, new_belief=new_belief, description=desc)

    def swap_order(self, belief_a: Belief, belief_b: Belief) -> List[CounterfactualResult]:
        results: List[CounterfactualResult] = []
        results.append(self.shift_time(belief_a, "earlier"))
        results.append(self.shift_time(belief_b, "later"))
        return results
