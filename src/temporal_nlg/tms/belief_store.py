"""
Minimal temporal belief store with dependency tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional


@dataclass
class Belief:
    belief_id: str
    payload: dict
    supports: List[str] = field(default_factory=list)
    evidence: List[Dict[str, str]] = field(default_factory=list)
    status: str = "active"  # active | dirty | retracted


@dataclass
class SupportLink:
    source: str
    target: str


class BeliefStore:
    def __init__(self):
        self.beliefs: Dict[str, Belief] = {}
        self.dependencies: Dict[str, Set[str]] = {}  # source -> set(targets)

    def add_belief(self, belief: Belief):
        self.beliefs[belief.belief_id] = belief
        for s in belief.supports:
            self.dependencies.setdefault(s, set()).add(belief.belief_id)

    def get_belief(self, belief_id: str) -> Optional[Belief]:
        return self.beliefs.get(belief_id)

    def add_support(self, belief_id: str, support_id: str):
        belief = self.get_belief(belief_id)
        if not belief:
            return
        if support_id not in belief.supports:
            belief.supports.append(support_id)
        self.dependencies.setdefault(support_id, set()).add(belief_id)

    def add_evidence(self, belief_id: str, evidence: Dict[str, str]):
        belief = self.get_belief(belief_id)
        if not belief:
            return
        belief.evidence.append(evidence)

    def retract(self, belief_id: str):
        if belief_id not in self.beliefs:
            return
        self.beliefs[belief_id].status = "retracted"
        for child in self.dependencies.get(belief_id, set()):
            self.mark_dirty(child)

    def mark_dirty(self, belief_id: str):
        if belief_id in self.beliefs and self.beliefs[belief_id].status == "active":
            self.beliefs[belief_id].status = "dirty"
            for child in self.dependencies.get(belief_id, set()):
                self.mark_dirty(child)

    def get_active_beliefs(self) -> List[Belief]:
        return [b for b in self.beliefs.values() if b.status == "active"]

    def get_dirty_beliefs(self) -> List[Belief]:
        return [b for b in self.beliefs.values() if b.status == "dirty"]

    def get_support_chain(self, belief_id: str) -> List[Belief]:
        """Return breadth-first support chain as a list."""
        visited = set()
        queue = [belief_id]
        chain: List[Belief] = []
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            belief = self.get_belief(current)
            if belief:
                chain.append(belief)
                queue.extend(belief.supports)
        return chain
