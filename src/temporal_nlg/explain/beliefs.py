"""Belief tracking and justification utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class Evidence:
    """Atomic evidence snippet supporting or refuting a claim."""

    source: str
    snippet: str
    weight: float = 1.0
    timestamp: Optional[str] = None


@dataclass
class BeliefRecord:
    """Tracked belief with justification."""

    claim_id: str
    claim: str
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_evidence(self, ev: Evidence) -> None:
        self.evidence.append(ev)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def justify(self) -> str:
        if not self.evidence:
            return f"Claim '{self.claim}' (conf {self.confidence:.2f}) with no evidence."
        parts = []
        for ev in self.evidence:
            stamp = f" @{ev.timestamp}" if ev.timestamp else ""
            parts.append(f"[{ev.source}{stamp}] {ev.snippet} (w={ev.weight:.1f})")
        return f"Claim '{self.claim}' (conf {self.confidence:.2f}): " + "; ".join(parts)


class BeliefTracker:
    """In-memory belief tracker with lightweight justification support."""

    def __init__(self):
        self._beliefs: Dict[str, BeliefRecord] = {}

    def upsert_belief(
        self,
        claim_id: str,
        claim: str,
        confidence: float,
        evidence: Optional[List[Evidence]] = None,
    ) -> BeliefRecord:
        confidence = max(0.0, min(1.0, confidence))
        record = self._beliefs.get(claim_id)
        if not record:
            record = BeliefRecord(claim_id=claim_id, claim=claim, confidence=confidence)
            self._beliefs[claim_id] = record
        else:
            record.claim = claim
            record.confidence = confidence
            record.updated_at = datetime.now(timezone.utc).isoformat()
        for ev in evidence or []:
            record.add_evidence(ev)
        return record

    def get_belief(self, claim_id: str) -> Optional[BeliefRecord]:
        return self._beliefs.get(claim_id)

    def justify(self, claim_id: str) -> str:
        record = self.get_belief(claim_id)
        if not record:
            return f"No belief found for id '{claim_id}'."
        return record.justify()

    def top_beliefs(self, k: int = 5) -> List[BeliefRecord]:
        return sorted(self._beliefs.values(), key=lambda b: b.confidence, reverse=True)[:k]
