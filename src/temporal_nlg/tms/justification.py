"""Template-based justifications for beliefs with evidence surfacing."""

from typing import List
from .belief_store import Belief


class JustificationBuilder:
    """Render human-readable justifications for beliefs."""

    def build(self, belief: Belief, supports: List[Belief]) -> str:
        support_ids = ", ".join(s.belief_id for s in supports) if supports else "none"
        evidence_bits = []
        for ev in belief.evidence:
            src = ev.get("source", "unknown")
            snippet = ev.get("snippet", "")
            weight = ev.get("weight", "")
            evidence_bits.append(f"[{src}] {snippet} (w={weight})".strip())

        evidence_text = "; ".join(evidence_bits) if evidence_bits else "no direct evidence"
        return (
            f"Belief {belief.belief_id} is supported by {support_ids}. "
            f"Evidence: {evidence_text}."
        )
