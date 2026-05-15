"""Counterfactual reasoning helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Fact:
    """Simple fact container for counterfactual generation."""
    subject: str
    predicate: str
    obj: str
    timeframe: Optional[str] = None

    def as_text(self) -> str:
        base = f"{self.subject} {self.predicate} {self.obj}"
        if self.timeframe:
            base += f" during {self.timeframe}"
        return base


class CounterfactualGenerator:
    """Create lightweight counterfactual statements."""

    def generate(self, factual: Fact, alternative: Fact) -> Dict[str, str]:
        factual_text = factual.as_text()
        alt_text = alternative.as_text()
        delta = self._compute_delta(factual, alternative)
        counterfactual = (
            f"If instead {alt_text}, then the outcome would diverge from the factual path."
        )
        return {
            "factual": factual_text,
            "counterfactual": counterfactual,
            "delta": delta,
        }

    def _compute_delta(self, factual: Fact, alternative: Fact) -> str:
        diffs = []
        if factual.subject != alternative.subject:
            diffs.append(f"subject changed from '{factual.subject}' to '{alternative.subject}'")
        if factual.predicate != alternative.predicate:
            diffs.append(f"predicate changed from '{factual.predicate}' to '{alternative.predicate}'")
        if factual.obj != alternative.obj:
            diffs.append(f"object changed from '{factual.obj}' to '{alternative.obj}'")
        if factual.timeframe != alternative.timeframe:
            diffs.append(f"timeframe changed from '{factual.timeframe}' to '{alternative.timeframe}'")
        return "; ".join(diffs) or "no change"
