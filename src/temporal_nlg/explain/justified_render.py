"""Combine surface text with TMS justifications."""

from __future__ import annotations

from typing import Dict, Optional, List
from temporal_nlg.tms.belief_store import BeliefStore
from temporal_nlg.tms.justification import JustificationBuilder


class JustifiedRenderer:
    """Renders text plus a why-chain using the BeliefStore."""

    def __init__(self, belief_store: BeliefStore, builder: Optional[JustificationBuilder] = None):
        self.store = belief_store
        self.builder = builder or JustificationBuilder()

    def render_with_justification(self, belief_id: str, surface_text: str) -> Dict[str, str]:
        belief = self.store.get_belief(belief_id)
        if not belief:
            return {"text": surface_text, "justification": "Belief not found."}
        supports = [self.store.get_belief(s) for s in belief.supports if self.store.get_belief(s)]
        justification = self.builder.build(belief, supports)
        return {"text": surface_text, "justification": justification}
