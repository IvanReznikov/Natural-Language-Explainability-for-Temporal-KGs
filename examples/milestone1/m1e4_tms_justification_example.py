#!/usr/bin/env python3
"""
Milestone 1: TMS Justification Example

Builds a belief chain with supports and renders a why-chain justification.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg.tms.belief_store import BeliefStore, Belief
from temporal_nlg.tms.justification import JustificationBuilder
from temporal_nlg.explain.justified_render import JustifiedRenderer


def build_justification() -> dict:
    store = BeliefStore()
    store.add_belief(
        Belief(
            belief_id="b1",
            payload={"text": "Launch readiness confirmed"},
            evidence=[{"source": "checklist", "snippet": "All systems go", "weight": 1.0}],
        )
    )
    store.add_belief(
        Belief(
            belief_id="b2",
            payload={"text": "Weather green"},
            supports=["b1"],
            evidence=[{"source": "wx", "snippet": "Winds nominal", "weight": 0.8}],
        )
    )

    renderer = JustifiedRenderer(store, JustificationBuilder())
    surface_text = "Mission launch is justified by pre-flight checks and weather."
    output = renderer.render_with_justification("b2", surface_text)
    return output


def main() -> None:
    print("=" * 70)
    print("Milestone 1 - TMS Justification Example")
    print("=" * 70)
    result = build_justification()
    print(result["text"])
    print(result["justification"])


if __name__ == "__main__":
    main()
