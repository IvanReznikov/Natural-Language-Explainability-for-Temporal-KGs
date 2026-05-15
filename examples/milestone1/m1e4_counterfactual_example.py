#!/usr/bin/env python3
"""
Milestone 1: Counterfactual Example

Shows factual vs counterfactual generation and TMS counterfactual shifts/swaps.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg.explain.counterfactuals import CounterfactualGenerator, Fact
from temporal_nlg.tms.counterfactual import CounterfactualEngine
from temporal_nlg.tms.belief_store import Belief


def run_counterfactuals() -> dict:
    factual = Fact(subject="Policy", predicate="reduced", obj="emissions", timeframe="2024")
    alternative = Fact(subject="Policy", predicate="increased", obj="emissions", timeframe="2024")

    gen = CounterfactualGenerator()
    cf_text = gen.generate(factual, alternative)

    engine = CounterfactualEngine()
    belief = Belief(
        belief_id="b1",
        payload={"event": "emissions reduced"},
        evidence=[{"source": "report", "snippet": "CO2 down 5%", "weight": 0.9}],
    )
    shifted = engine.shift_time(belief, "+3m")
    swapped = engine.swap_order(belief, Belief(belief_id="b2", payload={"event": "investment"}))

    return {
        "counterfactual_text": cf_text,
        "shifted": shifted,
        "swapped": swapped,
    }


def main() -> None:
    print("=" * 70)
    print("Milestone 1 - Counterfactual Example")
    print("=" * 70)
    results = run_counterfactuals()

    print("\nCounterfactual generation:")
    for k, v in results["counterfactual_text"].items():
        print(f"- {k}: {v}")

    print("\nTMS shift_time:")
    print(results["shifted"].description)

    print("\nTMS swap_order:")
    for item in results["swapped"]:
        print(f"- {item.description}")


if __name__ == "__main__":
    main()
