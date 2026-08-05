#!/usr/bin/env python3
"""M1-E4 counterfactual reasoning quick check."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from temporal_nlg.tms import Belief, CounterfactualEngine


def main():
    engine = CounterfactualEngine()
    b = Belief(belief_id="b_market", payload={"event": "Market rose", "date": "2010"})
    cf = engine.shift_time(b, "earlier")
    swaps = engine.swap_order(
        b, Belief(belief_id="b_policy", payload={"event": "Policy changed", "date": "2009"})
    )

    results = {
        "shift": {
            "desc": cf.description,
            "new_belief": cf.new_belief.belief_id,
            "payload": cf.new_belief.payload,
        },
        "swaps": [
            {
                "desc": s.description,
                "new_belief": s.new_belief.belief_id,
                "payload": s.new_belief.payload,
            }
            for s in swaps
        ],
    }

    out_dir = Path(__file__).resolve().parents[2] / "output" / "m1_e4_counterfactual"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "counterfactual_report.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved counterfactual report to {out_file}")
    print(results)


if __name__ == "__main__":
    main()
