#!/usr/bin/env python3
"""M1-E4 TMS-lite evaluation: belief tracking and justifications."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from temporal_nlg.tms import Belief, BeliefStore, JustificationBuilder


def main():
    store = BeliefStore()
    jb = JustificationBuilder()

    b1 = Belief(belief_id="b1", payload={"event": "A"})
    b2 = Belief(belief_id="b2", payload={"event": "B"}, supports=["b1"])
    b3 = Belief(belief_id="b3", payload={"event": "C"}, supports=["b2"])

    store.add_belief(b1)
    store.add_belief(b2)
    store.add_belief(b3)

    # Retract b1 to test propagation
    store.retract("b1")

    results = {
        "active": [b.belief_id for b in store.get_active_beliefs()],
        "dirty": [b.belief_id for b in store.get_dirty_beliefs()],
        "justifications": jb.build(b3, [b2]),
    }

    out_dir = Path(__file__).resolve().parents[2] / "output" / "m1_e4_tms"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "tms_report.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved TMS report to {out_file}")
    print(results)


if __name__ == "__main__":
    main()
