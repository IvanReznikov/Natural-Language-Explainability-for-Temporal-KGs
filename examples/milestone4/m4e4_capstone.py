"""Milestone 4 — Example 4 (capstone): Explaining a temporal inference path.

Runs ``m4e4_capstone.metta``: the headline M4 deliverable that composes M3 graph
retrieval, M2 truth-maintained inference recording, and M1 natural-language
verbalization into a single explainable inference path driven from MeTTa.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg_metta import TemporalBridge, hyperon_available, run_metta_file


def main() -> int:
    # Atoms may contain Unicode (e.g. Mermaid arrows); avoid cp1252 console errors.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not hyperon_available():
        print("[skip] 'hyperon' is not installed. Install with: pip install 'temporal-nlg[metta]'")
        return 0

    bridge = TemporalBridge()
    metta_file = Path(__file__).with_suffix(".metta")

    print("=" * 70)
    print("M4 Example 4 (capstone) — Explainable temporal inference path")
    print("=" * 70)
    results = run_metta_file(metta_file, bridge=bridge)
    for block in results:
        for atom in block:
            print(atom)
    print("=" * 70)
    trace = bridge.trace_to_dict()
    print(
        "Recorded trace:",
        trace.get("query_id"),
        "with",
        len(trace.get("rule_traces", [])),
        "rule firings.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
