"""Milestone 4 — Example 8: Temporal reasoning IN MeTTa.

Runs ``m4e8_metta_reasoning.metta``: temporal edges are injected as matchable
atoms, MeTTa's own pattern matcher derives the causal link, and that derived
conclusion — not a scripted literal — is recorded into the M2 trace and
explained. Runs without any model servers.
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
    print("M4 Example 8 — Temporal reasoning in MeTTa (match-derived trace)")
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
        "rule firing(s) derived by MeTTa match.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
