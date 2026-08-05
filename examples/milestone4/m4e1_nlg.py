"""Milestone 4 — Example 1: NLG from MeTTa (M1 surface).

Runs ``m4e1_nlg.metta`` through the temporal-nlg bridge and prints each
grounded-op result. Requires the optional ``hyperon`` dependency::

    pip install 'temporal-nlg[metta]'
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
    print("M4 Example 1 — Natural-language generation from MeTTa (M1)")
    print("=" * 70)
    results = run_metta_file(metta_file, bridge=bridge)
    for block in results:
        for atom in block:
            print(atom)
    print("=" * 70)
    print(f"Generator strategy routing used: {bridge.generator.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
