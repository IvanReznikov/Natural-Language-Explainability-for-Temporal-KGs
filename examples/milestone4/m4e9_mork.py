"""Milestone 4 — Example 9: Temporal facts in the MORK atomspace.

Demonstrates the MORK side of the integration: temporal edges are stored as
S-expression atoms in a live MORK atomspace, retrieved by pattern-matched
export, and joined via a server-side transform — the storage/retrieval layer
that complements the hyperon-hosted Python reasoning pipeline.

Requires a running MORK HTTP server (default ``http://127.0.0.1:8000``,
override with ``MORK_SERVER_URL``). The example uses its own namespace and
clears it on exit, leaving the shared atomspace as it was found. It skips
cleanly when no server is reachable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg_metta import MORKHttpRunner, mork_http_available

NAMESPACE = "m4demo"

# Temporal edges about the Model T (single-token labels: MORK parses each
# field as one S-expression token).
EDGES = [
    f"({NAMESPACE} Ford introduced moving_assembly_line 1913)",
    f"({NAMESPACE} moving_assembly_line caused Model_T_price_drop 1913)",
    f"({NAMESPACE} Model_T_price_drop reduced car_prices 1914)",
]


def _wait_export(
    runner: MORKHttpRunner, pattern: str, template: str, timeout_s: float = 5.0
) -> str:
    deadline = time.monotonic() + timeout_s
    out = ""
    while time.monotonic() < deadline:
        out = runner.export(pattern, template)
        if out.strip():
            return out
        time.sleep(0.1)
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not mork_http_available():
        print(
            "[skip] No MORK HTTP server reachable (looked at "
            f"{MORKHttpRunner.DEFAULT_URL}; override with MORK_SERVER_URL).\n"
            "       Start MORK's server binary and re-run this example."
        )
        return 0

    runner = MORKHttpRunner(timeout_s=10.0)
    pattern = f"({NAMESPACE} $s $r $t $y)"

    print("=" * 70)
    print("M4 Example 9 — Temporal facts in the MORK atomspace")
    print("=" * 70)
    try:
        print(f"\n1. Upload {len(EDGES)} temporal edges under ({NAMESPACE} ...):")
        for edge in EDGES:
            print("   ", edge)
        runner.upload(pattern, pattern, "\n".join(EDGES))

        print("\n2. Pattern-matched export — all edges:")
        print(_wait_export(runner, pattern, "($s $r $t $y)"))

        print("3. Pattern-matched export — only edges caused by something:")
        print(_wait_export(runner, f"({NAMESPACE} $s caused $t $y)", "($s $t $y)"))

        print("4. Server-side transform: project (src, tgt) causal pairs:")
        runner.transform(
            f"(transform (, ({NAMESPACE} $s caused $t $y)) (, ({NAMESPACE}_causal $s $t)))"
        )
        print(_wait_export(runner, f"({NAMESPACE}_causal $s $t)", "($s $t)"))
    finally:
        runner.clear(pattern)
        runner.clear(f"({NAMESPACE}_causal $s $t)")
        print(f"\n5. Cleaned up the {NAMESPACE} namespace.")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
