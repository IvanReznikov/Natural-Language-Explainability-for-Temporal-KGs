"""Run all Milestone 4 examples in sequence.

Examples 1 and 2 (M1, M2) are lightweight and run without any model servers.
Examples 3 and 4 (M3, capstone) load the graph pipeline and embedding/LLM
backends; they degrade gracefully if those services are unreachable but will be
slow on first run while models download.

Usage::

    python examples/milestone4/run_all.py
    python examples/milestone4/run_all.py --skip-graph   # M1 + M2 only
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg_metta import hyperon_available

EXAMPLES = [
    ("m4e1_nlg", "M1 — Natural-language generation from MeTTa", False),
    ("m4e2_tms", "M2 — Truth maintenance from MeTTa", False),
    ("m4e3_graph", "M3 — Temporal graph QA from MeTTa", True),
    ("m4e4_capstone", "Capstone — Explainable inference path (M1+M2+M3)", True),
    ("m4e5_justification_path", "M2 — Multi-hop justification paths", False),
    ("m4e6_counterfactual", "M1+M2+M3 — Counterfactual reasoning", False),
    ("m4e7_styles", "M1+M3 — Style/domain narrative adaptation", False),
    ("m4e8_metta_reasoning", "MeTTa-side reasoning — match-derived TMS trace", False),
    ("m4e9_mork", "MORK atomspace — temporal edge store/query/transform", False),
]


def _load_example(name: str):
    """Load an example runner module by file path (no package __init__ needed)."""
    here = Path(__file__).resolve().parent
    path = here / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"m4_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Milestone 4 examples.")
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip the M3/capstone examples that load the graph pipeline.",
    )
    args = parser.parse_args()

    if not hyperon_available():
        print(
            "[skip] The 'hyperon' package is not installed, so the MeTTa examples\n"
            "       cannot run. Install it with:  pip install 'temporal-nlg[metta]'\n"
            "       The bridge itself is fully functional without it (see tests/metta)."
        )
        return 0

    failed: list[str] = []
    for name, label, needs_graph in EXAMPLES:
        if args.skip_graph and needs_graph:
            print(f"\n[skip] {name} (--skip-graph)")
            continue
        print(f"\n{'#' * 72}\n# {label}\n{'#' * 72}")
        start = time.perf_counter()
        mod = _load_example(name)
        try:
            rc = mod.main()
        except Exception as exc:  # pragma: no cover - example runner
            print(f"[error] {name} raised: {exc!r}")
            failed.append(name)
            continue
        elapsed = time.perf_counter() - start
        print(f"[done]  {name} ({elapsed:.1f}s, rc={rc})")
        if rc != 0:
            failed.append(name)

    print("\n" + "=" * 72)
    if failed:
        print(f"Completed with failures: {failed}")
        return 1
    print("All Milestone 4 examples completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
