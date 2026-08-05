#!/usr/bin/env python3
"""Capture validated M4 example outputs as a structured artifact.

Runs each example's MeTTa program through the real hyperon interpreter and
writes the parsed results to output/examples/milestone4/example_outputs.json,
so the results in docs/RESULTS_M4.md are backed by a reproducible artifact.

The fast M1/M2 examples are always captured (no model servers required). The
M3/capstone examples (m4e3, m4e4) are captured too when the graph artifacts are
present; otherwise they are recorded as skipped so the artifact is complete and
honest about what was executed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg_metta import TemporalBridge, hyperon_available, run_metta_file
from temporal_nlg_metta.config import MettaConfig

OUT_DIR = ROOT / "output" / "examples" / "milestone4"

# Fast examples: M1/M2 surface only; run with a nonexistent graph dir so the M3
# pipeline stays lazily unbuilt.
FAST_EXAMPLES = (
    "m4e1_nlg",
    "m4e2_tms",
    "m4e5_justification_path",
    "m4e6_counterfactual",
    "m4e7_styles",
    "m4e8_metta_reasoning",
)

# Graph examples: need the M3 graph artifacts (and, for refined answers, the
# optional embedding/LLM backends). Captured with the default env-driven config.
GRAPH_EXAMPLES = (
    "m4e3_graph",
    "m4e4_capstone",
)


def _atom_to_str(atom) -> str:
    """Return the raw string value of a result atom."""
    s = str(atom).strip()
    # hyperon wraps string results in surrounding quotes; strip one layer.
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


def _try_json(value: str):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _capture(name: str, examples_dir: Path, bridge: TemporalBridge) -> dict:
    results = run_metta_file(examples_dir / f"{name}.metta", bridge=bridge)
    parsed = []
    for block in results:
        for atom in block:
            parsed.append(_try_json(_atom_to_str(atom)))
    return {
        "file": f"examples/milestone4/{name}.metta",
        "status": "executed",
        "expressions": len(parsed),
        "results": parsed,
    }


def main() -> int:
    if not hyperon_available():
        print("[skip] hyperon not installed")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples_dir = Path(__file__).resolve().parent

    captured: dict = {
        "hyperon_version": _hyperon_version(),
        "python_version": sys.version.split()[0],
        "examples": {},
    }

    fast_bridge = TemporalBridge(config=MettaConfig(graph_dir=Path("/nonexistent")))
    for name in FAST_EXAMPLES:
        captured["examples"][name] = _capture(name, examples_dir, fast_bridge)
        print(f"captured {name}: {captured['examples'][name]['expressions']} expressions")

    graph_dir = MettaConfig.from_env().graph_dir
    graph_ready = (graph_dir / "nodes.jsonl").exists()
    graph_bridge = None
    for name in GRAPH_EXAMPLES:
        if not graph_ready:
            captured["examples"][name] = {
                "file": f"examples/milestone4/{name}.metta",
                "status": f"skipped: graph artifacts not found at {graph_dir}",
            }
            print(f"[skip] {name}: graph artifacts not found at {graph_dir}")
            continue
        if graph_bridge is None:
            graph_bridge = TemporalBridge()
        captured["examples"][name] = _capture(name, examples_dir, graph_bridge)
        print(f"captured {name}: {captured['examples'][name]['expressions']} expressions")

    out_file = OUT_DIR / "example_outputs.json"
    out_file.write_text(json.dumps(captured, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_file}")
    return 0


def _hyperon_version() -> str:
    try:
        import hyperon

        return getattr(hyperon, "__version__", "unknown")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
