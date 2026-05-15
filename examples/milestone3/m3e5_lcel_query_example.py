"""Milestone 3: LCEL temporal graph query and built-in visualization demo."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg.graph_query import TemporalGraphLCELPipeline


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_demo() -> None:
    root = _repo_root()
    output_dir = root / "data" / "jsonls" / "temporal_graph_output"
    if not (output_dir / "nodes.jsonl").exists() or not (output_dir / "edges.jsonl").exists():
        output_dir = root / "data" / "jsonls" / "temporal_graph_output_v3"
    pipeline = TemporalGraphLCELPipeline(output_dir)

    questions = [
        "What likely caused the Model T price drop in 1913?",
        "Was Model T connected to Ford production scaling in 1913?",
        "When did the 2008 Financial Crisis first start affecting GM and Chrysler bankruptcy restructuring?",
        "How many suppliers did Company C have in 2024?",
    ]

    stamped = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = root / "output" / "examples" / "milestone3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"m3_lcel_graph_query_{stamped}.json"

    records = []
    for question in questions:
        result = pipeline.invoke(question)
        records.append(result)
        print("\nQ:", question)
        print("A:", result["answer_text"])
        print("Planned type:", result.get("plan", {}).get("query_type"))
        print("Confidence:", f"{result['confidence']:.2f}")
        print("Mermaid preview:")
        print(result["mermaid"].splitlines()[0])

    out_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nSaved detailed outputs to: {out_file}")


if __name__ == "__main__":
    run_demo()
