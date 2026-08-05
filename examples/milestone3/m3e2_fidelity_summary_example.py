#!/usr/bin/env python3
"""Milestone 3 E2 fidelity summary example using temporal_nlg evaluators."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg.evaluation import M3E2FidelityEvaluator, aggregate_by_bucket


def _load_jsonl(path: Path, limit: int) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            rows.append(json.loads(payload))
            if len(rows) >= limit:
                break
    return rows


def main() -> None:
    dataset = ROOT / "data" / "jsonls" / "temporal_graph.jsonl"
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    evaluator = M3E2FidelityEvaluator()
    sample_rows = _load_jsonl(dataset, limit=60)

    scored: List[Dict] = []
    for row in sample_rows:
        prediction = row.get("gold_answer") or row.get("answer") or row.get("query") or ""
        scored.append(evaluator.evaluate_example(row, str(prediction)))

    summary = aggregate_by_bucket(scored)

    print("M3-E2 Fidelity Summary (sample)")
    print("rows_scored:", len(scored))
    for bucket in sorted(summary.keys()):
        bucket_summary = summary[bucket]
        count = int(bucket_summary.get("count", 0))
        print(f"\n[{bucket}] count={count}")
        for key in (
            "timestamp_accuracy",
            "boundary_accuracy",
            "ordering_accuracy",
            "temporal_constraint_correctness",
            "entity_coverage",
            "context_relevance",
        ):
            if key in bucket_summary:
                print(f"  {key}: {bucket_summary[key]:.3f}")


if __name__ == "__main__":
    main()
