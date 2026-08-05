#!/usr/bin/env python3
"""Aggregate human evaluation CSVs for M1.

Usage:
  python experiments/m1_e5_integration/human_eval_aggregate.py docs/human_eval_template.csv
  # or a directory of CSVs

Outputs a JSON summary alongside the input with mean scores and pass rates.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

PASS_THRESHOLD = 4.25  # ≈85%


@dataclass
class EvalRow:
    clarity: float
    accuracy: float
    naturalness: float
    uncertain: bool


def load_rows(path: Path) -> List[EvalRow]:
    rows: List[EvalRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    EvalRow(
                        clarity=float(row.get("clarity_score_1to5", 0) or 0),
                        accuracy=float(row.get("accuracy_score_1to5", 0) or 0),
                        naturalness=float(row.get("naturalness_score_1to5", 0) or 0),
                        uncertain=str(row.get("uncertain_flag", "")).lower()
                        in {"1", "true", "yes", "y"},
                    )
                )
            except ValueError:
                # Skip bad rows
                continue
    return rows


def aggregate(rows: Sequence[EvalRow]) -> dict:
    if not rows:
        return {"count": 0, "clarity_mean": 0, "accuracy_mean": 0, "pass_rate": 0}
    clarity_mean = sum(r.clarity for r in rows) / len(rows)
    accuracy_mean = sum(r.accuracy for r in rows) / len(rows)
    pass_count = sum(
        1
        for r in rows
        if (r.clarity >= PASS_THRESHOLD and r.accuracy >= PASS_THRESHOLD and not r.uncertain)
    )
    return {
        "count": len(rows),
        "clarity_mean": clarity_mean,
        "accuracy_mean": accuracy_mean,
        "pass_rate": pass_count / len(rows),
    }


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python human_eval_aggregate.py <csv or dir>")
        return 1

    target = Path(argv[1])
    csv_paths: List[Path] = []
    if target.is_dir():
        csv_paths = list(target.glob("*.csv"))
    elif target.is_file():
        csv_paths = [target]
    else:
        print(f"Path not found: {target}")
        return 1

    all_rows: List[EvalRow] = []
    for p in csv_paths:
        all_rows.extend(load_rows(p))

    summary = aggregate(all_rows)
    summary_path = target.with_suffix("") if target.suffix else target
    out_path = Path(f"{summary_path}_summary.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Summary:", json.dumps(summary, indent=2))
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
