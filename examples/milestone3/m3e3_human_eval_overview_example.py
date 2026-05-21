#!/usr/bin/env python3
"""Milestone 3 E3 human-evaluation overview using temporal_nlg schemas."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg.evaluation.m3_e3 import (
    ComprehensionResponse,
    UtilityResponse,
    aggregate_comprehension,
    aggregate_utility,
)


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue


def _pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def main() -> None:
    comp_file = _pick_first_existing(
        [
            ROOT / "output" / "m3_e3a_comprehension" / "m3_e3a_results.jsonl",
            ROOT / "output" / "m3_e3a_comprehension" / "responses.jsonl",
        ]
    )
    util_file = _pick_first_existing(
        [
            ROOT / "output" / "m3_e3b_utility" / "m3_e3b_results.jsonl",
            ROOT / "output" / "m3_e3b_utility" / "responses.jsonl",
        ]
    )

    print("M3-E3 Human Evaluation Overview")

    if comp_file is not None:
        comp_rows: List[ComprehensionResponse] = []
        for row in _iter_jsonl(comp_file):
            try:
                comp_rows.append(ComprehensionResponse.model_validate(row))
            except Exception:
                continue
        summary = aggregate_comprehension(comp_rows)
        print("\ncomprehension_file:", comp_file)
        print("comprehension_n:", summary.get("n_responses"))
        print("comprehension_accuracy:", summary.get("overall_accuracy"))
        print("comprehension_mean_time_sec:", summary.get("mean_response_time_sec"))
    else:
        print("\ncomprehension_file: not found")

    if util_file is not None:
        util_rows: List[UtilityResponse] = []
        for row in _iter_jsonl(util_file):
            try:
                util_rows.append(UtilityResponse.model_validate(row))
            except Exception:
                continue
        summary = aggregate_utility(util_rows)
        print("\nutility_file:", util_file)
        print("utility_n:", summary.get("n_responses"))
        print("utility_success_with:", summary.get("success_rate_with_explanation"))
        print("utility_success_without:", summary.get("success_rate_without_explanation"))
    else:
        print("\nutility_file: not found")


if __name__ == "__main__":
    main()
