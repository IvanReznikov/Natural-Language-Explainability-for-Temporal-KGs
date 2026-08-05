#!/usr/bin/env python3
"""Milestone 3 E4 quality summary example using temporal_nlg aggregators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg.evaluation.m3_e4 import (
    CoherenceScore,
    ConsistencyResult,
    EfficiencyRun,
    GranularityVariant,
    aggregate_coherence,
    aggregate_consistency,
    aggregate_efficiency,
    aggregate_granularity,
)


def _read_summary(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    # Demonstrate package-level aggregators with minimal typed rows.
    eff_summary = aggregate_efficiency(
        [
            EfficiencyRun(
                scenario_id="s1",
                method="template",
                complexity_level=1,
                latency_ms=4.2,
                quality_proxy=0.84,
            ),
            EfficiencyRun(
                scenario_id="s2",
                method="template",
                complexity_level=2,
                latency_ms=5.0,
                quality_proxy=0.82,
            ),
            EfficiencyRun(
                scenario_id="s3",
                method="llm",
                complexity_level=4,
                latency_ms=920.0,
                quality_proxy=0.90,
            ),
        ]
    )
    cons_summary = aggregate_consistency(
        [
            ConsistencyResult(
                revision_id="r1",
                method="template",
                update_accuracy=0.88,
                contradiction_detected=True,
                coherence_rating_1_5=4.2,
                resolution_time_sec=2.1,
            ),
            ConsistencyResult(
                revision_id="r2",
                method="llm",
                update_accuracy=0.93,
                contradiction_detected=True,
                coherence_rating_1_5=4.5,
                resolution_time_sec=5.8,
            ),
        ]
    )
    coh_summary = aggregate_coherence(
        [
            CoherenceScore(
                scenario_id="c1",
                style="structured_narrative",
                semantic_consistency=0.84,
                narrative_consistency=0.82,
                logical_consistency=0.93,
            ),
            CoherenceScore(
                scenario_id="c2",
                style="timeline_plus_text",
                semantic_consistency=0.86,
                narrative_consistency=0.80,
                logical_consistency=0.95,
            ),
        ]
    )
    gran_summary = aggregate_granularity(
        [
            GranularityVariant(
                scenario_id="g1",
                granularity="days",
                text="Example A",
                quality_score=0.83,
                length_chars=140,
            ),
            GranularityVariant(
                scenario_id="g1",
                granularity="months",
                text="Example B",
                quality_score=0.81,
                length_chars=130,
            ),
            GranularityVariant(
                scenario_id="g1",
                granularity="years",
                text="Example C",
                quality_score=0.79,
                length_chars=118,
            ),
        ]
    )

    print("M3-E4 Quality Summary")
    print("typed_aggregator_demo:")
    print("  efficiency_methods:", list(eff_summary.get("by_method", {}).keys()))
    print("  consistency_methods:", list(cons_summary.get("by_method", {}).keys()))
    print("  coherence_styles:", list(coh_summary.get("by_style", {}).keys()))
    print("  granularity_levels:", list(gran_summary.get("by_granularity", {}).keys()))

    # Show repository summary files when present.
    summary_files = [
        ROOT / "output" / "m3_e4a_efficiency_analysis_methods" / "m3_e4a_efficiency.summary.json",
        ROOT / "output" / "m3_e4c_coherence_analysis_auto" / "m3_e4c_coherence.summary.json",
        ROOT / "output" / "m3_e4d_granularity_analysis_methods" / "m3_e4d_granularity.summary.json",
    ]
    for path in summary_files:
        data = _read_summary(path)
        if data is None:
            print(f"  missing: {path}")
        else:
            print(f"  found: {path} (top_keys={list(data.keys())[:5]})")


if __name__ == "__main__":
    main()
