from temporal_nlg.evaluation.m3_e4 import (
    EfficiencyRun,
    aggregate_efficiency,
    ConsistencyResult,
    aggregate_consistency,
    CoherenceScore,
    aggregate_coherence,
    GranularityVariant,
    aggregate_granularity,
)


def test_m3_e4a_efficiency_aggregation() -> None:
    runs = [
        EfficiencyRun(
            scenario_id="s1",
            method="template",
            complexity_level=1,
            latency_ms=100,
            tokens_out=50,
            cost=0.0,
            quality_proxy=0.8,
        ),
        EfficiencyRun(
            scenario_id="s2",
            method="template",
            complexity_level=1,
            latency_ms=200,
            tokens_out=60,
            cost=0.0,
            quality_proxy=0.9,
        ),
    ]
    summary = aggregate_efficiency(runs)
    assert "template" in summary["by_method"]
    assert summary["by_method"]["template"]["latency_ms"]["p95"] is not None


def test_m3_e4b_consistency_aggregation() -> None:
    results = [
        ConsistencyResult(
            revision_id="r1",
            method="template",
            update_accuracy=1.0,
            contradiction_detected=True,
            coherence_rating_1_5=4.5,
            resolution_time_sec=2.0,
        ),
        ConsistencyResult(
            revision_id="r2",
            method="template",
            update_accuracy=0.9,
            contradiction_detected=False,
            coherence_rating_1_5=4.0,
            resolution_time_sec=3.0,
        ),
    ]
    summary = aggregate_consistency(results)
    assert "template" in summary["by_method"]
    assert summary["by_method"]["template"]["update_accuracy_mean"] is not None


def test_m3_e4c_coherence_aggregation() -> None:
    scores = [
        CoherenceScore(
            scenario_id="s1",
            style="template",
            semantic_consistency=0.95,
            narrative_consistency=0.9,
            logical_consistency=0.98,
        )
    ]
    summary = aggregate_coherence(scores)
    assert summary["overall"]["semantic_consistency_mean"] == 0.95


def test_m3_e4d_granularity_aggregation() -> None:
    variants = [
        GranularityVariant(
            scenario_id="s1",
            granularity="days",
            text="x",
            quality_score=0.85,
            length_chars=1,
        )
    ]
    summary = aggregate_granularity(variants)
    assert summary["by_granularity"]["days"]["quality_mean"] == 0.85
