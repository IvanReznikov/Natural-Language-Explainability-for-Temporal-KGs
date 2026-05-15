"""
Accuracy, evaluation, and reporting coverage for milestone 1.
"""

import pytest

from temporal_nlg.evaluation import AccuracyEvaluator, AccuracyMetrics
from temporal_nlg.evaluation.m1_e1_evaluation import (
    M1E1EvaluatorV3,
    calculate_flesch_score,
    calculate_information_density,
)


def test_point_in_time_high_accuracy():
    evaluator = AccuracyEvaluator()

    class Fact:
        entity = "Neil Armstrong"
        fact_type = "point_in_time"

    fact = Fact()
    text = "Neil Armstrong landed on the Moon in 1969."
    metrics = evaluator.evaluate(fact, text)

    assert metrics.overall_accuracy > 0
    assert metrics.hallucination_detected is True


def test_interval_incomplete_dates_reduces_score():
    evaluator = AccuracyEvaluator()

    class Fact:
        entity = "Voyager mission"
        fact_type = "interval"
        start_date = "1977"
        end_date = "1989"

    fact = Fact()
    text = "The Voyager mission started in 1977."
    metrics = evaluator.evaluate(fact, text)

    assert metrics.overall_accuracy < 1.0
    assert metrics.hallucination_detected is True


def test_hallucination_flag_extra_year():
    evaluator = AccuracyEvaluator()

    class Fact:
        entity = "Event X"
        fact_type = "point_in_time"
        date = "2000"

    fact = Fact()
    text = "Event X happened in 2000 and again in 2020."
    metrics = evaluator.evaluate(fact, text)

    assert metrics.hallucination_detected is True


def test_accuracy_scoring_and_hallucination_penalty():
    fact = {"entity": "Einstein", "event": "born", "date": "1879"}
    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(fact, "Einstein was born in 1879 before 1900")
    assert isinstance(metrics, AccuracyMetrics)
    assert metrics.overall_accuracy <= 1.0
    assert metrics.hallucination_detected is True

    batch = evaluator.batch_evaluate([fact], ["Einstein born 1879"])
    agg = evaluator.aggregate_metrics(batch)
    assert agg["mean_overall_accuracy"] <= 1.0


def test_accuracy_batch_length_mismatch_raises():
    evaluator = AccuracyEvaluator()
    with pytest.raises(ValueError):
        evaluator.batch_evaluate([{}], [])


def test_flesch_and_information_density_bounds():
    assert 0 <= calculate_flesch_score("Short sentence.") <= 100
    assert calculate_information_density("") == 0.0
    assert 0 < calculate_information_density("Complexity abounds") <= 1.0


def test_m1_e1_evaluator_small_run_and_text_report():
    evaluator = M1E1EvaluatorV3()
    report = evaluator.evaluate_all_templates(n_examples_per_type=1)
    assert report.total_renders > 0
    text_report = evaluator.generate_text_report()
    assert "SUMMARY METRICS" in text_report
    assert report.type_metrics
    assert report.overall_information_density >= 0.0

    as_dict = report.to_dict()
    assert "render_details" in as_dict


def test_initialization():
    evaluator = AccuracyEvaluator()
    assert evaluator is not None
    assert len(evaluator.temporal_words) > 0


def test_date_preservation_perfect():
    evaluator = AccuracyEvaluator()
    fact = {"date": "1879", "event": "birth"}
    text = "Einstein was born in 1879."
    score = evaluator._score_date_preservation(fact, text)
    assert score >= 0.9


def test_date_preservation_missing():
    evaluator = AccuracyEvaluator()
    fact = {"date": "1879", "event": "birth"}
    text = "Einstein was born in the 19th century."
    score = evaluator._score_date_preservation(fact, text)
    assert score < 1.0


def test_date_preservation_no_dates():
    evaluator = AccuracyEvaluator()
    fact = {"event": "meeting"}
    text = "They met at the conference."
    score = evaluator._score_date_preservation(fact, text)
    assert score == 1.0


def test_entity_preservation_exact():
    evaluator = AccuracyEvaluator()

    class MockFact:
        entity = "Albert Einstein"

    fact = MockFact()
    text = "Albert Einstein was a physicist."
    score = evaluator._score_entity_preservation(fact, text)
    assert score == 1.0


def test_entity_preservation_partial():
    evaluator = AccuracyEvaluator()

    class MockFact:
        entity = "Albert Einstein"

    fact = MockFact()
    text = "Einstein was a physicist."
    score = evaluator._score_entity_preservation(fact, text)
    assert score >= 0.7


def test_entity_preservation_missing():
    evaluator = AccuracyEvaluator()

    class MockFact:
        entity = "Albert Einstein"

    fact = MockFact()
    text = "A physicist made a discovery."
    score = evaluator._score_entity_preservation(fact, text)
    assert score == 0.0


def test_relation_preservation_causal():
    evaluator = AccuracyEvaluator()

    class MockFact:
        fact_type = "causality"

    fact = MockFact()
    text = "The discovery led to new theories."
    score = evaluator._score_relation_preservation(fact, text)
    assert score >= 0.9


def test_relation_preservation_sequence():
    evaluator = AccuracyEvaluator()

    class MockFact:
        fact_type = "sequence"

    fact = MockFact()
    text = "First he studied, then he published."
    score = evaluator._score_relation_preservation(fact, text)
    assert score >= 0.9


def test_hallucination_detection_extra_years():
    evaluator = AccuracyEvaluator()
    fact = {"date": "1879"}
    text = "Born in 1879, died in 1955."
    hallucination = evaluator._detect_hallucination(fact, text)
    assert hallucination is True


def test_hallucination_detection_long_text():
    evaluator = AccuracyEvaluator()
    fact = {"event": "birth"}
    text = " ".join(["word"] * 60)
    hallucination = evaluator._detect_hallucination(fact, text)
    assert hallucination is True


def test_hallucination_detection_clean():
    evaluator = AccuracyEvaluator()
    fact = {"date": "1879", "entity": "Einstein"}
    text = "Einstein was born in 1879."
    hallucination = evaluator._detect_hallucination(fact, text)
    assert hallucination is False


def test_evaluate_complete():
    evaluator = AccuracyEvaluator()

    class MockFact:
        entity = "Einstein"
        fact_type = "point_in_time"

        def __str__(self):
            return "Einstein birth 1879"

    fact = MockFact()
    text = "Einstein was born in 1879."
    metrics = evaluator.evaluate(fact, text)

    assert isinstance(metrics, AccuracyMetrics)
    assert 0 <= metrics.date_preservation <= 1
    assert 0 <= metrics.entity_preservation <= 1
    assert 0 <= metrics.relation_preservation <= 1
    assert 0 <= metrics.overall_accuracy <= 1
    assert isinstance(metrics.hallucination_detected, bool)


def test_batch_evaluate():
    evaluator = AccuracyEvaluator()

    class MockFact:
        entity = "Test"
        fact_type = "point_in_time"

        def __str__(self):
            return "Test 2000"

    facts = [MockFact() for _ in range(3)]
    texts = ["Test in 2000." for _ in range(3)]

    results = evaluator.batch_evaluate(facts, texts)

    assert len(results) == 3
    assert all(isinstance(m, AccuracyMetrics) for m in results)


def test_batch_evaluate_length_mismatch():
    evaluator = AccuracyEvaluator()
    facts = [{"event": "test"}]
    texts = ["Text 1", "Text 2"]
    with pytest.raises(ValueError, match="same length"):
        evaluator.batch_evaluate(facts, texts)


def test_aggregate_metrics():
    evaluator = AccuracyEvaluator()
    metrics = [
        AccuracyMetrics(0.9, 0.8, 0.7, False, 0.85),
        AccuracyMetrics(0.8, 0.9, 0.6, True, 0.75),
        AccuracyMetrics(1.0, 0.7, 0.8, False, 0.90),
    ]
    aggregated = evaluator.aggregate_metrics(metrics)

    assert "mean_date_preservation" in aggregated
    assert "mean_entity_preservation" in aggregated
    assert "mean_relation_preservation" in aggregated
    assert "hallucination_rate" in aggregated
    assert "mean_overall_accuracy" in aggregated
    assert 0 <= aggregated["mean_overall_accuracy"] <= 1
    assert aggregated["hallucination_rate"] == pytest.approx(1 / 3)


def test_aggregate_metrics_empty():
    evaluator = AccuracyEvaluator()
    aggregated = evaluator.aggregate_metrics([])
    assert aggregated == {}
