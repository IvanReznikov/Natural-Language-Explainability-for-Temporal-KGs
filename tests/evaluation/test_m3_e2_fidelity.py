import pytest

from temporal_nlg.evaluation.m3_e2_fidelity import M3E2FidelityEvaluator


def test_point_timestamp_accuracy_tiers():
    evaluator = M3E2FidelityEvaluator()
    record = {
        "id": "x",
        "domain": "test",
        "time_scope": "point",
        "gold_facts": [
            {"subject": "Event", "relation": "dated", "object": "", "start": "2020-01-10", "end": "2020-01-10"}
        ],
    }

    m_exact = evaluator.evaluate_example(record, "Happened on 2020-01-10.")
    assert m_exact["timestamp_accuracy"] == 1.0

    m_week = evaluator.evaluate_example(record, "Happened on 2020-01-15.")
    assert m_week["timestamp_accuracy"] == 0.8

    m_far = evaluator.evaluate_example(record, "Happened on 2020-02-10.")
    assert m_far["timestamp_accuracy"] == 0.0


def test_interval_boundary_accuracy():
    evaluator = M3E2FidelityEvaluator()
    record = {
        "id": "y",
        "domain": "test",
        "time_scope": "interval",
        "gold_facts": [
            {"subject": "Project", "relation": "ran", "object": "", "start": "2020-01-01", "end": "2020-01-10"}
        ],
    }

    m_ok = evaluator.evaluate_example(record, "It ran from 2020-01-01 to 2020-01-10.")
    assert m_ok["boundary_accuracy"] == pytest.approx(1.0)

    m_off = evaluator.evaluate_example(record, "It ran from 2020-01-02 to 2020-01-17.")
    # start diff 1 day -> 1.0, end diff 7 days -> 0.8 => mean 0.9
    assert m_off["boundary_accuracy"] == pytest.approx(0.9)
