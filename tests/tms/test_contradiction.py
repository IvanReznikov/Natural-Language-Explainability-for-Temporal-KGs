from temporal_nlg.tms.trace import QueryTrace, RuleTrace
from temporal_nlg.tms.contradiction import ContradictionDetector


def test_detects_conflicting_values():
    rt1 = RuleTrace(
        rule_id="r1",
        rule_name="rule_one",
        inputs=[],
        conclusion={"fact_id": "f", "value": 1},
        fired_at=0.1,
    )
    rt2 = RuleTrace(
        rule_id="r2",
        rule_name="rule_two",
        inputs=[],
        conclusion={"fact_id": "f", "value": 2},
        fired_at=0.2,
    )
    qt = QueryTrace(query_id="q1", started_at=0.0, rule_traces=[rt1, rt2])

    detector = ContradictionDetector()
    results = detector.detect(qt)
    assert len(results) == 1
    assert results[0].fact_id == "f"
    assert 1 in results[0].values and 2 in results[0].values
    assert set(results[0].rule_ids) == {"r1", "r2"}


def test_no_conflict_single_value():
    rt = RuleTrace(
        rule_id="r1",
        rule_name="rule_one",
        inputs=[],
        conclusion={"fact_id": "f", "value": 1},
        fired_at=0.1,
    )
    qt = QueryTrace(query_id="q1", started_at=0.0, rule_traces=[rt])

    detector = ContradictionDetector()
    assert detector.detect(qt) == []
