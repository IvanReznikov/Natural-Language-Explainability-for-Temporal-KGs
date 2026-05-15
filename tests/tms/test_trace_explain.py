from temporal_nlg.tms.trace import QueryTrace, RuleTrace
from temporal_nlg.tms.trace_explain import TraceJustifier


def make_trace():
    r1 = RuleTrace(
        rule_id="r1",
        rule_name="rule_one",
        inputs=[{"fact_id": "a", "value": 1}],
        conclusion={"fact_id": "b", "value": 2},
        fired_at=0.1,
    )
    r2 = RuleTrace(
        rule_id="r2",
        rule_name="rule_two",
        inputs=[{"fact_id": "b", "value": 2}],
        conclusion={"fact_id": "c", "value": 3},
        fired_at=0.2,
    )
    qt = QueryTrace(query_id="q1", started_at=0.0, rule_traces=[r1, r2])
    qt.completed_at = 0.3
    return qt


def test_explain_fact_builds_path_text():
    trace = make_trace()
    justifier = TraceJustifier(trace)

    paths = justifier.paths_for_fact("c")
    assert len(paths) == 1
    text = paths[0].as_text()
    assert "rule_two" in text
    assert "b=2" in text

    explanation = justifier.explain_fact("c")
    assert "rule_two" in explanation
    assert "rule_one" in explanation


def test_explain_unknown_fact():
    trace = make_trace()
    justifier = TraceJustifier(trace)
    msg = justifier.explain_fact("nonexistent")
    assert "No justification" in msg
