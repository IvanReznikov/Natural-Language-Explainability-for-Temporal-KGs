from temporal_nlg.tms.trace import QueryTrace, RuleTrace
from temporal_nlg.tms import meta_query


def make_trace():
    r1 = RuleTrace(
        rule_id="r1",
        rule_name="rule_one",
        inputs=[{"fact_id": "a"}],
        conclusion={"fact_id": "b", "value": 2},
        fired_at=0.1,
    )
    r2 = RuleTrace(
        rule_id="r2",
        rule_name="rule_two",
        inputs=[{"fact_id": "b"}],
        conclusion={"fact_id": "c", "value": 3},
        fired_at=0.2,
    )
    qt = QueryTrace(query_id="q1", started_at=0.0, rule_traces=[r1, r2])
    return qt


def test_rules_fired_and_why_not():
    qt = make_trace()
    fired = meta_query.rules_fired(qt)
    assert fired == ["r1", "r2"]
    why = meta_query.why_not_fired(qt, ["r2", "r3"])
    assert why["r2"] == "fired"
    assert why["r3"] == "absent in trace"


def test_influential_and_explain():
    qt = make_trace()
    infl = meta_query.influential_facts(qt, top_k=2)
    assert infl[0][0] == "a"
    explanation = meta_query.explain_fact(qt, "c")
    assert "rule_two" in explanation


def test_contradictions():
    qt = make_trace()
    # add conflicting rule
    qt.rule_traces.append(
        RuleTrace(
            rule_id="r3",
            rule_name="rule_three",
            inputs=[],
            conclusion={"fact_id": "c", "value": 99},
            fired_at=0.3,
        )
    )
    contras = meta_query.contradictions(qt)
    assert contras
    assert contras[0]["fact_id"] == "c"
