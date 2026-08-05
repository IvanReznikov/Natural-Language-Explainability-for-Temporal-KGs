"""Unit tests for the temporal_nlg_metta bridge (no hyperon required).

These tests exercise the M1/M2/M3 wiring of :class:`TemporalBridge` directly.
They construct the bridge with a minimal config and avoid the graph pipeline
except where the graph artifacts are available, so they run hermetically and fast.
"""

from __future__ import annotations

import json

import pytest

from temporal_nlg_metta import TemporalBridge
from temporal_nlg_metta.config import MettaConfig


@pytest.fixture
def bridge() -> TemporalBridge:
    return TemporalBridge(config=MettaConfig(graph_dir=__import__("pathlib").Path("/nonexistent")))


# ── M1 surface ──────────────────────────────────────────────────────────────


def test_nlg_fact_template_strategy(bridge: TemporalBridge):
    content = json.dumps({"entity": "Einstein", "event": "was born", "date": "1879"})
    result = bridge.nlg_fact("point_in_time", content, strategy="template")
    assert result["strategy"] == "template"
    assert "1879" in result["text"]
    assert result["confidence"] == pytest.approx(0.9)


def test_nlg_fact_accepts_fact_type_aliases(bridge: TemporalBridge):
    content = json.dumps({"entity": "X", "event": "started", "date": "2000"})
    # "point" is an alias for point_in_time
    result = bridge.nlg_fact("point", content, strategy="template")
    assert result["strategy"] == "template"


def test_nlg_fact_rejects_unknown_type(bridge: TemporalBridge):
    with pytest.raises(ValueError, match="Unknown fact type"):
        bridge.nlg_fact("nope", "{}")


def test_nlg_readability_returns_flesch(bridge: TemporalBridge):
    metrics = bridge.nlg_readability("Einstein was born in 1879.")
    assert "flesch_score" in metrics
    # calculate_flesch_score may return an int (e.g. 100) for simple text.
    assert isinstance(metrics["flesch_score"], (int, float))


def test_nlg_evaluate_scores_text(bridge: TemporalBridge):
    content = json.dumps({"entity": "Einstein", "event": "was born", "date": "1879"})
    metrics = bridge.nlg_evaluate(content, "Einstein was born in 1879.")
    assert 0.0 <= metrics["overall_accuracy"] <= 1.0


# ── M2 surface ───────────────────────────────────────────────────────────────


def test_trace_records_and_lists_rules(bridge: TemporalBridge):
    qid = bridge.start_trace("q1")
    assert qid == "q1"
    bridge.record_rule(
        "r1",
        "rule_one",
        json.dumps([{"fact_id": "f1"}]),
        json.dumps({"fact_id": "f2", "value": 43}),
    )
    assert bridge.rules_fired() == ["r1"]


def test_record_rule_auto_starts_trace(bridge: TemporalBridge):
    # No explicit start_trace call.
    rid = bridge.record_rule("auto", "auto_rule", "[]", "{}")
    assert rid == "auto"
    assert bridge.rules_fired() == ["auto"]


def test_contradictions_detected(bridge: TemporalBridge):
    bridge.start_trace("q-contr")
    bridge.record_rule(
        "r1", "n", json.dumps([{"fact_id": "a"}]), json.dumps({"fact_id": "f1", "value": "1879"})
    )
    bridge.record_rule(
        "r2", "n", json.dumps([{"fact_id": "b"}]), json.dumps({"fact_id": "f1", "value": "1880"})
    )
    contras = bridge.contradictions()
    assert len(contras) == 1
    assert contras[0]["fact_id"] == "f1"
    assert set(contras[0]["values"]) == {"1879", "1880"}


def test_why_not_fired_reports_absent_rules(bridge: TemporalBridge):
    bridge.start_trace("q-why")
    bridge.record_rule("fired_one", "n", "[]", "{}")
    result = bridge.why_not(json.dumps(["fired_one", "missing_one"]))
    assert result["fired_one"] == "fired"
    assert "absent" in result["missing_one"]


def test_belief_support_chain_and_justification(bridge: TemporalBridge):
    bridge.add_belief(
        "B1",
        json.dumps({"claim": "base"}),
        "[]",
        json.dumps([{"source": "graph", "snippet": "s", "weight": 1.0}]),
    )
    bridge.add_belief("B2", json.dumps({"claim": "derived"}), json.dumps(["B1"]))
    chain = bridge.support_chain("B2")
    assert [b["belief_id"] for b in chain] == ["B2", "B1"]
    explanation = bridge.explain_belief("B2")
    assert "B1" in explanation


def test_retract_propagates_dirty(bridge: TemporalBridge):
    bridge.add_belief("B1", "{}", "[]")
    bridge.add_belief("B2", "{}", json.dumps(["B1"]))
    result = bridge.retract("B1")
    assert "B1" in result["affected"]
    assert "B2" in result["affected"]
    assert "B2" in bridge.dirty_beliefs()


def test_reset_clears_beliefs(bridge: TemporalBridge):
    bridge.add_belief("X", "{}", "[]")
    summary = bridge.reset()
    assert summary["cleared_beliefs"] == 1
    assert bridge.active_beliefs() == []


def test_trace_to_dict_empty_when_no_session(bridge: TemporalBridge):
    assert bridge.trace_to_dict() == {}


# ── M3 surface (path explanation only — no graph pipeline needed) ────────────


def test_explain_path_records_belief_and_returns_narrative(bridge: TemporalBridge):
    adj = json.dumps(
        {
            "ModelT": [("AssemblyLine", "produced_with", "1913")],
            "AssemblyLine": [("PriceDrop", "enabled", "1913")],
        }
    )
    result = bridge.explain_path(adj, "ModelT", "PriceDrop", belief_id="belief_test")
    assert result["summary"]
    assert result["narrative"]
    # The narrative was recorded as a justified belief.
    assert (
        "belief_test" in bridge.active_beliefs()
        or bridge.belief_store.get_belief("belief_test") is not None
    )


def test_pipeline_lazy_and_raises_when_no_graph(bridge: TemporalBridge):
    # The bridge was constructed with a nonexistent graph dir.
    with pytest.raises(FileNotFoundError, match="Graph artifacts not found"):
        _ = bridge.pipeline
