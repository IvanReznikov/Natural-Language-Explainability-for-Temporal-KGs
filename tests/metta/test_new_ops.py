"""Tests for the new M4 bridge methods and grounded ops (no hyperon required).

Covers: justification_paths, counterfactual (fact-level), counterfactual_shift_time
(belief-level), and explain_path_styled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporal_nlg_metta import TemporalBridge
from temporal_nlg_metta.atoms import all_specs
from temporal_nlg_metta.config import MettaConfig


@pytest.fixture
def bridge() -> TemporalBridge:
    return TemporalBridge(config=MettaConfig(graph_dir=Path("/nonexistent")))


# ── Multi-hop justification paths ────────────────────────────────────────────


def test_justification_paths_reconstructs_chain(bridge: TemporalBridge):
    bridge.start_trace("q-jp")
    bridge.record_rule(
        "rule:extract-year",
        "extract",
        json.dumps([{"fact_id": "raw_record", "value": "x"}]),
        json.dumps({"fact_id": "year_extracted", "value": "1879"}),
    )
    bridge.record_rule(
        "rule:normalize-year",
        "normalize",
        json.dumps([{"fact_id": "year_extracted", "value": "1879"}]),
        json.dumps({"fact_id": "birth_year", "value": "1879"}),
    )
    paths = bridge.justification_paths("birth_year")
    assert len(paths) >= 1
    # The reconstructed chain must mention the upstream rule(s).
    chain_text = paths[0]["as_text"]
    assert "extract" in chain_text or "normalize" in chain_text


def test_justification_paths_empty_for_unknown_fact(bridge: TemporalBridge):
    bridge.start_trace("q-jp-empty")
    paths = bridge.justification_paths("nonexistent_fact")
    assert paths == []


# ── Fact-level counterfactual ────────────────────────────────────────────────


def test_counterfactual_returns_factual_and_alternative(bridge: TemporalBridge):
    result = bridge.counterfactual(
        subject="assembly line",
        predicate="caused",
        obj="price drop",
        alt_subject="hand production",
        alt_predicate="caused",
        alt_obj="price rise",
        timeframe="1913",
        alt_timeframe="1913",
    )
    assert "assembly line" in result["factual"]
    assert "hand production" in result["counterfactual"]
    assert "price drop" in result["factual"]
    assert "subject changed" in result["delta"]


def test_counterfactual_no_change_delta(bridge: TemporalBridge):
    result = bridge.counterfactual(
        subject="X",
        predicate="caused",
        obj="Y",
        alt_subject="X",
        alt_predicate="caused",
        alt_obj="Y",
    )
    assert result["delta"] == "no change"


# ── Belief-level temporal counterfactual ─────────────────────────────────────


def test_counterfactual_shift_time_registers_new_belief(bridge: TemporalBridge):
    bridge.add_belief(
        "B1",
        json.dumps({"claim": "Adopted in 1913"}),
        "[]",
        json.dumps([{"source": "graph", "snippet": "s", "weight": 1.0}]),
    )
    result = bridge.counterfactual_shift_time("B1", "5 years earlier")
    assert result["new_belief_id"] == "cf_B1"
    assert result["original_id"] == "B1"
    assert "5 years earlier" in result["description"]
    # The counterfactual belief cf_B1 was registered and claims B1 as its
    # support (cf_B1 depends on B1), so it appears in active beliefs and its
    # own support chain reaches back to B1.
    assert "cf_B1" in bridge.active_beliefs()
    cf_chain = bridge.support_chain("cf_B1")
    cf_chain_ids = [b["belief_id"] for b in cf_chain]
    assert cf_chain_ids[0] == "cf_B1"
    assert "B1" in cf_chain_ids


def test_counterfactual_shift_time_missing_belief(bridge: TemporalBridge):
    result = bridge.counterfactual_shift_time("absent", "1 year")
    assert "error" in result


# ── Styled path explanation ──────────────────────────────────────────────────


def test_explain_path_styled_changes_with_register(bridge: TemporalBridge):
    adj = json.dumps(
        {
            "ModelT": [("AssemblyLine", "produced_with", "1913")],
            "AssemblyLine": [("PriceDrop", "enabled", "1913")],
        }
    )
    novice = bridge.explain_path_styled(adj, "ModelT", "PriceDrop", style="novice")
    expert = bridge.explain_path_styled(adj, "ModelT", "PriceDrop", style="expert")
    # Both render the path, but the intros differ by style.
    assert "plain terms" in novice["narrative"]
    assert "Tracing the dependency path" in expert["narrative"]


def test_explain_path_styled_domain_finance(bridge: TemporalBridge):
    adj = json.dumps(
        {
            "A": [("B", "led_to", "2020")],
            "B": [("C", "enabled", "2020")],
        }
    )
    fin = bridge.explain_path_styled(adj, "A", "C", style="neutral", domain="finance")
    assert "Event flow" in fin["narrative"]


# ── New ops present in the spec registry ─────────────────────────────────────


def test_new_tokens_registered():
    tokens = {s["token"] for s in all_specs()}
    for expected in (
        "tms-justification-path",
        "tms-counterfactual-shift",
        "graph-explain-path-styled",
        "counterfactual",
        "tms-record-rule-facts",
        "graph-evidence-atoms",
        "edges-from-json",
    ):
        assert expected in tokens


# ── Atom-friendly rule recording (MeTTa-derived facts) ───────────────────────


def test_record_rule_facts_builds_trace_entries(bridge: TemporalBridge):
    bridge.start_trace("q-facts")
    bridge.record_rule_facts(
        "rule:causal-chain-in-metta",
        "Derived by MeTTa pattern match",
        "moving assembly line",
        "1913",
        "Model T price drop",
        "causal",
        0.9,
    )
    assert bridge.rules_fired() == ["rule:causal-chain-in-metta"]
    text = bridge.explain_belief_trace("Model T price drop")
    assert "moving assembly line" in text
    assert "Model T price drop" in text


# ── Evidence as matchable edge tuples/atoms ──────────────────────────────────


def test_evidence_edge_tuples_reads_answer_evidence(bridge: TemporalBridge):
    # No graph pipeline needed: seed the last-answer cache directly.
    bridge._last_answer = {
        "evidence": [
            {"source": "A", "relation": "caused", "target": "B", "start": "1913"},
            {"src": "C", "rel": "made", "tgt": "D", "year": "1914"},
        ]
    }
    assert bridge.evidence_edge_tuples() == [
        ["A", "caused", "B", "1913"],
        ["C", "made", "D", "1914"],
    ]


def test_evidence_edge_tuples_empty_without_answer(bridge: TemporalBridge):
    assert bridge.evidence_edge_tuples() == []


def test_edge_tuples_from_json_normalizes_shapes():
    from temporal_nlg_metta.atoms import _edge_tuples_from_json

    positional = _edge_tuples_from_json('[["A", "caused", "B", "1913"], ["C", "made", "D"]]')
    assert positional == [["A", "caused", "B", "1913"], ["C", "made", "D", ""]]
    objects = _edge_tuples_from_json(
        '[{"source": "A", "relation": "caused", "target": "B", "start": "1913"}]'
    )
    assert objects == [["A", "caused", "B", "1913"]]


def test_edge_tuples_from_json_rejects_non_list():
    from temporal_nlg_metta.atoms import _edge_tuples_from_json

    with pytest.raises(ValueError):
        _edge_tuples_from_json('{"a": 1}')


def test_atom_returning_ops_disable_unwrap():
    # Atom-returning ops must be registered with unwrap=False, otherwise
    # hyperon wraps the returned list as a single opaque grounded atom and the
    # edge expressions never reach the MeTTa evaluator.
    specs = {s["token"]: s for s in all_specs()}
    assert specs["graph-evidence-atoms"].get("unwrap") is False
    assert specs["edges-from-json"].get("unwrap") is False
