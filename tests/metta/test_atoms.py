"""Unit tests for the grounded-operation registry (no hyperon required).

These tests validate the operation specs and the no-hyperon fallback path of
:mod:`temporal_nlg_metta.atoms`. The hyperon-backed registration path is
exercised in ``test_integration.py`` (auto-skipped when hyperon is absent).
"""

from __future__ import annotations

import json

import pytest

from temporal_nlg_metta import TemporalBridge
from temporal_nlg_metta.atoms import (
    all_specs,
    available_tokens,
    register_atoms,
)
from temporal_nlg_metta.config import MettaConfig


def test_all_milestones_represented_in_tokens():
    tokens = available_tokens()
    # M1
    assert any(t.startswith("nlg-") for t in tokens)
    # M2
    assert any(t.startswith("tms-") for t in tokens)
    # M3
    assert any(t.startswith("graph-") for t in tokens)


def test_specs_have_token_and_factory():
    for spec in all_specs():
        assert "token" in spec and isinstance(spec["token"], str)
        assert callable(spec["factory"])
        assert spec["token"]  # non-empty


def test_register_atoms_returns_mapping():
    # register_atoms returns either raw callables (no hyperon) or OperationAtom
    # wrappers (hyperon present). In both cases it's a dict keyed by token.
    registry = register_atoms()
    assert isinstance(registry, dict)
    assert len(registry) == len(all_specs())
    assert "nlg-fact" in registry


def test_spec_factory_callables_invoke_bridge(tmp_path):
    # The spec factory produces the raw Python callable that underlies each
    # operation, independent of whether hyperon is installed. This is the layer
    # the bridge and the MeTTa atoms both ultimately call.
    bridge = TemporalBridge(config=MettaConfig(graph_dir=tmp_path))
    spec = next(s for s in all_specs() if s["token"] == "nlg-fact")
    fn = spec["factory"](bridge)
    result = fn(
        "point_in_time",
        json.dumps({"entity": "X", "event": "born", "date": "2000"}),
        "template",
    )
    parsed = json.loads(result)
    assert parsed["strategy"] == "template"
    assert "2000" in parsed["text"]


def test_fact_type_alias_works_through_factory(tmp_path):
    bridge = TemporalBridge(config=MettaConfig(graph_dir=tmp_path))
    spec = next(s for s in all_specs() if s["token"] == "nlg-fact")
    fn = spec["factory"](bridge)
    # "point" alias for point_in_time
    out = json.loads(
        fn(
            "point",
            json.dumps({"entity": "X", "event": "born", "date": "2001"}),
            "template",
        )
    )
    assert out["strategy"] == "template"


def test_m2_tms_factory_callables_end_to_end(tmp_path):
    bridge = TemporalBridge(config=MettaConfig(graph_dir=tmp_path))
    specs = {s["token"]: s["factory"](bridge) for s in all_specs()}
    specs["tms-start-trace"]()
    specs["tms-record-rule"](
        "r1",
        "n",
        json.dumps([{"fact_id": "a"}]),
        json.dumps({"fact_id": "f1", "value": "v"}),
    )
    fired = json.loads(specs["tms-rules-fired"]())
    assert fired == ["r1"]
