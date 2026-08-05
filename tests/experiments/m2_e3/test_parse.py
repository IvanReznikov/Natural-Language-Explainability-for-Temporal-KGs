import types

import pytest

from experiments.m2_e3_parse import run_parse


@pytest.fixture()
def causal_row():
    return {"id": "q1", "text": "Did the storm cause flooding?"}


def test_parse_row_rules_handles_causal(causal_row):
    result = run_parse.parse_row_rules(causal_row)
    assert "causal" in result["intent_labels"]
    assert result["frame"].get("cause")
    assert result["frame"].get("effect")


def test_parse_row_fallback_when_validation_errors(monkeypatch, causal_row):
    dummy_bundle = object()

    def fake_predict_parser(_bundle, _text):
        return {
            "spans": [],
            "frame": {"cause": "", "effect": ""},
            "intent_labels": ["causal"],
            "raw": "{}",
        }

    def fake_validate(pred):
        if pred.get("source") == "model-parser":
            return {"errors": ["bad_model_output"], "warnings": []}
        return {"errors": [], "warnings": []}

    monkeypatch.setattr(run_parse, "predict_parser", fake_predict_parser)
    monkeypatch.setattr(run_parse, "validate_prediction", fake_validate)

    result, justification = run_parse.parse_row(
        causal_row,
        {"parser": dummy_bundle, "intent": None},
        threshold=0.5,
        fallback_on_error=True,
    )

    assert result["source"] == "rule-parser"
    assert "fallback_rules_on_error" in justification["notes"]
    assert justification["validation"].get("errors") == ["bad_model_output"]
    assert justification.get("validation_rules", {}).get("errors") == []
