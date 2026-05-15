"""
Smoke tests to ensure milestone 1 examples execute without external services.
"""

from examples.milestone1.m1e3_hybrid_generation_example import demo_hybrid_strategies
from examples.milestone1.m1e4_tms_justification_example import build_justification
from examples.milestone1.m1e4_counterfactual_example import run_counterfactuals


def test_hybrid_example_runs():
    results = demo_hybrid_strategies()
    assert set(results.keys()) == {"template", "polish", "llm"}
    assert all(r.text for r in results.values())


def test_tms_justification_example_runs():
    output = build_justification()
    assert "justification" in output
    assert "supported" in output["justification"] or "Belief not found" in output["justification"]


def test_counterfactual_example_runs():
    result = run_counterfactuals()
    cf = result["counterfactual_text"]
    assert "factual" in cf and "counterfactual" in cf
    assert result["shifted"].description
    assert result["swapped"]
