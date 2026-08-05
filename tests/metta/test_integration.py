"""Integration tests: full MeTTa programs run through the hyperon interpreter.

These require the optional ``hyperon`` dependency and are skipped automatically
when it is not installed. When hyperon is present, they verify that the grounded
operations are callable from real ``.metta`` programs end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporal_nlg_metta import TemporalBridge, hyperon_available, run_metta, run_metta_file
from temporal_nlg_metta.config import MettaConfig

pytestmark = pytest.mark.skipif(
    not hyperon_available(),
    reason="hyperon not installed; install with `pip install 'temporal-nlg[metta]'`",
)

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "milestone4"


@pytest.fixture
def bridge(tmp_path) -> TemporalBridge:
    # tmp_path graph dir keeps the lazy M3 pipeline from loading; M1/M2 paths work.
    return TemporalBridge(config=MettaConfig(graph_dir=tmp_path))


def _first_result(results):
    """Extract the first atom's string value from a MeTTa run() result."""
    return str(results[0][0])


def test_m1_nlg_fact_callable_from_metta(bridge: TemporalBridge):
    results = run_metta(
        '!(nlg-fact "point_in_time" '
        '"{\\"entity\\":\\"Einstein\\",\\"event\\":\\"was born\\",\\"date\\":\\"1879\\"}" '
        '"template")',
        bridge=bridge,
    )
    text = _first_result(results)
    assert "1879" in text


def test_m2_tms_round_trip_from_metta(bridge: TemporalBridge):
    program = (
        "!(tms-start-trace)\n"
        '!(tms-record-rule "r1" "n" "[{\\"fact_id\\":\\"a\\"}]" '
        '"{\\"fact_id\\":\\"f1\\",\\"value\\":\\"v\\"}")\n'
        "!(tms-rules-fired)\n"
    )
    results = run_metta(program, bridge=bridge)
    # Third top-level expression is tms-rules-fired.
    fired_text = str(results[2][0])
    assert "r1" in fired_text


def test_register_with_attaches_all_tokens(bridge: TemporalBridge):
    # If we got here, hyperon is available; build a runner and confirm tokens resolve.
    from temporal_nlg_metta import make_metta_runner
    from temporal_nlg_metta.atoms import register_with, available_tokens

    metta = make_metta_runner(bridge=bridge, register=False)
    registered = register_with(metta, bridge)
    assert set(registered) == set(available_tokens())


def test_m1_example_file_loads(bridge: TemporalBridge):
    # The M1 example file uses only grounded ops; it should run to completion.
    results = run_metta_file(EXAMPLES_DIR / "m4e1_nlg.metta", bridge=bridge)
    assert len(results) >= 4  # at least the four top-level expressions


def test_m2_example_file_loads(bridge: TemporalBridge):
    results = run_metta_file(EXAMPLES_DIR / "m4e2_tms.metta", bridge=bridge)
    # The file has many expressions; just confirm it executed without raising.
    assert len(results) >= 5


def test_m4e5_justification_path_example_loads(bridge: TemporalBridge):
    results = run_metta_file(EXAMPLES_DIR / "m4e5_justification_path.metta", bridge=bridge)
    # 6 top-level expressions: start-trace, 3 record-rule, justification-path, explain-trace.
    assert len(results) >= 6


def test_m4e6_counterfactual_example_loads(bridge: TemporalBridge):
    results = run_metta_file(EXAMPLES_DIR / "m4e6_counterfactual.metta", bridge=bridge)
    # counterfactual x2, add-belief, cf-shift, support-chain, active-beliefs.
    assert len(results) >= 6


def test_m4e7_styles_example_loads(bridge: TemporalBridge):
    results = run_metta_file(EXAMPLES_DIR / "m4e7_styles.metta", bridge=bridge)
    # 4 styled renders (3 styles + 1 domain).
    assert len(results) >= 4


def test_m4e8_metta_reasoning_example_derives_trace(bridge: TemporalBridge):
    results = run_metta_file(EXAMPLES_DIR / "m4e8_metta_reasoning.metta", bridge=bridge)
    # start-trace, assert, match, record, rules-fired, explain-trace.
    assert len(results) >= 6
    # The match step derived the causal link inside the MeTTa evaluator...
    match_block = results[2]
    assert any("causal-link" in str(atom) for atom in match_block)
    # ...and that MeTTa-derived conclusion was recorded into the M2 trace.
    assert bridge.rules_fired() == ["rule:causal-chain-in-metta"]
    explanation = bridge.explain_belief_trace("Model T price drop")
    assert "moving assembly line" in explanation


def test_pure_metta_wrapper_file_loads_and_helpers_run(bridge: TemporalBridge):
    # The kernel-portable wrapper (metta/temporal_nlg.metta) must load under
    # hyperon and its pure-MeTTa helpers must compose with the grounded ops.
    wrapper = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "temporal_nlg_metta"
        / "metta"
        / "temporal_nlg.metta"
    ).read_text(encoding="utf-8")
    bridge.add_belief("b1", '{"claim":"test claim"}')
    results = run_metta(
        wrapper + '\n!(justify "b1")\n!(contradiction-status)\n',
        bridge=bridge,
    )
    flat = [str(atom) for block in results for atom in block]
    assert any("Justification" in a and "b1" in a for a in flat)
    assert any("Contradictions" in a for a in flat)
