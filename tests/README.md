# Test Guide

This layout mirrors the repository structure guide. Tests are grouped by scope:

- `test_templates_core.py`: Core unit tests for template primitives and renderer validation.
- `test_templates_integration.py`: Generator integration (template/LLM/hybrid) and routing flows.
- `test_templates/`: Deeper coverage for template libraries and accuracy evaluation.
- `experiments/`: Experiment-aligned suites grouped by milestone/experiment id (for example `experiments/m2_e2/`, `experiments/m2_e3/`, `experiments/m2_e7_harness/`).
- `test_baseline.py`: Smoke checks for data loaders, belief tracking, and counterfactual utilities.
- `test_visualization_features.py`: Path narratives, graph explanations, and justification rendering.
- `test_data/`: Place fixtures under `deltas/`, `snapshots/`, or `temporal/` when needed.

## Quick commands

- Run all tests: `pytest tests`
- Run template core unit tests: `pytest tests/test_templates_core.py`
- Run integration (generators): `pytest tests/test_templates_integration.py`
- Run experiment-aligned tests: `pytest tests/experiments`
- Run visualization tests: `pytest tests/test_visualization_features.py`
- Run baseline smoke: `pytest tests/test_baseline.py`

Markers and additional configuration live in `conftest.py`.


