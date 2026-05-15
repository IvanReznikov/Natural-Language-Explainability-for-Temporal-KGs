# TECH_DOCUMENTATION_M1

## Scope
Milestone 1: Temporal NLG library (templates, LLM, hybrid, evaluation, data loaders). This document links to API definitions (see `docs/API.md`) and summarizes architecture at a high level.

## Architecture (summary)
- Core templates: `src/temporal_nlg/core/templates.py` — `TemporalFact`, `TemplateType`, `TemplateRenderer`.
- Models: `src/temporal_nlg/models/` — `LLMGenerator`, `HybridGenerator`, `GenerationResult`.
- Evaluation: `src/temporal_nlg/evaluation/` — `AccuracyEvaluator`, `calculate_flesch_score`.
- Data loaders: `src/temporal_nlg/data/loaders.py` — example generators via `generate_examples`.

## Public API Surface
- Exported via `src/temporal_nlg/__init__.py` (`TemplateRenderer`, `TemplateType`, `TemporalFact`, `LLMGenerator`, `HybridGenerator`, `AccuracyEvaluator`, `GenerationResult`, `calculate_flesch_score`).
- Detailed usage and signatures: see `docs/API.md`.

## Tests & Coverage
- Unit/integration tests under `tests/` and `experiments/m1_e5_integration/integration_tests.py`.
- CI runs unit + integration with coverage (see `.github/workflows/ci.yml`).

## Benchmarks
- Offline performance script: `experiments/m1_e5_integration/performance_benchmarks.py` (template, cached, uncached hybrid, evaluator batch).
- Primary M1 result copies are centralized under `output/m1_e5_integration/results/<timestamp>/`.

## Human Evaluation
- Guidance: `docs/HUMAN_EVAL.md`; data schema: `docs/human_eval_template.csv`.
- Aggregator: `experiments/m1_e5_integration/human_eval_aggregate.py`.

## Packaging
- Configured via `pyproject.toml`, `setup.py`, `MANIFEST.in`; version set in `src/temporal_nlg/__init__.py`.

## Dependencies
- Runtime pins in `requirements.txt` (OpenAI/langchain/textstat/rich/etc.). Dev extras in `pyproject.toml` (`pytest`, `pytest-cov`, `black`, `mypy`).
