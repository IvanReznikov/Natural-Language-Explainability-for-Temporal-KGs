# temporal_nlg package

This folder contains the reusable Python package for temporal explanation generation, temporal expression handling, temporal-memory-style reasoning, graph retrieval, and milestone evaluation helpers.

It is the core implementation surface used by experiments and scripts across Milestones 1, 2, and 3.

## Design goals

- Keep reusable logic in importable modules under `src/temporal_nlg`.
- Keep experiment orchestration, benchmarks, and reporting scripts outside the package.
- Keep service/runtime wrappers outside the package, while they import package components.

## Package layout

### `core/`

Foundational template abstractions and fact containers.

- `core/templates.py`
  - `TemplateType`, `TemporalFact`
  - abstract template base classes and rendering helpers

### `templates/`

Milestone 1 template libraries and concrete template banks.

- `templates/point_in_time.py`
- `templates/intervals.py`
- `templates/sequences.py`
- `templates/causality.py`
- `templates/overlaps.py`
- `templates/containment.py`
- `templates/precedence.py`
- `templates/recurrence.py`

These modules provide high-coverage, deterministic template rendering for temporal relation types used in M1 and later hybrid pipelines.

### `models/`

Generation and embedding model adapters.

- `models/llm_generator.py` - generic LLM wrapper
- `models/hybrid_generator.py` - template-first/hybrid generation path
- `models/qwen_generator.py` - local Qwen generation and embedding wrappers

### `data/`

Data loading helpers used by evaluation and examples.

- `data/loaders.py`
- `data/_loaders.py`

### `evaluation/`

Milestone evaluation logic and aggregation schemas.

- `evaluation/m1_e1_evaluation.py` - M1-E1 evaluation framework
- `evaluation/accuracy.py` - M1-style accuracy/readability helpers
- `evaluation/m3_e2_fidelity.py` - M3-E2 proxy fidelity metrics
- `evaluation/m3_e2_human_loop.py` - human or LLM-in-the-loop scoring helpers
- `evaluation/m3_e3.py` - comprehension and utility study schemas/aggregation
- `evaluation/m3_e4.py` - efficiency and consistency schemas/aggregation

### `temporal_expr/`

Milestone 2 temporal expression pipeline.

- `temporal_expr/tagger.py` - baseline temporal expression tagger
- `temporal_expr/normalizer.py` - normalization logic
- `temporal_expr/context_resolver.py` - context-aware interpretation
- `temporal_expr/schemas.py` - datatypes for temporal extraction
- `temporal_expr/datasets.py` and `temporal_expr/evaluation.py` - dataset/eval helpers

### `tms/`

Temporal memory and traceability components (M2-E5/E6 class of features).

- `tms/belief_store.py`
- `tms/justification.py`
- `tms/trace.py`
- `tms/trace_explain.py`
- `tms/meta_query.py`
- `tms/contradiction.py`
- `tms/counterfactual.py`
- `tms/query_store.py`
- `tms/result_store.py`
- `tms/trigger_engine.py`

### `explain/`

Explanation rendering and path narrative assembly.

- path extraction and explanation text builders
- belief- and justification-aware rendering adapters
- counterfactual explanation helpers

### `graph_query/`

Milestone 3 retrieval and QA pipeline primitives.

- `graph_query/index.py` - in-memory graph index over JSONL artifacts
- `graph_query/semantic.py` - semantic edge retrieval and embedding-backed search
- `graph_query/grounding.py` - semantic grounding logic
- `graph_query/retrieval.py` - query-type-aware answer construction
- `graph_query/lcel.py` - end-to-end temporal graph QA pipeline
- `graph_query/row_index.py` - row-level QA retrieval support
- `graph_query/visualization.py` - answer-to-mermaid utility

### `path_narratives/`

Path narrative renderer and report datatypes for path-style explanations.

### `utils/`

Utility namespace for cross-cutting helpers (currently minimal/placeholder).

## Public API surface

Top-level imports in `temporal_nlg/__init__.py` expose the intended package surface for external callers, including:

- `TemplateRenderer`, `TemplateType`, `TemporalFact`
- `LLMGenerator`, `HybridGenerator`, `QwenLocalGenerator`, `QwenEmbeddingModel`
- `AccuracyEvaluator`, `calculate_flesch_score`
- temporal expression classes (`TemporalTagger`, `TemporalNormalizer`, etc.)
- graph query classes (`TemporalGraphIndex`, `GraphRetriever`, `TemporalGraphLCELPipeline`)

Prefer importing from these package exports where possible.

## Milestone coverage map

This package covers most reusable milestone logic, but not all operational code.

- M1
  - Covered in package: templates, hybrid generation core, M1 evaluation helpers.
  - Outside package: experiment drivers and benchmark scripts.
- M2
  - Covered in package: temporal expression stack (E1), TMS/trace/meta-query/query-store/result-store/trigger logic (E5/E6), and related explanation helpers.
  - Partially outside package: intent/parser training and CLI inference utilities under `scripts/m2_e3` and experiment orchestration.
- M3
  - Covered in package: graph retrieval/indexing/query pipeline, fidelity and study/efficiency metric schemas and aggregations.
  - Partially outside package: graph artifact construction utilities, embedding precompute/build orchestration scripts, benchmark matrix runners, and service process wrappers.

## What is intentionally outside this package

The following are intentionally not in `src/temporal_nlg` and should remain separate unless a refactor is planned:

- `experiments/` - run scripts, benchmark orchestration, artifact writing, study flows
- `scripts/` - operational pipelines, one-off generators, environment bootstrap scripts
- `services/` - HTTP servers and runtime wrappers
- external graph-construction utilities - dataset-to-graph artifact build pipeline

This separation keeps package code importable and testable while allowing operational scripts to evolve faster.

## How to decide where new code belongs

Put code in `src/temporal_nlg` if it is:

- reusable by multiple experiments
- importable without side effects
- unit-testable without external process orchestration

Keep code outside package if it is:

- a benchmark runner, migration, or one-time utility
- tightly coupled to filesystem layouts or artifact directories
- a service startup script or shell-oriented orchestration

## Suggested next refactor candidates

If you want tighter package ownership, the highest-value migration path is:

1. Extract reusable logic from `scripts/m2_e3/*` into `src/temporal_nlg` submodules.
2. Keep thin CLI wrappers in `scripts/` that call those new package modules.
3. Do the same for reusable parts of graph rebuild/precompute scripts.

This preserves CLI workflows while reducing cross-folder coupling.