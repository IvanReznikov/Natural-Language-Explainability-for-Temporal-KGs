# Milestone 2 - Additional Notes

## Dataset and Artifact Inventory
- Intent classification (M2-E2): 1,251 annotated queries at `experiments/m2_e2_intent/data/annotated_queries.jsonl` (multi-label: point_in_time, interval, sequence, causal, comparative, aggregation, prediction, explanation). Split: 20% test (250), 10% val (100), train (900); seed=42. All 15 TF-IDF ablation run artifacts (T1–T15) committed under `experiments/m2_e2_intent/results/<run_id>/metrics.json`; reproducible via `scripts/run_intent_sweep.py`.
- Parser/intent hybrid (M2-E3): 1,137 gold temporal queries at `experiments/m2_e3_parse/data/temporal_queries_gold.jsonl` with splits regenerated (seed=13) into `experiments/m2_e3_parse/data/splits`. Model weights are downloaded via `scripts/download_m2_models.py` into `experiments/m2_e3_parse/artifacts/intent/model.safetensors` (MiniLM, 87 MB) and `experiments/m2_e3_parse/artifacts/parser/model.safetensors` (Flan-T5-small, 293 MB). Canonical construct run `0dc0fd29ef4a49d3b6bb84291813368f` at `experiments/m2_e3_construct/runs/0dc0fd29ef4a49d3b6bb84291813368f/metrics.json` (template_accuracy=1.000).
- Result taxonomy (M2-E4): ~1.4k rows each for taxonomy/summaries/consistency at `experiments/m2_e4_taxonomy/data/{taxonomy.jsonl,summaries.jsonl,narrative_consistency.jsonl}` plus stratified splits at `experiments/m2_e4_taxonomy/data/splits` (train/val/test = 1,126/136/148 for taxonomy).
- Trace corpora (M2-E5): synthetic traces via `experiments/m2_e5_trace_meta_query/generate_trace_corpus.py` (default 1,000); curated small sample at `experiments/m2_e5_trace_meta_query/output/small_traces.jsonl`; micro-benchmark log at `experiments/m2_e5_trace_meta_query/output/trace_bench.txt`.
- Query/trigger corpora (M2-E6): synthetic query corpus generator at `experiments/m2_e6_query_store_triggers/generate_query_corpus.py` (default 500); trigger demo outputs stored under `experiments/m2_e6_query_store_triggers/output/`.
- E2E harness corpora (M2-E7): `experiments/m2_e7_harness/input/queries.jsonl` (2,210 rows — 1,199 real queries from annotated_queries.jsonl + 1,011 template-synthetic; all using the 8-class M2 taxonomy with `required_facts` and `max_latency_ms` annotations) and matching `trace.jsonl` (2,210 rows with proper intent conclusions and `extra_conclusions` expansion). Harness outputs: `experiments/m2_e7_harness/output/report.json` and `results.jsonl`.

## Model/Code Artifacts
- Intent classifier sweep results under `experiments/m2_e2_intent/results/<run_id>/metrics.json` for all 15 TF-IDF ablation runs (T1–T15); best macro/micro/subset from isotonic-calibrated TF-IDF + LR (T13/T15). Neural runs M16/M17 reported in RESULTS_M2.md; M16 weights at `experiments/m2_e3_parse/artifacts/intent/`.
- Parser fine-tunes: intent encoder at `experiments/m2_e3_parse/artifacts/intent/model.safetensors`; Flan-T5-small parser at `experiments/m2_e3_parse/artifacts/parser/model.safetensors`; eval outputs at `output/m2_e3_eval/<run_id>/`.
- Query construction: canonical run `0dc0fd29ef4a49d3b6bb84291813368f` at `experiments/m2_e3_construct/runs/0dc0fd29ef4a49d3b6bb84291813368f/`; see `experiments/m2_e3_construct/README.md` for full run inventory.
- Taxonomy classifier: default LinearSVC word+char model saved at `experiments/m2_e4_taxonomy/output/e4a_taxonomy/taxonomy_model.joblib` with metrics + sweep at the same folder.
- Trace/meta-query utilities: `temporal_nlg.tms.trace`, `temporal_nlg.tms.meta_query`, contradiction detector in `temporal_nlg.tms.contradiction`; CLI in `experiments/m2_e5_trace_meta_query/meta_query_cli.py`.
- Query/result/trigger engine: `temporal_nlg.tms.query_store.QueryStore`, `temporal_nlg.tms.result_store.ResultStore`, `temporal_nlg.tms.trigger_engine.TriggerEngine` with demos in `experiments/m2_e6_query_store_triggers`.
- E2E harness: `experiments/m2_e7_harness/run_e2e.py` plus specs (CORPUS_SPEC.md, TRACE_SPEC.md, QUERY_ANNOTATION_GUIDE.md) and generators `generate_e2e_queries.py` / `generate_traces.py` (rewritten to use M2 taxonomy intents and required_facts).

## Experiment Coverage Summary
| Experiment | Coverage / result |
|-----------|-------------------|
| M2-E2 | 15-run TF-IDF ablation sweep (artifacts committed); top macro F1 **0.851** (T13/T15). Neural M16 achieves **0.911** and hybrid M17 achieves **0.970** — milestone target >0.90 met. |
| M2-E3 | Parser + intent hybrid; test split 115 rows; 0 validation errors; intent macro F1 **0.9703**. Model weights available via download script. Canonical construct run: 1,137/1,137 templates correct. |
| M2-E4 | Taxonomy classification; best test accuracy **0.993** and macro F1 **0.994** with linear SVM (E6). |
| M2-E5 | Trace/meta-query tooling validated on the sample corpus; mean overhead **0.005 ms** per trace batch. |
| M2-E6 | Trigger/query/result store chain validated end to end, including stale marking on KB update. |
| M2-E7 | Harness over **2,210** real+synthetic query/trace pairs (M2 taxonomy intents, required_facts, latency budget): **ok=2210, fail=0**. |

## Temporal Graph Query Bridge
- Added a lightweight query layer over `data/jsonls/temporal_graph_output_v3/` for temporal reasoning questions without Neo4j.
- New package: `src/temporal_nlg/graph_query/` with:
  - `index.py`: in-memory indexes over `nodes.jsonl`, `edges.jsonl`, `tags.jsonl`, and `processed_graph.jsonl`.
  - `retrieval.py`: rule-focused query retrieval for reason, analogical transfer, and start-affecting question classes.
  - `lcel.py`: LangChain LCEL pipeline for parse -> retrieve -> answer, with optional LLM refinement.
  - `visualization.py`: built-in Mermaid subgraph rendering from retrieved evidence.
- This bridge keeps the milestone-2 semantics visible while reusing the M3 graph artifacts for retrieval and answer assembly.

## Usage Pointers
- Run all M2 examples: `python examples/run_all_m2_examples.py`
- Run all M2 experiments: `python experiments/run_all_m2_experiments.py`
- Run M2 benchmarks: `python benchmarks/milestone2/run.py`
- Quick inference: taxonomy model via experiments/m2_e4_taxonomy/predict_taxonomy.py; trace meta-queries via experiments/m2_e5_trace_meta_query/meta_query_cli.py; trigger chain via experiments/m2_e6_query_store_triggers/e2e_chain_demo.py; E7 harness via experiments/m2_e7_harness/run_e2e.py.
- Annotation guides: experiments/m2_e7_harness/QUERY_ANNOTATION_GUIDE.md and TRACE_SPEC.md for hand-editing corpora; m2_e4/split_data.py and report_splits.py for regenerating stratified splits.
- Repro: prefer stable filenames noted above; regeneration scripts accept --count/--seed for deterministic reruns.

## Milestone 2 Architecture

### Overview
Milestone 2 adds intent/taxonomy models, parser hybrid, trace/meta-query utilities, triggerable query/result stores, and an end-to-end harness. Components stay modular: core TMS services in `src/temporal_nlg/tms`, experiments and corpora in `experiments/m2_*`, and runnable demos in `examples/`.

### Component Map
```text
       +--------------------+
       |  Corpora (JSONL)   |
       |  m2_e2/m2_e3/m2_e4 |
       |  m2_e5 traces      |
       |  m2_e7 queries     |
       +----------+---------+
      |
      v
+-------------------------------+          +---------------------------+
|  Models (E2/E3/E4)            |          |  TMS Runtime (E5/E6/E7)   |
|  - Intent TF-IDF + LR/SVM      |          |  - TraceRecorder/QueryTrace |
|  - MiniLM intent encoder       |          |  - meta_query, contradiction |
|  - Flan-T5-small parser        |          |  - QueryStore/ResultStore  |
|  - Taxonomy LinearSVC          |          |  - TriggerEngine           |
+---------------+---------------+          +--------------+------------+
    |                                 |
    v                                 v
  +----------------+               +-----------------------+
  | Generation /   |               | End-to-End Harness    |
  | Evaluation     |               | (experiments/m2_e7_harness) |
  | (parser + rules)|              | - ingest traces+queries |
  +----------------+               | - eval intent/facts    |
                                   | - emit report/results  |
                                   +-----------------------+
```

### Data and Artifacts
- Intent (E2): `experiments/m2_e2_intent/data/annotated_queries.jsonl`; runs under `experiments/m2_e2_intent/results/` with metrics.json per ID.
- Parser (E3): `experiments/m2_e3_parse/data/temporal_queries_gold.jsonl`; models at `experiments/m2_e3_parse/artifacts/{intent,parser}`.
- Taxonomy (E4): `experiments/m2_e4_taxonomy/data/...`; default model at `experiments/m2_e4_taxonomy/output/e4a_taxonomy/taxonomy_model.joblib`.
- Traces (E5): synthetic via `generate_trace_corpus.py`, sample at `experiments/m2_e5_trace_meta_query/output/small_traces.jsonl`, micro-bench at `trace_bench.txt`.
- Triggers/stores (E6): query/result JSONL in `experiments/m2_e6_query_store_triggers/output/` from demos and benchmarks.
- Harness (E7): `experiments/m2_e7_harness/input/{queries.jsonl,trace.jsonl}` (2,210 each), outputs under `experiments/m2_e7_harness/output/`.

### Runtime Flow (E5->E7)
1. **Trace capture (E5):** Rules fire through `TraceRecorder` -> `QueryTrace` JSONL.
2. **Meta-queries (E5):** Operators inspect traces (rules fired, explanations, contradictions).
3. **Triggers (E6):** `TriggerEngine` evaluates predicates over `TriggerContext`, writes queries to `QueryStore`.
4. **Result reification (E6):** `ResultStore.upsert` stores results with dependent facts; `mark_stale_by_facts` invalidates on changes.
5. **Harness eval (E7):** `run_e2e.py` ingests annotated queries + traces, expands extra conclusions, checks intent/required_facts/latency, emits `results.jsonl` + `report.json`.

### Integration Points
- **Models -> Rules:** Intent/taxonomy outputs can seed rule inputs (fact_ids like `intent`, `diagnosis`).
- **Rules -> Stores:** Rule conclusions become facts in traces; triggers use facts to spawn new queries; results track dependent_facts for invalidation.
- **Human-in-the-loop:** Query annotations (`required_facts`, `max_latency_ms`) and trace specs are editable; rerun harness to validate.

### Testing and Demos
- CLI: `experiments/m2_e5_trace_meta_query/meta_query_cli.py`, `experiments/m2_e6_query_store_triggers/e2e_chain_demo.py`, `experiments/m2_e7_harness/run_e2e.py`.
- Examples: `examples/milestone2/*.py` mirror the same flows for quick smoke tests.

### Operational Notes
- All workflows are CPU-only; deterministic seeds on splits/generators.
- Stable file names for artifacts to ease repro; JSONL everywhere for streaming friendliness.
- Keep fact IDs short and consistent across traces, triggers, and harness checks.

