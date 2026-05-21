# Milestone 3 Technical Documentation

## Scope

- M3-E1: Multi-domain temporal graph construction (51,232 records, 35+ domains)
- M3-E2: Temporal explanation fidelity metrics (automatic + LLM-in-the-loop)
- M3-E3: Comprehension and utility user studies (human-collected responses)
- M3-E4: Explanation quality — efficiency (E4a), consistency (E4b), coherence (E4c), granularity (E4d)
- M3-E5: End-to-end QA system benchmark (4 LLM sizes × 2 emb sizes × 5 modes)

---

## Data Inventory

| Dataset | Path | Records |
|---------|------|---------|
| Evaluation QA set | `data/jsonls/temporal_evaluation_set_v2.jsonl` | 295 questions |
| Precomputed graph (v3) | `data/jsonls/temporal_graph_output_v3/` | 51,232 |

---

## Components

### M3-E1 — Multi-Domain Temporal Graph

- Source file: `data/jsonls/temporal_graph.jsonl` (51,232 human-curated temporal facts)
- Key domain distribution: historical=11,408; financial=9,479; technical=6,888; geopolitical=5,533; cultural=5,081; science=4,060; medical=3,663; sports=1,077; business=679 (+ 26 more domains)
- All four required domains (historical, financial, science, medical) are covered with ≥3,663 records each
- Graph preprocessing: external graph-construction utilities maintained outside active repository structure
- Precomputed output: `data/jsonls/temporal_graph_output_v3/`
- Construction command: maintained in the external graph-construction workspace

### M3-E2 — Fidelity Metrics

- Module: `src/temporal_nlg/evaluation/m3_e2.py`
- Runner: `experiments/m3_e2_fidelity/run_fidelity.py`
- Evaluates temporal NLG explanations across five quality dimensions:
  - Point: `timestamp_accuracy`, `entity_coverage`, `context_relevance`
  - Interval: `boundary_accuracy`, `duration_correctness`, `interval_marker_presence`
  - Sequence: `ordering_accuracy`, `step_completeness`, `sequence_marker_presence`
  - Causal: `temporal_constraint_correctness`, `causal_marker_presence`, `entity_coverage`
- Modes: automatic proxy metrics (`--no-human-loop`) or LLM-in-the-loop (`--human-loop llm`)
- Outputs per run: `m3_e2_fidelity.per_item.jsonl`, `m3_e2_fidelity.summary.json`, (optionally) `m3_e2_fidelity.human_loop_llm.jsonl`
- Key runs and artifacts: see `docs/RESULTS_M3.md`
- LLM judges used: `gpt-4.1-nano`, `gpt-4o`, `gpt-5.1`, `gpt-5.2`

### M3-E3 — Comprehension & Utility Studies

- Module: `src/temporal_nlg/evaluation/m3_e3.py`
- Key schemas: `ComprehensionResponse`, `ExplanationItem`, `ComprehensionQuestion`

#### M3-E3a — Comprehension Assessment (human responses)

- Runner: `experiments/m3_e3_human_eval/run_comprehension.py`
- Tasks: `output/m3_e3a_comprehension/m3_e3a_tasks.jsonl` (50 tasks, 5 questions each)
- Results (to populate): `output/m3_e3a_comprehension/m3_e3a_results.jsonl`
- Question types: `mcq`, `fill_blank`, `timeline`, `inference`
- Analysis command:
  ```
  python experiments/m3_e3_human_eval/run_comprehension.py analyze \
      --tasks output/m3_e3a_comprehension/m3_e3a_tasks.jsonl \
      --responses output/m3_e3a_comprehension/m3_e3a_results.jsonl \
      --output-dir output/m3_e3a_comprehension_analysis_llm
  ```

#### M3-E3b — Utility Study (online participant portal)

- Runner: `experiments/m3_e3_human_eval/run_utility.py`
- Tasks: `output/m3_e3b_utility/m3_e3b_tasks.jsonl` (1,109 tasks)
- Web UI: `experiments/m3_e3_human_eval/ui/` — hosted for online participants
- Conditions: `with_explanation` vs. `without_explanation`
- Response schema: `{task_id, participant_id, condition, success, confidence_1_5, time_sec, expert_agreement}`

#### M3-E3c — Cognitive Load Study (online participant portal)

- Runner: `experiments/m3_e3_human_eval/run_cognitive_load.py`
- Scenarios: `output/m3_e3c_cognitive_load/m3_e3c_scenarios.jsonl` (40 scenarios, 4 conditions each)
- Web UI: `experiments/m3_e3_human_eval/ui/` — hosted for online participants
- Conditions: `dense_text`, `structured_narrative`, `timeline_plus_text`, `interactive`
- Response schema: NASA-TLX subscales + retention score + attention on key facts

### M3-E4 — Explanation Quality

Module: `src/temporal_nlg/evaluation/m3_e4.py`

#### M3-E4a — Update Efficiency

- Runner: `experiments/m3_e4_efficiency_consistency/run_efficiency.py`
- Scenarios: 1,000 scenarios × 7 update methods
- Results: `output/m3_e4a_efficiency_analysis_methods/`
- Methods: `template`, `hybrid`, `llm`, `selective`, `full_regen`, `delta_only`, `cached`

#### M3-E4b — Consistency Under Fact Revision

- Runner: `experiments/m3_e4_efficiency_consistency/run_consistency.py`
- Schemas: `ConsistencyFact`, `ConsistencyRevision`, `ConsistencyResult`
- Facts: `output/m3_e4b_consistency/m3_e4b_facts.jsonl` (1,000 facts)
- Revisions: `output/m3_e4b_consistency/m3_e4b_revisions.jsonl` (4,000: date_correction, add_causal, contradiction, removal)
- Results CSV (to populate): `output/m3_e4b_consistency/m3_e4b_results_template.csv`
- Analysis command:
  ```
  python experiments/m3_e4_efficiency_consistency/run_consistency.py analyze \
      --facts output/m3_e4b_consistency/m3_e4b_facts.jsonl \
      --revisions output/m3_e4b_consistency/m3_e4b_revisions_filled.jsonl \
      --results output/m3_e4b_consistency/m3_e4b_results_filled.csv \
      --output-dir output/m3_e4b_consistency_analysis_llm
  ```

#### M3-E4c — Narrative Coherence

- Runner: `experiments/m3_e4_efficiency_consistency/run_coherence.py`
- Scenarios: 100 scenarios × 5 narrative styles
- Results: `output/m3_e4c_coherence_analysis_auto/`
- Key results: logical=95.3%, semantic=67.1%, narrative=54.0%

#### M3-E4d — Temporal Granularity

- Runner: `experiments/m3_e4_efficiency_consistency/run_granularity.py`
- Scenarios: 200 scenarios × 7 granularity scales
- Results: `output/m3_e4d_granularity_analysis_methods/`, `output/m3_e4d_granularity_analysis_proxy/`

### M3-E5 — End-to-End QA Benchmark

- Runner: `experiments/m3_e5_benchmark/run_m3_e5.py`
- Milestone orchestrator: `experiments/run_all_m3_experiments.py`
- Schema spec: `experiments/m3_e5_benchmark/results_schema.json`
- Eval set: `data/jsonls/temporal_evaluation_set_v2.jsonl` (295 questions)
- Results directory: `output/m3_e5_results/` (run IDs + `MATRIX.json`)
- Core retrieval assets: `data/jsonls/temporal_graph.jsonl` and `data/jsonls/temporal_graph_output_v3/` (graph-grounded evidence source)

#### Experiment matrix

| Dimension | Values |
|-----------|--------|
| LLM model | llm_0.8b, llm_2b, llm_4b, llm_9b |
| Embedding model | emb_0.6b, emb_4b |
| Pipeline mode | pure_llm, rag_small_emb, rag_large_emb, graph_small_emb, graph_large_emb |
| **Total** | **36 configurations** |

Run ID format: `<llm_id>__<mode>[__<emb_id>]`

#### Matrix Snapshot

- Finalized matrix contains 25 configs: 15 Qwen project runs (`llm_0.8b`, `llm_2b`, `llm_4b` × 5 modes) plus 10 OpenAI LCEL comparison runs.
- Aggregated matrix generated: `output/m3_e5_results/MATRIX.json`
- Raw duplicate artifacts were archived outside active repo paths.
- Graph-grounded modes consistently lead the matrix, reinforcing that temporal graph dataset quality is the primary M3 performance lever.
- All-result conclusion: graph retrieval outperforms pure and RAG baselines for each tested model size in exact-match evaluation.
- Final comparison tables and charts: `docs/RESULTS_M3.md` (M3-E5 section).
- Static chart images: `docs/images/m3/`.

#### Commands

```bash
# Milestone 3 orchestrator (all M3 experiments with defaults/skips)
python experiments/run_all_m3_experiments.py

# List all configs
python experiments/m3_e5_benchmark/run_m3_e5.py --list --output-dir output/m3_e5_results

# Run a single config
python experiments/m3_e5_benchmark/run_m3_e5.py \
    --llm-id llm_9b --mode graph_large_emb --emb-id emb_4b \
    --eval-set data/jsonls/temporal_evaluation_set_v2.jsonl \
    --graph-dir data/jsonls/temporal_graph_output_v3 \
    --output-dir output/m3_e5_results

# Run all (skips existing outputs)
python experiments/m3_e5_benchmark/run_m3_e5.py --run-all \
    --eval-set data/jsonls/temporal_evaluation_set_v2.jsonl \
    --graph-dir data/jsonls/temporal_graph_output_v3 \
    --output-dir output/m3_e5_results

# Aggregate results
python experiments/m3_e5_benchmark/run_m3_e5.py --aggregate --output-dir output/m3_e5_results
```

#### Orchestrator policy (current)

- OpenAI benchmark wrappers (`run_lcel_gpt_models.py`, `run_m3_e5_lcel_openai_models.py`) are skipped by default in `run_all_m3_experiments.py`.
- Local Qwen M3-E5 runs are executed only when a CUDA-capable GPU is detected; otherwise the orchestrator records an explicit skip reason.
- M3-E4 prediction generation (`run_generate_predictions.py`) is run with template/baseline defaults when prerequisite scenarios are available.
- Embedding assets are expected under `data/jsonls/temporal_graph_output_v3/embeddings/` and `data/jsonls/temporal_graph_output_v3/embeddings_4b/`.

---

## Services

| Service | File | Purpose |
|---------|------|---------|
| LLM server | `services/llm_server.py` | Local LLM inference endpoint |
| Embeddings server | `services/embeddings_server.py` | Local embedding endpoint |

See `docker-compose.yml` for containerized deployment.

---

## Trained Models

| Artifact | Path | Description |
|----------|------|-------------|
| QLoRA adapter | `models/temporal_nlg_lora/` | Phi-4-mini QLoRA for temporal NLG |
| Merged model | `models/temporal_nlg_merged/` | Full merged weights |


