# Milestone 3 — Additional Notes

## Dataset and Artifact Inventory

- **Temporal KG** (`data/jsonls/temporal_graph.jsonl`): 51,232 facts across 35+ domains, human-curated. Top domains: historical=11,408; financial=9,479; technical=6,888; geopolitical=5,533; cultural=5,081; science=4,060; medical=3,663. Required M3 domains (historical, financial, science, medical) all present with substantial coverage.
- **Evaluation QA set** (`data/jsonls/temporal_evaluation_set_v2.jsonl`): 295 questions (easy=108, medium=125, hard=62). Domains: economics=85, politics=37, sports=33, entertainment=33, technology=32, temporal_reasoning=30, history=19. Types: entity_at_time=62, before_after=59, temporal_comparison_yes_no=52, champion_at_time=27 (+ more).
- **Precomputed graph** (`data/jsonls/temporal_graph_output_v3/`): the retrieval backbone for M3-E5; this graph index is the main reason graph-grounded systems outperform pure parametric QA.
- **Fidelity outputs** (`output/m3_e2_fidelity_*/`): runs with `gpt-4.1-nano`, `gpt-4o`, `gpt-5.1`, `gpt-5.2` judges.
- **Comprehension tasks** (`output/m3_e3a_comprehension/`): 50 tasks × 5 questions with human-collected responses.
- **Utility tasks** (`output/m3_e3b_utility/`): 1,109 tasks for online participant collection.
- **Cognitive load scenarios** (`output/m3_e3c_cognitive_load/`): 40 scenarios × 4 conditions for online participant collection.
- **Efficiency analysis** (`output/m3_e4a_efficiency_analysis_methods/`): 1,000 scenarios × 7 methods.
- **Consistency dataset** (`output/m3_e4b_consistency/`): 1,000 facts × 4 revision types; human study results recorded in filled artifacts.
- **Coherence analysis** (`output/m3_e4c_coherence_analysis_auto/`): 100 × 5 styles; logical=95.3%, semantic=67.1%, narrative=54.0%.
- **Granularity analysis** (`output/m3_e4d_granularity_analysis_{proxy,methods}/`): 200 × 7 scales.
- **QA benchmark** (`output/m3_e5_results/`): finalized matrix with 25 configs (`MATRIX.json`): 15 Qwen project runs + 10 OpenAI LCEL comparison runs.

---

## Experiment Coverage Summary

| Experiment | Coverage / result |
|-----------|-------------------|
| M3-E1 | 51,232-record temporal graph across 35+ domains. |
| M3-E2 | Fidelity metrics with 4 LLM judges and proxy scoring. |
| M3-E3a | Comprehension tasks exported and evaluated with participant responses. |
| M3-E3b | Utility portal built for online participant collection. |
| M3-E3c | Cognitive-load portal built for online participant collection. |
| M3-E4a | Efficiency analysis across 1,000 scenarios and 7 methods. |
| M3-E4b | Consistency dataset and revision protocol completed. |
| M3-E4c | Coherence analysis completed with logical, semantic, and narrative scores. |
| M3-E4d | Granularity analysis completed across 200 scenarios and 7 scales. |
| M3-E5 | Final benchmark matrix with 25 completed runs. |

---

## Design Decisions

### M3-E3: Human response collection
- **E3a** (comprehension): 50 × 5 = 250 questions were exported and evaluated with real participant responses.
- **E3b/E3c** (utility, cognitive load): online participant portal used for utility and NASA-TLX collection. Web UI at `experiments/m3_e3_human_eval/ui/` handles data collection.

### M3-E4b: Fact revision consistency protocol
- Phase 1: revision deltas are prepared for all 4,000 revision pairs (facts × revision_type). Delta schema varies by type: `date_correction`, `add_causal`, `contradiction`, `removal`.
- Phase 2: revised explanations are evaluated on `update_accuracy`, `contradiction_detected`, `coherence_rating_1_5`, `resolution_time_sec`.

### M3-E5: Full 295-question eval set
- GPU notebook outputs from `temporal_eval_experiments.ipynb` were normalized into run IDs under `output/m3_e5_results/`.
- Run set: 25 configurations total in `MATRIX.json` (15 Qwen modes + 10 OpenAI LCEL pure runs).
- Most runs use `data/jsonls/temporal_evaluation_set_v2.jsonl` (295 questions); earlier pure runs for `llm_0.8b` and `llm_2b` were produced on `data/jsonls/temporal_evaluation_set.jsonl` (310 questions) and are retained for reproducibility.
- Duplicate GPU artifacts (`matrix_results.*`, `logs/`, and original `runs/` folder names) were archived outside the active repository structure.
- The graph dataset assets (`temporal_graph.jsonl` + `temporal_graph_output_v3`) are the central M3 deliverable; M3-E5 gains should be interpreted as evidence quality + retrieval quality improvements.
- All-result conclusion: graph retrieval is best by exact-match at every tested model size (0.8B, 2B, 4B), and mode-level means keep graph modes at the top.
- Detailed final tables and charts are documented in `docs/RESULTS_M3.md` (M3-E5 section).
- Static chart images are stored in `docs/images/m3/`.

### Trained model
- Fine-tuned Phi-4-mini with QLoRA on temporal NLG tasks (adapter at `models/temporal_nlg_lora/`, merged weights at `models/temporal_nlg_merged/`). Used in GPU notebook experiments (`temporal_eval_experiments.ipynb`).

---

## Usage Pointers

- **Run all M3 examples**: `python examples/run_all_m3_examples.py`
- **Run all M3 experiments (orchestrator)**: `python experiments/run_all_m3_experiments.py`
- **Seed M3-E5 placeholders**: `python scripts/seed_m3_e5_placeholders.py`
- **M3-E5 aggregate matrix**: `python experiments/m3_e5_benchmark/run_m3_e5.py --aggregate --output-dir output/m3_e5_results`
- **Fidelity re-run**: `python experiments/m3_e2_fidelity/run_fidelity.py` (see runner `--help`)
- **Coherence/granularity**: `python experiments/m3_e4_efficiency_consistency/run_coherence.py` / `run_granularity.py`

---

## Reproducibility Notes

- All randomness in graph sampling and experiment splits uses `seed=13` unless otherwise noted.
- The eval set `temporal_evaluation_set_v2.jsonl` is stable and versioned — do not modify it; create a new version file if questions must change.
- LLM judge calls (M3-E2) are cached in `data/jsonls/` LLM response files where applicable.
- M3-E5 run IDs encode the full configuration (`llm_id__mode[__emb_id]`) to prevent collision.
- Local orchestration policy in `experiments/run_all_m3_experiments.py`: OpenAI wrappers are skipped by default; local Qwen M3-E5 runs require CUDA GPU detection.



## Download Embedding Files
- embeddings folder - https://storage.googleapis.com/explanability-for-temporal-graphs/embeddings.zip
- embeddings_4b folder - https://storage.googleapis.com/explanability-for-temporal-graphs/embeddings_4b.zip

### Required target location

Unpack both archives into:

- `data/jsonls/temporal_graph_output_v3/`

Expected result:

- `data/jsonls/temporal_graph_output_v3/embeddings/`
- `data/jsonls/temporal_graph_output_v3/embeddings_4b/`

If your download arrives as split zip files (for example `embeddings_4b-...-001.zip`, `...-002.zip`, etc.), extract all parts to the same destination folder above.

### Windows PowerShell example

```powershell
$dest = "data/jsonls/temporal_graph_output_v3"

# Extract standard archives
Expand-Archive -Path "data/jsonls/temporal_graph_output_v3/embeddings.zip" -DestinationPath $dest -Force
Expand-Archive -Path "data/jsonls/temporal_graph_output_v3/embeddings_4b.zip" -DestinationPath $dest -Force

# Or, if you downloaded split embeddings_4b zip parts
Get-ChildItem -Path $dest -Filter "embeddings_4b-*.zip" |
	Sort-Object Name |
	ForEach-Object { Expand-Archive -Path $_.FullName -DestinationPath $dest -Force }
```

### Verify extraction

```powershell
Get-ChildItem data/jsonls/temporal_graph_output_v3
```

You should see the `embeddings/` and `embeddings_4b/` directories in addition to the graph files (`nodes.jsonl`, `edges.jsonl`, `processed_graph.jsonl`, `qa_index.jsonl`).