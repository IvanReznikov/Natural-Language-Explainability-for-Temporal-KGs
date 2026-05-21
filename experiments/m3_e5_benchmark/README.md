# M3-E5: End-to-End QA System Benchmark

## Overview

M3-E5 is a systematic benchmark of the full temporal QA pipeline across multiple
model sizes and retrieval strategies. It directly measures the integration of all
Milestone components under realistic conditions on the curated evaluation set.

## Experiment Matrix

### LLM Models (4)
| ID | Model size |
|----|-----------|
| llm_0.8b | ~0.8B parameter LLM (e.g., Qwen2.5-0.5B or similar) |
| llm_2b   | ~2B parameter LLM |
| llm_4b   | ~4B parameter LLM |
| llm_9b   | ~9B parameter LLM (e.g., Qwen2.5-7B / Llama-3.1-8B) |

### Embedding Models (2)
| ID | Model size |
|----|-----------|
| emb_0.6b | ~0.6B embedding model (e.g., all-mpnet-base-v2 or similar) |
| emb_4b   | ~4B embedding model (e.g., E5-mistral-7b or large multilingual) |

### Pipeline Modes (5)
| ID | Description |
|----|-------------|
| pure_llm            | LLM only, no retrieval |
| rag_small_emb       | RAG with emb_0.6b |
| rag_large_emb       | RAG with emb_4b |
| graph_small_emb     | Graph-augmented retrieval with emb_0.6b |
| graph_large_emb     | Graph-augmented retrieval with emb_4b |

### Run Configuration Matrix

`pure_llm` mode uses no embeddings → 4 runs.
All other modes use both embedding sizes → 4 × 2 × 4 = 32 runs.
**Total: 36 distinct configurations.**

Run ID format: `<llm_id>__<mode>[__<emb_id>]`

Examples:
- `llm_9b__pure_llm`
- `llm_9b__graph_large_emb__emb_4b`
- `llm_2b__rag_small_emb__emb_0.6b`

## Datasets

- **Primary eval set**: `data/jsonls/temporal_evaluation_set_v2.jsonl` (295 questions)
  - Difficulties: easy=108, medium=125, hard=62
  - Domains: economics=85, politics=37, sports=33, entertainment=33, technology=32, ...
- **Graph**: `data/jsonls/temporal_graph_output_v3/` (precomputed)

### Embedding asset setup (required)

Before running M3-E5 retrieval modes, download and unpack the embedding artifacts into the graph directory:

- `https://storage.googleapis.com/explanability-for-temporal-graphs/embeddings.zip`
- `https://storage.googleapis.com/explanability-for-temporal-graphs/embeddings_4b.zip`

Target directory:

- `data/jsonls/temporal_graph_output_v3/`

After extraction, both directories must exist:

- `data/jsonls/temporal_graph_output_v3/embeddings/`
- `data/jsonls/temporal_graph_output_v3/embeddings_4b/`

If your environment downloads split files (for example `embeddings_4b-...-001.zip`, `...-002.zip`, etc.), extract all parts into the same target directory.

Windows PowerShell example:

```powershell
$dest = "data/jsonls/temporal_graph_output_v3"
Get-ChildItem -Path $dest -Filter "embeddings_4b-*.zip" |
  Sort-Object Name |
  ForEach-Object { Expand-Archive -Path $_.FullName -DestinationPath $dest -Force }
```

Do NOT use `data/jsonls/p5d_eval_subset.jsonl` (that was a 40-question pilot —
those results live in `output/temporal_eval_v34_p5d3/` for reference).

## Existing Results

Final aggregate is available at:
- **`output/m3_e5_results/MATRIX.json`**

Current finalized matrix coverage:
- 25 completed runs total.
- 15 project runs (Qwen `llm_0.8b`, `llm_2b`, `llm_4b` x 5 modes).
- 10 OpenAI LCEL pure-LLM comparison runs.

Primary conclusions and charts are documented in `docs/RESULTS_M3.md`.

## Output Directory

Results go to `output/m3_e5_results/<run_id>/`:
```
output/m3_e5_results/
  llm_4b__graph_small_emb__emb_0.6b/
    predictions.jsonl
    summary.json
    debug_log.jsonl
  llm_2b__pure_llm/
    ...
  lcel__gpt-5.2/
    ...
  ...
  MATRIX.json          ← aggregated results across all runs
```

## Metrics

Per run:
- `n`: number of questions evaluated
- `exact`: exact string match rate
- `contains`: gold answer contained in prediction rate
- `latency_sec_mean`: mean per-question latency
- Breakdowns: `by_difficulty` (easy/medium/hard), `by_domain`

## Usage

```bash
# Run a single configuration
python experiments/m3_e5_benchmark/run_m3_e5.py \
    --llm-id llm_9b \
    --mode graph_large_emb \
    --emb-id emb_4b \
    --eval-set data/jsonls/temporal_evaluation_set_v2.jsonl \
    --graph-dir data/jsonls/temporal_graph_output_v3 \
    --output-dir output/m3_e5_results

# Run all configurations
python experiments/m3_e5_benchmark/run_m3_e5.py --run-all \
    --eval-set data/jsonls/temporal_evaluation_set_v2.jsonl \
    --graph-dir data/jsonls/temporal_graph_output_v3 \
    --output-dir output/m3_e5_results

# Aggregate completed results into MATRIX.json
python experiments/m3_e5_benchmark/run_m3_e5.py --aggregate \
    --output-dir output/m3_e5_results
```

## Connection to Other Milestones

| Component | Source |
|-----------|--------|
| NLG / Templates | M1 — `src/temporal_nlg/templates/` |
| TMS dependency tracking | M1 — `src/temporal_nlg/tms/` |
| Intent classifier | M2 — intent classification |
| Graph retrieval | M3 — `data/jsonls/temporal_graph_output_v3/` |
| Embedding service | `services/embeddings_server.py` |
| LLM service | `services/llm_server.py` |


