# scripts folder guide

This folder contains operational scripts and reproducibility utilities.

The package implementation lives in `src/temporal_nlg`. Scripts here are mostly orchestration, data prep, and local runtime helpers.

## Classification legend

- `keep-now`: required by active pipeline/docs or hard runtime dependency
- `refactor-first`: move only after extracting shared logic to package modules
- `optional-candidate`: low coupling utility that can be removed or relocated later

## Top-level scripts

| Script | Class | Why |
|---|---|---|
| `scripts/rebuild_graph_pipeline.ps1` | `keep-now` | Main end-to-end graph rebuild orchestration (processing, build, embeddings). |
| `scripts/precompute_graph_embeddings.py` | `keep-now` | Produces embedding caches used by graph retrieval stack. |
| `scripts/start_docker_servers.ps1` | `keep-now` | Documented quick-start helper for local Qwen servers. |
| `scripts/seed_m3_e5_placeholders.py` | `keep-now` | Documented M3 benchmark scaffold helper. |
| `scripts/generate_results_charts.py` | `optional-candidate` | Report/chart utility; not a core runtime dependency. |
| `scripts/build_qa_index.py` | `optional-candidate` | Benchmark support utility for QA row index artifacts. |
| `scripts/compare_models_temporal_eval.py` | `optional-candidate` | Standalone comparison runner; not imported by core pipelines. |
| `scripts/run_qwen_model_server.py` | `optional-candidate` | Alternate local server implementation; keep only if actively used. |

## M2-E3 subfolder (`scripts/m2_e3`)

| Script | Class | Why |
|---|---|---|
| `scripts/m2_e3/consistency.py` | `keep-now` | Hard runtime import from `experiments/m2_e3_parse/run_parse.py`. |
| `scripts/m2_e3/data_prep.py` | `keep-now` | Reproducibility/data split script for M2-E3 training flow. |
| `scripts/m2_e3/train_intent.py` | `keep-now` | Reproducibility training script (intent model). |
| `scripts/m2_e3/train_parser_t5.py` | `keep-now` | Reproducibility training script (parser model). |
| `scripts/m2_e3/infer_intent.py` | `optional-candidate` | Utility inference CLI; not required by core runtime. |
| `scripts/m2_e3/infer_parser_t5.py` | `optional-candidate` | Utility inference CLI; not required by core runtime. |

## Recommended migration order

1. Move only `optional-candidate` scripts first.
2. Keep path compatibility for one release cycle using thin wrappers or docs redirects.
3. For `refactor-first` or dependency-bound scripts, extract reusable logic into `src/temporal_nlg` first, then retain slim CLI wrappers in `scripts/`.

## Important dependency to preserve

`experiments/m2_e3_parse/run_parse.py` imports:

`from scripts.m2_e3.consistency import validate`

Do not move `scripts/m2_e3/consistency.py` unless this import is migrated.

Archived helper scripts are intentionally omitted from the active scripts table above.