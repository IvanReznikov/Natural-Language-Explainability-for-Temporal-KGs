# Data Catalog

This folder contains the shared M3 graph artifacts and evaluation set used by the benchmarked retrieval tasks.

---

## Milestone Dataset Map

| Milestone | Datasets used | Primary purpose |
|-----------|----------------|-----------------|
| M1 | Custom temporal NLG corpora and evaluation artifacts produced in the M1 experiments | Template, LLM, and hybrid generation/evaluation plus integration benchmarks. |
| M2 | Experiment-local annotation corpora under `experiments/m2_*/data/` | Query understanding, parser/taxonomy training, trace/trigger harness inputs. |
| M3 | `jsonls/temporal_graph.jsonl`, `jsonls/temporal_graph_output_v3/`, `jsonls/temporal_evaluation_set_v2.jsonl` | Graph retrieval, fidelity/comprehension/quality studies, and the M3-E5 benchmark matrix. |

Notes:
- M1 and M2 rely on experiment-local corpora and outputs under `experiments/**/` and `output/**/` rather than a shared project-wide dataset.
- M3 benchmark runs must use `jsonls/temporal_evaluation_set_v2.jsonl`.

---

## Primary Datasets in This Folder

| Path | Description |
|------|-------------|
| `jsonls/temporal_graph.jsonl` | 51,232 human-curated temporal facts across 35+ domains. |
| `jsonls/temporal_graph_output_v3/` | Precomputed graph artifacts (`nodes.jsonl`, `edges.jsonl`, `tags.jsonl`, `processed_graph.jsonl`, report files). |
| `jsonls/temporal_evaluation_set_v2.jsonl` | Official M3 QA set (295 questions). Stable and versioned. |
| `jsonls/aggregated.jsonl` | Aggregated domain-level temporal facts (intermediate preprocessing artifact). |

---

## M1 Output Locations

M1 result copies are centralized under `output/m1_*/results/`, for example:
- `output/m1_e1_templates/results/*/m1_e1_report.json`
- `output/m1_e1_templates/results/*/m1_e1_report.txt`

---

## Domain Subfolders

| Folder | Description |
|--------|-------------|
| `jsonls/business/` | Business domain source facts |
| `jsonls/culture/` | Cultural domain source facts |
| `jsonls/geopolitics/` | Geopolitical domain source facts |
| `jsonls/history/` | Historical domain source facts |
| `jsonls/science_tech/` | Science and technology domain source facts |
| `jsonls/processed/` | Intermediate processed outputs |

---


