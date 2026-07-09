# M3-E4b: Consistency Under Fact Revision — Output Directory

## Status: Benchmark Tooling Deliverable

This directory contains the **benchmark dataset and evaluation tooling** for the
consistency-under-revision experiment, delivered per the Milestone 3 proposal
(deliverables: benchmark datasets, evaluation metrics, documentation and tools).

The results template (`m3_e4b_results_template.csv`) is an **unfilled participant template**
— it is the input form to be completed by evaluators or an automated scoring pipeline.

---

## Contents

| File | Description |
|---|---|
| `m3_e4b_facts.jsonl` | 1,000 temporal facts with base explanations |
| `m3_e4b_revisions.jsonl` | 4,000 revision scenarios (4 types × 1,000 facts): `date_correction`, `add_causal`, `contradiction`, `removal` |
| `m3_e4b_results_template.csv` | Participant/evaluator fill-in CSV (unfilled — see below) |
| `llm_simulation_prompt.txt` | Ready-to-use prompt for automated LLM scoring of the revision pairs |

---

## How to Fill the Results Template

### Option A — LLM Simulation

Paste `llm_simulation_prompt.txt` into an LLM (e.g. GPT-4o) and save the structured
output back into a filled CSV matching the template schema.

### Option B — Human Evaluators

Distribute `m3_e4b_results_template.csv` to evaluators. They fill in:
- `method` — which explanation method was evaluated
- `update_accuracy` — 0 or 1 (was the explanation correctly updated?)
- `contradiction_detected` — 0 or 1 (was the contradiction identified?)
- `coherence_rating_1_5` — integer 1–5
- `resolution_time_sec` — time in seconds to resolve the revision

---

## How to Analyze Once Filled

```bash
python experiments/m3_e4_efficiency_consistency/run_consistency.py analyze \
    --results output/m3_e4b_consistency/m3_e4b_results_template.csv \
    --output-dir output/m3_e4b_consistency_analysis
```

This writes `output/m3_e4b_consistency_analysis/m3_e4b_consistency.summary.json`.

---

## How the Dataset Was Generated

```bash
python experiments/m3_e4_efficiency_consistency/run_consistency.py export \
    --dataset data/jsonls/temporal_graph.jsonl \
    --output-dir output/m3_e4b_consistency \
    --n-facts 1000 \
    --seed 13
```
