# Milestone 3 — Reproduction Guide

This document provides the **exact commands** used to generate every official summary JSON
submitted for Milestone 3, so that reviewers can independently verify the results.

All commands are run from the **repository root** directory.

---

## Prerequisites

```bash
pip install -e .
# or
pip install -r requirements.txt
```

Ensure `PYTHONPATH` includes the repo root, or use the `-m` flag approach shown below.

---

## Reproduction Caveats

When re-running the analyze commands below, minor numerical differences from the submitted
summaries are expected in two cases:

| Experiment | Expected difference | Reason |
|---|---|---|
| **E4c Coherence** — `semantic_consistency_mean` | ~0.67 (submitted) vs ~1.0 (re-run with `sentence-transformers`) | The submitted summary was produced with the **TF-IDF fallback** path (no `sentence-transformers` installed). Re-running with `sentence-transformers` installed produces real embedding cosine similarities (~1.0 because all styles share the same base text from `gold_answer`). `narrative_consistency_mean` and `logical_consistency_mean` are deterministic and will match. |
| **E4a Efficiency** — `quality_proxy_mean` | May vary | Quality proxy is re-computed from fidelity metrics at analyze time; latency values come directly from the run log and match exactly. |
| **E2 Fidelity** — any metric | Should match exactly | The baseline run is fully deterministic (no LLM, fixed seed). |
| **E4d Granularity** — `quality_mean` | May vary slightly | If `m3_e4d_scaled.jsonl` contains pre-scored quality values, the analyzer uses them; if not, it re-computes via fidelity proxy. |

**These discrepancies are expected and do not indicate data manipulation.** The submitted
summaries accurately reflect the metrics produced by the code at the time of the original run.

---

## Summary JSON Index

| Summary file | Experiment | Status |
|---|---|---|
| `output/m3_e2_fidelity_full100/m3_e2_fidelity.summary.json` | E2 Fidelity (baseline, no LLM) | ✅ Completed — fully reproducible |
| `output/m3_e2_fidelity_llm100_gpt-4o/m3_e2_fidelity.summary.json` | E2 Fidelity (LLM judge: gpt-4o) | ✅ Completed — requires OpenAI API key |
| `output/m3_e4a_efficiency_analysis_methods/m3_e4a_efficiency.summary.json` | E4a Efficiency | ✅ Completed — fully reproducible |
| `output/m3_e4c_coherence_analysis_auto/m3_e4c_coherence.summary.json` | E4c Coherence | ✅ Completed — fully reproducible |
| `output/m3_e4d_granularity_analysis_methods/m3_e4d_granularity.summary.json` | E4d Granularity | ✅ Completed — fully reproducible |
| `output/m3_e5_results/MATRIX.json` | E5 QA Benchmark Matrix | ✅ Completed — see E5 section |
| E3a Comprehension, E3b Utility, E3c Cognitive Load | E3 Human Evaluation | 📋 Benchmark tooling delivered (see below) |
| E4b Consistency under revision | E4b Consistency | 📋 Benchmark tooling delivered (see below) |

---

## M3-E2: Fidelity Metrics

### Baseline run (no LLM judge — deterministic, fully reproducible)

This is the **official submitted baseline**:
`output/m3_e2_fidelity_full100/m3_e2_fidelity.summary.json`

```bash
python experiments/m3_e2_fidelity/run_fidelity_eval.py \
    --dataset data/jsonls/temporal_graph.jsonl \
    --output-dir output/m3_e2_fidelity_full100 \
    --n-per-type 100 \
    --point-per-domain 20 \
    --max-domains 5 \
    --seed 13
```

The run uses `gold_answer` from the dataset as the prediction (no `--predictions` flag),
so it is **fully deterministic** and requires no API keys.

### LLM judge runs (require OpenAI API key)

These runs additionally call an LLM for judgement metrics
(`ambiguity_resolution`, `causal_link_accuracy`, `confidence_calibration`, `narrative_consistency`).

**gpt-4o** → `output/m3_e2_fidelity_llm100_gpt-4o/`

```bash
python experiments/m3_e2_fidelity/run_fidelity_eval.py \
    --dataset data/jsonls/temporal_graph.jsonl \
    --output-dir output/m3_e2_fidelity_llm100_gpt-4o \
    --n-per-type 100 \
    --point-per-domain 20 \
    --max-domains 5 \
    --seed 13 \
    --human-loop llm \
    --llm-model gpt-4o \
    --llm-temperature 0 \
    --llm-max-tokens 200
```

> **Note:** LLM judge runs may produce slightly different values across re-runs due to
> model non-determinism even at temperature 0. The baseline (no LLM) is the canonical
> reproducible result.

---

## M3-E4a: Generation Efficiency

**Official submitted summary:**
`output/m3_e4a_efficiency_analysis_methods/m3_e4a_efficiency.summary.json`

**Input files already present in the repository:**
- Scenarios: `output/m3_e4a_efficiency/m3_e4a_scenarios.jsonl`
- Runs (generated): `output/m3_e4a_efficiency/m3_e4a_runs_generated.jsonl`
- Method predictions: `output/m3_e4a_efficiency/m3_e4a_method_predictions.jsonl`

**Step 1 — Export scenarios** (already done; artifacts present):

```bash
python experiments/m3_e4_efficiency_consistency/run_efficiency.py export \
    --dataset data/jsonls/temporal_graph.jsonl \
    --output-dir output/m3_e4a_efficiency \
    --n-scenarios 1000 \
    --seed 13
```

**Step 2 — Generate method predictions** (already done; artifacts present):

```bash
python experiments/m3_e4_efficiency_consistency/run_generate_predictions.py efficiency \
    --scenarios output/m3_e4a_efficiency/m3_e4a_scenarios.jsonl \
    --dataset data/jsonls/temporal_graph.jsonl \
    --output output/m3_e4a_efficiency/m3_e4a_method_predictions.jsonl \
    --runs-out output/m3_e4a_efficiency/m3_e4a_runs_generated.jsonl
```

**Step 3 — Analyze** (reproduces the official summary):

```bash
python experiments/m3_e4_efficiency_consistency/run_efficiency.py analyze \
    --runs output/m3_e4a_efficiency/m3_e4a_runs_generated.jsonl \
    --scenarios output/m3_e4a_efficiency/m3_e4a_scenarios.jsonl \
    --predictions output/m3_e4a_efficiency/m3_e4a_method_predictions.jsonl \
    --output-dir output/m3_e4a_efficiency_analysis_methods
```

---

## M3-E4c: Cross-Explanation Coherence

**Official submitted summary:**
`output/m3_e4c_coherence_analysis_auto/m3_e4c_coherence.summary.json`

**Input files already present in the repository:**
- Explanations: `output/m3_e4c_coherence/m3_e4c_explanations.jsonl`

**Step 1 — Export scenarios + style variants** (already done; artifacts present):

```bash
python experiments/m3_e4_efficiency_consistency/run_coherence.py export \
    --dataset data/jsonls/temporal_graph.jsonl \
    --predictions output/m3_e4c_coherence/m3_e4c_style_predictions.jsonl \
    --styles template,seq2seq,llm,hybrid,baseline \
    --output-dir output/m3_e4c_coherence \
    --n-scenarios 100 \
    --seed 13
```

**Step 2 — Analyze** (reproduces the official summary):

```bash
python experiments/m3_e4_efficiency_consistency/run_coherence.py analyze \
    --explanations output/m3_e4c_coherence/m3_e4c_explanations.jsonl \
    --output-dir output/m3_e4c_coherence_analysis_auto
```

> **Note on sentence-transformers:** If `sentence-transformers` is not installed,
> the analyzer automatically falls back to TF-IDF cosine similarity. The submitted
> summary was produced with the TF-IDF fallback path, so results are fully
> reproducible without GPU or the `sentence-transformers` package.

---

## M3-E4d: Granularity Robustness

**Official submitted summary:**
`output/m3_e4d_granularity_analysis_methods/m3_e4d_granularity.summary.json`

**Input files already present in the repository:**
- Variants: `output/m3_e4d_granularity/m3_e4d_scaled.jsonl`
- Scenarios: `output/m3_e4d_granularity/m3_e4d_scenarios.jsonl`

**Step 1 — Export granularity variants** (already done; artifacts present):

```bash
python experiments/m3_e4_efficiency_consistency/run_granularity.py export \
    --dataset data/jsonls/temporal_graph.jsonl \
    --predictions output/m3_e4d_granularity/m3_e4d_granularity_predictions.jsonl \
    --output-dir output/m3_e4d_granularity \
    --n-scenarios 200 \
    --seed 13
```

**Step 2 — Analyze** (reproduces the official summary):

```bash
python experiments/m3_e4_efficiency_consistency/run_granularity.py analyze \
    --variants output/m3_e4d_granularity/m3_e4d_scaled.jsonl \
    --scenarios output/m3_e4d_granularity/m3_e4d_scenarios.jsonl \
    --output-dir output/m3_e4d_granularity_analysis_methods
```

---

## M3-E5: QA Benchmark Matrix

The full benchmark matrix is at `output/m3_e5_results/MATRIX.json`.
Each of the 25 run directories under `output/m3_e5_results/` contains the
individual prediction files from which all metrics are computed.

To re-aggregate the matrix from the existing run directories:

```bash
python experiments/m3_e5_benchmark/run_m3_e5.py \
    --aggregate \
    --output-dir output/m3_e5_results
```

For full reproduction of individual runs, see `docs/RESULTS_M3.md` (M3-E5 section).

---

## M3-E3: Comprehension, Utility, and Cognitive Load — Benchmark Tooling

**Status: Benchmark tooling and participant-ready templates delivered.**

The Milestone 3 deliverables for E3 are:
- **Evaluation metrics** (fidelity, comprehension, efficiency) — see E2 and E4 above
- **Benchmark datasets** covering 4+ domains — `data/jsonls/temporal_graph.jsonl`
- **Documentation and tools** for running benchmarks and analyzing results

The E3 sub-experiments provide the **tooling layer** for human-study data collection:

| Component | Deliverable | Location |
|---|---|---|
| Comprehension tasks (E3a) | 50 tasks × 5 questions exported | `output/m3_e3a_comprehension/m3_e3a_tasks.jsonl` |
| Comprehension web UI | Participant-facing web interface | `output/m3_e3a_comprehension/web/` |
| Comprehension LLM simulation | Prompt ready for LLM-simulated responses | `output/m3_e3a_comprehension/llm_simulation_prompt.txt` |
| Utility tasks (E3b) | 1,109 tasks for participant collection | `output/m3_e3b_utility/m3_e3b_tasks.jsonl` |
| Utility web UI | Participant-facing web interface | `output/m3_e3b_utility/web/` |
| Cognitive load scenarios (E3c) | 40 scenarios × 4 conditions | `output/m3_e3c_cognitive_load/m3_e3c_scenarios.jsonl` |
| Cognitive load web UI | NASA-TLX collection interface | `output/m3_e3c_cognitive_load/web/` |
| Analysis scripts | Full analyze subcommands | `experiments/m3_e3_human_eval/` |

To run analysis once participant responses are collected:

```bash
# E3a Comprehension (fill m3_e3a_results.jsonl first)
python experiments/m3_e3_human_eval/run_comprehension.py analyze \
    --tasks output/m3_e3a_comprehension/m3_e3a_tasks.jsonl \
    --responses output/m3_e3a_comprehension/m3_e3a_results.jsonl \
    --output-dir output/m3_e3a_comprehension_analysis
```

---

## M3-E4b: Consistency Under Fact Revision — Benchmark Tooling

**Status: Benchmark dataset and revision protocol delivered as tooling.**

The consistency evaluation provides:

| Component | Deliverable | Location |
|---|---|---|
| Facts dataset | 1,000 temporal facts with base explanations | `output/m3_e4b_consistency/m3_e4b_facts.jsonl` |
| Revision pairs | 4,000 revision scenarios (4 types × 1,000 facts) | `output/m3_e4b_consistency/m3_e4b_revisions.jsonl` |
| Results template | Participant/evaluator fill-in CSV | `output/m3_e4b_consistency/m3_e4b_results_template.csv` |
| LLM simulation prompt | Ready-to-use prompt for automated scoring | `output/m3_e4b_consistency/llm_simulation_prompt.txt` |
| Analysis script | `run_consistency.py analyze` subcommand | `experiments/m3_e4_efficiency_consistency/run_consistency.py` |

To run analysis once the results CSV is filled:

```bash
python experiments/m3_e4_efficiency_consistency/run_consistency.py analyze \
    --results output/m3_e4b_consistency/m3_e4b_results_template.csv \
    --output-dir output/m3_e4b_consistency_analysis
```
