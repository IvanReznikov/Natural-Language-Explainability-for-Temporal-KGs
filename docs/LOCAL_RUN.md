# Local Reproduction Guide — Milestone 2

This guide walks through reproducing every M2 experiment result from a clean `git clone` on a
CPU-only machine. All commands assume the repository root as the working directory.

## Prerequisites

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"
```

> **Note:** GPU is not required. All models run on CPU. The Flan-T5 parser inference is the
> slowest step (~3 min for 115 examples on a modern CPU).

---

## M2-E2 Intent Classifier (TF-IDF Ablation + Neural)

### Reproduce all 15 TF-IDF ablation runs (T1–T15)

```powershell
.venv\Scripts\python scripts\run_intent_sweep.py
# Outputs: experiments/m2_e2_intent/results/<run_id>/metrics.json  (15 files)
# Runtime: ~5 min CPU
```

Expected best test macro F1: **0.851** (T13/T15)

### Run a single custom configuration

```powershell
.venv\Scripts\python experiments\m2_e2_intent\run_intent_classifier.py `
    --dataset experiments\m2_e2_intent\data\annotated_queries.jsonl `
    --output-dir experiments\m2_e2_intent\results `
    --threshold 0.45 --use-char-ngrams --char-ngram-min 3 --char-ngram-max 5 `
    --char-max-features 20000 --calibration isotonic
```

---

## M2-E3 Parser + Intent Hybrid (M16/M17 — Milestone Target >0.90)

### Model artifacts (download required)

The neural intent and parser models are large and hosted externally. Download them before running inference:

```powershell
.venv\Scripts\python scripts\download_m2_models.py
```

This will download:
| Artifact | Path | Size |
|----------|------|------|
| MiniLM intent encoder | `experiments/m2_e3_parse/artifacts/intent/model.safetensors` | 87 MB |
| Flan-T5-small parser | `experiments/m2_e3_parse/artifacts/parser/model.safetensors` | 294 MB |

### Run inference on the test split (n=115)

```powershell
.venv\Scripts\python experiments\m2_e3_parse\run_parse.py `
    --data experiments\m2_e3_parse\data\splits\test.jsonl `
    --use-model --fallback-on-error `
    --output-dir output\m2_e3_eval\my_run
# Runtime: ~3 min CPU
# Output: output/m2_e3_eval/my_run/<uuid>/preds.jsonl
```

### Evaluate (generates intent and span metrics)

```powershell
.venv\Scripts\python experiments\m2_e3_parse\eval_parse.py `
    --gold experiments\m2_e3_parse\data\splits\test.jsonl `
    --pred output\m2_e3_eval\my_run\<uuid>\preds.jsonl `
    --output output\m2_e3_eval\my_run\<uuid>\metrics.json
```

Expected results (canonical run `a67b735dbbd84f9d8da31483e7b78648`):

| Metric | Value |
|--------|-------|
| Examples | 115 |
| Span F1 | 0.5702 |
| Frame Exact Match | 0.4261 |
| **Intent Micro F1** | **0.9832** |
| **Intent Macro F1** | **0.9703** |

Per-label F1: aggregation 0.973, causal 1.000, comparative 1.000, explanation 0.857,
interval 1.000, overlap 1.000, point_in_time 0.980, prediction 0.923, sequence 1.000.

> This is the M17 production configuration. The >0.90 milestone target is met.

### Re-train models (optional — weights already committed)

```powershell
# Intent encoder
.venv\Scripts\python scripts\m2_e3\train_intent.py `
    --data experiments\m2_e3_parse\data\splits\train.jsonl `
    --val experiments\m2_e3_parse\data\splits\val.jsonl `
    --output-dir experiments\m2_e3_parse\artifacts\intent

# Parser
.venv\Scripts\python scripts\m2_e3\train_parser_t5.py `
    --data experiments\m2_e3_parse\data\splits\train.jsonl `
    --val experiments\m2_e3_parse\data\splits\val.jsonl `
    --output-dir experiments\m2_e3_parse\artifacts\parser
```

---

## M2-E3b Query Construction (AGG, PREDICT, POINT, …)

### Canonical gold-frame run (100% accuracy)

```powershell
.venv\Scripts\python experiments\m2_e3_construct\run_construct.py `
    --data experiments\m2_e3_parse\data\temporal_queries_gold.jsonl `
    --use-gold --output-dir experiments\m2_e3_construct\runs
# Canonical run: 0dc0fd29ef4a49d3b6bb84291813368f
# template_accuracy = 1.000 (1137/1137)
# Coverage: AGG(143) CAUSAL(126) COMPARE(126) INTERVAL(144) OVERLAP(111) POINT(181) PREDICT(90) SEQUENCE(216)
```

---

## M2-E4 Result Taxonomy

```powershell
.venv\Scripts\python experiments\m2_e4_taxonomy\run_taxonomy_classifier.py `
    --data experiments\m2_e4_taxonomy\data\taxonomy.jsonl `
    --output-dir experiments\m2_e4_taxonomy\output\e4a_taxonomy
# Expected: test accuracy 0.993, macro F1 0.994 (LinearSVC word+char E6)
```

---

## M2-E5 Trace & Meta-Query

```powershell
# Generate 1000 synthetic traces
.venv\Scripts\python experiments\m2_e5_trace_meta_query\generate_trace_corpus.py --count 1000

# Interactive meta-query CLI
.venv\Scripts\python experiments\m2_e5_trace_meta_query\meta_query_cli.py
```

---

## M2-E6 Trigger, Query Store, Result Store

```powershell
.venv\Scripts\python experiments\m2_e6_query_store_triggers\e2e_chain_demo.py
# Produces: e2e_queries.jsonl, e2e_results.jsonl with stale marking
```

---

## M2-E7 End-to-End Harness (2,210 queries)

```powershell
# Regenerate corpus (deterministic)
.venv\Scripts\python experiments\m2_e7_harness\generate_e2e_queries.py --count 2210 --seed 42
.venv\Scripts\python experiments\m2_e7_harness\generate_traces.py --seed 1337

# Run harness
.venv\Scripts\python experiments\m2_e7_harness\run_e2e.py `
    --output output\m2_e7_harness\results.jsonl `
    --report output\m2_e7_harness\report.json

# Expected: {"total": 2210, "ok": 2210, "fail": 0, "failures": []}
```

---

## Complete Integrated Flow Demo (NL → TMS Trace)

Demonstrates the full pipeline end-to-end across POINT, CAUSAL, and SEQUENCE query types:

```powershell
.venv\Scripts\python examples\milestone2\m2_e2e_demo.py
```

**Flow:** Natural-language query → MiniLM intent labels + Flan-T5 frame → `build_template_improved()` → canonical query (e.g., `CAUSAL(cause=supply_chain_disruption, effect=inventory_shortage)`) → `QueryTrace` rule firing → `TraceJustifier` explanation chain.

---

## Run All M2 Examples at Once

```powershell
.venv\Scripts\python examples\run_all_m2_examples.py
```

---

## Checklist: Verifying All Reviewer Requirements

| Check | Command | Expected |
|-------|---------|----------|
| T1–T15 metrics exist | `ls experiments\m2_e2_intent\results\*/metrics.json` | 15 files |
| M17 macro F1 ≥ 0.90 | eval_parse.py on test split | `intent_macro_f1 = 0.9703` |
| Model weights present | `ls experiments\m2_e3_parse\artifacts\*/model.safetensors` | 2 files (~87 + 294 MB) |
| AGG/PREDICT/POINT covered | run_construct.py --use-gold | `template_accuracy = 1.000` |
| E7 harness passes | run_e2e.py | `ok=2210, fail=0` |
| E2E demo runs | m2_e2e_demo.py | 3 queries fully traced |
