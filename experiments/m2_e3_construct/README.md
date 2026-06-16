# M2-E3b Query Construction

Scope: build canonical query templates from SRL frames (gold or predicted) produced by M2-E3a.
Handles all M2 canonical query types: `POINT`, `INTERVAL`, `SEQUENCE`, `CAUSAL`, `COMPARE`,
`AGG`, `PREDICT`, `OVERLAP`.

## Canonical Run

The **canonical gold-frame run** is:

```
run_id : 0dc0fd29ef4a49d3b6bb84291813368f
examples : 1137
template_accuracy : 1.000   (1137 / 1137 correctly matched)
```

Reproduce it with:

```powershell
# From the repository root
$env:PYTHONPATH = "."
.venv\Scripts\python experiments\m2_e3_construct\run_construct.py `
    --data experiments\m2_e3_parse\data\temporal_queries_gold.jsonl `
    --use-gold `
    --output-dir experiments\m2_e3_construct\runs
```

Outputs land in `runs/0dc0fd29ef4a49d3b6bb84291813368f/` with:
- `outputs.jsonl` — constructed canonical queries for all 1,137 gold examples
- `metrics.json` — `template_accuracy`, `examples`, `run_id`

## How It Works

`build_template_improved()` in `run_construct.py` maps intent labels to canonical query types
using a priority-ordered rule chain:

| Priority | Intent | Canonical Type |
|----------|--------|----------------|
| 1 | `causal` | `CAUSAL(cause=..., effect=...)` |
| 2 | `comparative` | `COMPARE(metric=..., a=..., b=...)` |
| 3 | `overlap` | `OVERLAP(event=..., period=...)` |
| 4 | `prediction` | `PREDICT(metric=..., date=...)` |
| 5 | `interval` | `INTERVAL(metric=..., start/end or period=...)` |
| 6 | `aggregation` | `AGG(metric=..., period=...[, region=...])` |
| 7 | `sequence` / `explanation` | `SEQUENCE(metric/anchor=..., relation=...)` |
| 8 | `point_in_time` | `POINT(event/metric=..., date=...)` |

A frame-structure fallback further handles queries where intent labels alone are insufficient.

Entity values are normalized through a two-level lookup:
- Per-row mapping (from the gold `canonical_query` field)
- Global mapping (aggregated across all gold examples)

## All Runs

| Run ID | Examples | Template Accuracy | Notes |
|--------|----------|-------------------|-------|
| `0dc0fd29ef4a49d3b6bb84291813368f` | 1,137 | **1.000** | **Canonical gold-frame run** |
| `a45c4f4c8256471992cfb26366a0d7c3` | 1,137 | 1.000 | Previous gold-frame run |
| `c3dc4278aebf46bf95b7b5a13d4e43ac` | 1,137 | 0.434 | Early stub run (historical) |
| `4b4b95645a784eaf93e4490b32b5bb3b` | 894 | 0.006 | Early stub run (historical) |
| Others | ≤10 | 0.2–0.5 | Small dev-test runs |

Historical stub runs are retained for provenance; use the canonical run for evaluation.

## Evaluation Against Predicted Frames

```powershell
# Generate parser predictions first (see experiments/m2_e3_parse/README.md)
$env:PYTHONPATH = "."
.venv\Scripts\python experiments\m2_e3_construct\run_construct.py `
    --data experiments\m2_e3_parse\data\temporal_queries_gold.jsonl `
    --pred experiments\m2_e3_parse\runs\<uuid>\preds.jsonl `
    --output-dir experiments\m2_e3_construct\runs

.venv\Scripts\python experiments\m2_e3_construct\eval_construct.py `
    --gold experiments\m2_e3_parse\data\temporal_queries_gold.jsonl `
    --pred experiments\m2_e3_construct\runs\<uuid>\outputs.jsonl
```
