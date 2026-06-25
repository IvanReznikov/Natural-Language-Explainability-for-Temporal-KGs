# M2-E3a Query Parsing / Semantic Role Labelling

Scope: parse temporal natural-language queries into structured frames (slots + spans) for downstream construction and optimization.

---

## Data Schema

See [`schema.json`](schema.json) for a formal JSON Schema definition of every field.

### Top-Level Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | ✓ | Unique example ID (e.g. `q001`) |
| `text` | string | ✓ | Raw natural-language query |
| `intent_labels` | list[string] | ✓ | Ordered list of intent labels; first element is the primary intent |
| `spans` | list[dict] | | Token-level spans: `{label, start, end, text}` |
| `frame` | dict | | Slot-value pairs; valid keys depend on primary intent (see below) |
| `canonical_query` | string | | Downstream canonical form, e.g. `POINT(event='…', date='…')` |

---

## Intent Taxonomy & Frame Schema

### `point_in_time`
Asks *when* a named event occurred.

**Key clarification**: the `time` / `date` slot is the **answer to be retrieved** from the knowledge graph — it is **not** extracted from the query text. The query supplies the event name; the KB returns the date at inference time. Gold labels include concrete dates; model predictions are expected to leave `time` empty (or hallucinate from parametric memory, which is the dominant source of `time` field errors).

```
Frame keys:  event (required), time (optional, KB-populated)
Canonical:   POINT(event='treaty_of_rome_signing', date='1957-03-25')
```

---

### `prediction`
Asks for a forecast of a metric at a future date or period.

**Key clarification**: both `date` (bare year: `"2029"`) and `period` (quarter: `"2028-Q2"`) are valid. Use `date` when the query contains a bare year; use `period` when the query explicitly mentions a quarter. Both are never simultaneously required.

```
Frame keys:  metric (required), date (optional), period (optional)
Canonical:   PREDICT(metric='global_gaming_revenue', date='2027')
```

---

### `explanation`
**`explanation` is NOT a standalone primary intent.** It is always a modifier co-occurring with `sequence`:

```
intent_labels: ["explanation", "sequence"]
```

It signals that the question asks for a *causal explanation of a temporal sequence* rather than merely a sequencing fact. The frame schema is identical to `sequence`. Downstream, `explanation+sequence` queries produce `SEQUENCE(…)` canonical queries.

Do **not** model `explanation` as an independent frame type or canonical query type.

---

### `sequence`
Asks what happened before/after a named anchor event, or what chain of events surrounds it.

```
Frame keys:  metric (optional), anchor_event or anchor (required), relation (required: before/after/followed)
Canonical:   SEQUENCE(metric='latency', anchor='patch', relation='after')
```

---

### `causal`
Asks whether a specific cause produced a specific effect. Both must be extractable from the query text.

```
Frame keys:  cause (required), effect (required)
Canonical:   CAUSAL(cause='network_partition', effect='service_outage')
```

---

### `interval`
Asks for a metric's value over a continuous date range.

```
Frame keys:  metric (required), start (optional), end (optional), period (optional)
Canonical:   INTERVAL(metric='bounce_rate', start='2020', end='2022')
```

---

### `aggregation`
Asks for a summary statistic (total, average, count) over a period, optionally filtered by region.

```
Frame keys:  metric (required), period (required), region (optional)
Canonical:   AGG(metric='revenue', period='2023-Q3', region='north_america')
```

---

### `comparative`
Asks for a comparison of a metric between two time points or cohorts.

```
Frame keys:  metric (required), a (required), b (required)
Canonical:   COMPARE(metric='bounce_rate', a='2022', b='2023')
```

---

### `overlap`
Asks for concurrent/overlapping events within a time window.

```
Frame keys:  event (required), period (required)
Canonical:   OVERLAP(event='maintenance_windows', period='2024-Q3')
```

---

## Commands

### Parse (rule-based, no model needed)
```powershell
python experiments/m2_e3_parse/run_parse.py `
    --data experiments/m2_e3_parse/data/splits_qwen/test_prompts.jsonl `
    --mode rules
```

### Parse (Qwen LoRA model + fallback)
```powershell
python experiments/m2_e3_parse/run_parse.py `
    --data experiments/m2_e3_parse/data/splits_qwen/test_prompts.jsonl `
    --mode qwen+fallback `
    --qwen-adapter-dir experiments/m2_e3_parse/artifacts/qwen_parser_lora
```

### Evaluate pipeline (rules vs qwen vs qwen+fallback)
```powershell
python experiments/m2_e3_parse/eval_pipeline.py `
    --test-data  experiments/m2_e3_parse/data/splits_qwen/test_prompts.jsonl `
    --gold-data  experiments/m2_e3_parse/data/temporal_queries_merged.jsonl `
    --adapter-dir experiments/m2_e3_parse/artifacts/qwen_parser_lora
```

### Evaluate end-to-end canonical query accuracy from predicted frames
```powershell
python scripts/m2_e3/eval_construct_from_preds.py `
    --preds experiments/m2_e3_parse/runs/eval_pipeline/preds_rules.jsonl `
    --gold  experiments/m2_e3_parse/data/temporal_queries_merged.jsonl
```

### Evaluate parse quality only
```powershell
python experiments/m2_e3_parse/eval_parse.py `
    --gold experiments/m2_e3_parse/data/splits/test.jsonl `
    --pred experiments/m2_e3_parse/runs/<uuid>/preds.jsonl
```

---

## Outputs

All outputs land in `runs/<uuid>/` (or `runs/eval_pipeline/`) with:

| File | Contents |
|---|---|
| `preds.jsonl` / `preds_<mode>.jsonl` | Predicted rows (intent_labels, spans, frame, source) |
| `metrics.json` / `eval_summary.json` | Intent accuracy, frame F1, span F1, fallback rate |
| `justifications.jsonl` | Per-example parsing justification (rule path or model confidence) |
