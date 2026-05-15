# M2-E3a Query Parsing / SRL

Scope: parse temporal queries into structured frames (slots + spans) for downstream construction/optimization.

## Data schema
- id: string unique example id
- text: raw query
- intent_labels: list of high-level intents (e.g., point_in_time, interval, sequence, causal, comparative, aggregation, prediction, explanation)
- spans: list of {label, start, end, text}
- frame: structured slots (free-form dict)
- canonical_query: simple canonical form used later by construct/optimize
- cost_before: mock baseline cost (float)

See schema.json for a formal version.

## Commands (stub)
- Parse: `python run_parse.py --data data/temporal_queries_gold.jsonl --output-dir runs`
- Eval: `python eval_parse.py --gold data/temporal_queries_gold.jsonl --pred runs/<uuid>/preds.jsonl`

Outputs land in `runs/<uuid>/` with `metrics.json` and `preds.jsonl`.
