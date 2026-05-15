# M2-E2: Query Intent Classification

This experiment implements the M2-E2 plan from `ideas/temporal-experiments.md` and `ideas/temporal_experiments_summary.csv`:
- Build a clear intent taxonomy for temporal queries.
- Train an intent classifier with macro F1 goals >0.90 on single intents.
- Detect multi-intent queries (combinations such as Interval+Cause+Explanation).

## Taxonomy
Eight intents are supported (expandable via config):
1. point_in_time
2. interval
3. sequence
4. causal
5. comparative
6. aggregation
7. prediction
8. explanation

## Data format
Input is JSONL with fields:
- `id`: unique query id
- `query`: natural language temporal query
- `intents`: list of intent strings (single or multiple)

A starter set lives at `data/annotated_queries.jsonl` (24 examples spanning single and multi-intent cases). Replace/extend with your annotated corpus (target: 5k single-intent + 500 multi-intent per plan).

## Quickstart
1) Install deps (from repo root):
```
pip install -r requirements.txt
```

2) Run the classifier baseline:
```
python experiments/m2_e2_intent/run_intent_classifier.py \
  --dataset experiments/m2_e2_intent/data/annotated_queries.jsonl \
  --output-dir experiments/m2_e2_intent/results \
  --threshold 0.35
```

Outputs:
- `metrics.json`: macro/micro F1, per-intent precision/recall/F1, subset accuracy for multi-intent.
- `predictions.jsonl`: model predictions with probabilities.

## Notes
- Default model: TF-IDF + One-vs-Rest Logistic Regression (lightweight, reproducible). Swap in HF models later if desired.
- Threshold controls multi-intent activation; rows with all scores below threshold fall back to the argmax intent to avoid empty predictions.
- Keep taxonomy strings consistent; the script normalizes by lowercasing and replacing spaces with underscores.
