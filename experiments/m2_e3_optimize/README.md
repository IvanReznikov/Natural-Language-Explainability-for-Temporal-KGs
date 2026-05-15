# M2-E3c Query Optimization

Scope: apply simple rewrite/optimization rules to canonical queries and track improvement against a mock cost model.

## Commands (stub)
- Optimize gold canonical queries: `python run_optimize.py --data ../m2_e3_parse/data/temporal_queries_gold.jsonl --output-dir runs`
- Optimize constructed outputs: `python run_optimize.py --pred ../m2_e3_construct/runs/<uuid>/outputs.jsonl --data ../m2_e3_parse/data/temporal_queries_gold.jsonl --output-dir runs`
- Eval: `python eval_optimize.py --gold ../m2_e3_parse/data/temporal_queries_gold.jsonl --pred runs/<uuid>/optimized.jsonl`

Outputs land in `runs/<uuid>/` with `optimized.jsonl` and `metrics.json`.
