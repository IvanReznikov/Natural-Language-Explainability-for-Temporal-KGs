# M2-E3b Query Construction

Scope: build canonical query templates from SRL frames (gold or predicted) produced by M2-E3a.

## Commands (stub)
- Construct from gold SRL: `python run_construct.py --data ../m2_e3_parse/data/temporal_queries_gold.jsonl --output-dir runs --use-gold`
- Construct from predicted SRL: `python run_construct.py --data ../m2_e3_parse/data/temporal_queries_gold.jsonl --pred ../m2_e3_parse/runs/<uuid>/preds.jsonl --output-dir runs`
- Eval: `python eval_construct.py --gold ../m2_e3_parse/data/temporal_queries_gold.jsonl --pred runs/<uuid>/outputs.jsonl`

Outputs land in `runs/<uuid>/` with `outputs.jsonl` and `metrics.json`.
