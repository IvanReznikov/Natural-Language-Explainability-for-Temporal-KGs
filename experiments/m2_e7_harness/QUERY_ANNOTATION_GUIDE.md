# M2-E7 Query Annotation Guide

Use this guide to manually enrich `experiments/m2_e7_harness/input/queries.jsonl` with evaluation hints. These hints are read by `run_e2e.py` via `build_expected_map()`.

## Fields you can add per query
- `required_facts`: list of fact IDs that must appear in any rule conclusion within the trace-derived results (default: `["intent"]`).
- `max_latency_ms`: numeric threshold; any rule trace with `latency_ms` above this fails the query (default: `15.0`).
- `intent`: already present; still required for intent matching.
- (Optional) `notes`: free text for your own reference (ignored by the harness).

## Where to annotate
Edit `experiments/m2_e7_harness/input/queries.jsonl`. Each line is an independent JSON object; you can add new keys without reformatting other lines.

## Example annotations (4 lines)
```jsonl
{"query_id": "q0001", "text": "What are the primary symptoms of Type 2 diabetes?", "intent": "medical", "required_facts": ["diagnosis", "symptom"], "max_latency_ms": 20.0}
{"query_id": "q0002", "text": "Explain the concept of compound interest in savings accounts.", "intent": "financial", "required_facts": ["interest_rate", "balance_projection"]}
{"query_id": "q0003", "text": "Describe the events leading to the fall of the Roman Empire.", "intent": "historical", "required_facts": ["event_sequence"]}
{"query_id": "q0004", "text": "How does photosynthesis convert light energy into chemical energy?", "intent": "science", "required_facts": ["chloroplast", "energy_conversion"], "max_latency_ms": 12.0}
```

## Tips
- You can annotate only a subset; unannotated queries will use defaults (`required_facts=["intent"]`, `max_latency_ms=15.0`).
- Keep fact IDs short and consistent with what your traces emit in `conclusion.fact_id`.
- If you don't care about latency for a query, set `"max_latency_ms": null` on that line to disable the check.
- After editing, regenerate traces if you changed query IDs or intents; otherwise existing traces remain compatible.

## Re-run after annotation
```
python experiments/m2_e7_harness/run_e2e.py --queries experiments/m2_e7_harness/input/queries.jsonl --trace experiments/m2_e7_harness/input/trace.jsonl --output experiments/m2_e7_harness/output/results.jsonl --report experiments/m2_e7_harness/output/report.json
```
Review the report (`ok`/`fail` and reasons) and adjust annotations or traces as needed.

