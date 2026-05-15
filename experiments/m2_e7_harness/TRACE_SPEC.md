# M2-E7 Trace File Specification

Use this to hand-craft the JSONL traces consumed by the E2E harness at `experiments/m2_e7_harness/input/trace.jsonl`.

## Purpose
Each line represents a `QueryTrace` (the structure produced by `TraceRecorder` in `src/temporal_nlg/tms/trace.py`). It is query-related: `query_id` must correspond to a query in `queries.jsonl`. Rule firings (`rule_traces`) describe how the system derived conclusions for that query.

## Format
- UTF-8 JSONL (one JSON object per line).
- One line per query.
- Floating-point timestamps are UNIX seconds.

## Required fields per `QueryTrace`
- `query_id`: matches a query in the corpus (e.g., `q0001`).
- `started_at`: float timestamp when processing began.
- `completed_at`: float timestamp when processing finished.
- `instrumentation_overhead_ms`: total tracing overhead in milliseconds (set to `0.0` if unknown).
- `dropped`: `false` unless sampling skipped the trace.
- `over_budget`: `false` unless overhead exceeded your budget.
- `meta`: arbitrary dict (session/user/context tags). Use `{}` if nothing to add.
- `rule_traces`: list of rule firing objects (see below). Can be empty if nothing fired.

## Rule trace objects (inside `rule_traces`)
- `rule_id`: stable identifier for the rule (string).
- `rule_name`: human-friendly label.
- `inputs`: list of fact dicts consumed by the rule. Each fact typically has:
  - `fact_id`: unique fact key (string).
  - `value`: any JSON-serializable value (string/number/object).
- `conclusion`: fact dict produced by the rule (should include `fact_id`, `value`).
- `fired_at`: float timestamp when the rule fired.
- `confidence`: numeric weight (default `1.0`).
- `latency_ms`: optional numeric latency for the rule (ms) or `null`.
- `meta`: optional dict for rule-level details; `{}` if none.

## Minimal completeness
- Provide at least one `rule_trace` per query to exercise justification and trigger paths.
- Keep IDs stable so result correlation remains deterministic across runs.

## Example JSONL (3 queries)
```jsonl
{"query_id": "q0001", "started_at": 1736275200.0, "completed_at": 1736275201.2, "instrumentation_overhead_ms": 1.4, "dropped": false, "over_budget": false, "meta": {"session": "s1"}, "rule_traces": [{"rule_id": "r_med_1", "rule_name": "detect_side_effects", "inputs": [{"fact_id": "drug", "value": "antihistamine"}], "conclusion": {"fact_id": "side_effects", "value": ["drowsiness", "dry_mouth"]}, "fired_at": 1736275200.5, "confidence": 0.92, "latency_ms": 8.0, "meta": {"source": "kb"}}]}
{"query_id": "q0002", "started_at": 1736275202.0, "completed_at": 1736275203.1, "instrumentation_overhead_ms": 0.9, "dropped": false, "over_budget": false, "meta": {"session": "s1"}, "rule_traces": [{"rule_id": "r_fin_1", "rule_name": "calc_interest", "inputs": [{"fact_id": "principal", "value": 10000}, {"fact_id": "rate", "value": 0.04}], "conclusion": {"fact_id": "balance_year5", "value": 12166.53}, "fired_at": 1736275202.6, "confidence": 0.95, "latency_ms": 5.1, "meta": {"method": "compound"}}]}
{"query_id": "q0003", "started_at": 1736275204.0, "completed_at": 1736275205.0, "instrumentation_overhead_ms": 1.1, "dropped": false, "over_budget": false, "meta": {"session": "s1"}, "rule_traces": [{"rule_id": "r_hist_1", "rule_name": "identify_causes", "inputs": [{"fact_id": "event", "value": "1848_revolutions"}], "conclusion": {"fact_id": "causes", "value": ["economic_crisis", "liberal_reform"]}, "fired_at": 1736275204.4, "confidence": 0.9, "latency_ms": 6.7, "meta": {"source": "notes"}}]}
```

## Tips
- Keep timestamps increasing but they need not be real; consistency matters more than wall-clock accuracy.
- If you need multiple rule firings per query, append more objects to `rule_traces` in that line.
- If sampling drops a trace, set `dropped` to `true` and leave `rule_traces` empty.
- If you add new fact shapes, ensure downstream consumers are tolerant (they currently treat facts as opaque dicts with `fact_id`).

