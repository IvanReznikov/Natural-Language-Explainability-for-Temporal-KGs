# M2-E7 End-to-End Harness

Scripts to generate synthetic query corpora and run the end-to-end trigger pipeline.

## Files
- `generate_e2e_queries.py`: make a simple synthetic JSONL query set.
- `run_e2e.py`: ingest trace + queries, run triggers, and emit results + a report.
- `input/trace.jsonl`: placeholder; provide events compatible with TraceRecorder.

## Usage
Generate queries:
```
python experiments/m2_e7_harness/generate_e2e_queries.py --count 20 --output experiments/m2_e7_harness/input/queries.jsonl
```
Run harness:
```
python experiments/m2_e7_harness/run_e2e.py --queries experiments/m2_e7_harness/input/queries.jsonl --trace experiments/m2_e7_harness/input/trace.jsonl --output experiments/m2_e7_harness/output/results.jsonl --report experiments/m2_e7_harness/output/report.json
```

