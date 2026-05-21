#!/usr/bin/env python3
"""Run the M2-E7 end-to-end harness over the annotated corpus and traces."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.m2_e7_harness.run_e2e import build_expected_map, build_results_from_traces, eval_results, load_jsonl, save_jsonl


def main():
    root = Path(__file__).resolve().parents[2]
    q_path = root / "experiments" / "m2_e7_harness" / "input" / "queries.jsonl"
    t_path = root / "experiments" / "m2_e7_harness" / "input" / "trace.jsonl"

    queries = load_jsonl(q_path)
    traces = load_jsonl(t_path)

    results = build_results_from_traces(traces)
    expected = build_expected_map(queries)
    ok, total, failures = eval_results(results, expected)

    out_dir = root / "experiments" / "m2_e7_harness" / "output" / "example_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_dir / "results.jsonl", results)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps({"total": total, "ok": ok, "fail": len(failures), "failures": failures}, indent=2))

    sample = results[:3]
    print({
        "queries": len(queries),
        "traces": len(traces),
        "ok": ok,
        "fail": len(failures),
        "report": str(report_path),
        "sample_results": sample,
    })


if __name__ == "__main__":
    main()
