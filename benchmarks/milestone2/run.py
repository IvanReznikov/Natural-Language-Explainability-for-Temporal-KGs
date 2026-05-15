#!/usr/bin/env python3
"""Run all Milestone 2 performance benchmarks.

Benchmarks:
- Intent classifier inference throughput (TF-IDF + LR)
- TMS trace recorder overhead (trace_bench replicated)
- Parser hybrid inference throughput
"""
from __future__ import annotations
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "benchmarks" / "milestone2"


def bench_tms_trace() -> dict:
    from temporal_nlg.tms.trace import TraceRecorder

    recorder = TraceRecorder()
    N = 500
    t0 = time.perf_counter()
    for i in range(N):
        with recorder.session(f"q{i}", meta={"idx": i}) as trace:
            recorder.record_rule_firing(trace, f"rule_{i % 3}", {"fact_id": f"f{i}"})
    elapsed = time.perf_counter() - t0
    return {"benchmark": "tms_trace_recorder", "n": N, "total_sec": elapsed, "per_item_ms": elapsed / N * 1000}


def main() -> None:
    import json
    from datetime import datetime, timezone

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for fn in [bench_tms_trace]:
        try:
            r = fn()
            print(f"  {r['benchmark']:35s}  {r['per_item_ms']:.3f} ms/item")
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            print(f"  {fn.__name__}: ERROR — {exc}")

    out = OUTPUT_DIR / f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
