#!/usr/bin/env python3
"""Micro-benchmark for TraceRecorder overhead (M2-E5).

Measures instrumentation overhead per recorded rule firing and the overall
trace size for synthetic workloads. Intended to keep us under the target
<5ms overhead and <1MB/trace budget.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path
from typing import List

from temporal_nlg.tms.trace import TraceRecorder


def run_benchmark(num_traces: int, rules_per_trace: int, seed: int = 1234):
    rng = random.Random(seed)
    recorder = TraceRecorder(sampling_rate=1.0, max_overhead_ms=10.0)

    overheads: List[float] = []
    trace_sizes: List[int] = []
    start = time.perf_counter()

    for idx in range(num_traces):
        qt = recorder.start_query(f"bench_{idx}")
        for r in range(rules_per_trace):
            recorder.record_rule_firing(
                qt,
                rule_id=f"rule_{r}",
                rule_name=f"rule_{r}",
                inputs=[{"fact_id": f"f_{idx}_{r}", "value": rng.randint(1, 100)}],
                conclusion={"fact_id": f"g_{idx}_{r}", "value": rng.randint(1, 100)},
                confidence=1.0,
                latency_ms=rng.uniform(0.05, 2.0),
            )
        recorder.complete_query(qt)
        overheads.append(qt.instrumentation_overhead_ms)
        trace_sizes.append(len(recorder.to_json(qt).encode("utf-8")))

    wall_ms = (time.perf_counter() - start) * 1000.0
    return {
        "num_traces": num_traces,
        "rules_per_trace": rules_per_trace,
        "mean_overhead_ms": statistics.mean(overheads),
        "p95_overhead_ms": percentile(overheads, 95),
        "max_overhead_ms": max(overheads),
        "mean_trace_bytes": statistics.mean(trace_sizes),
        "p95_trace_bytes": percentile(trace_sizes, 95),
        "max_trace_bytes": max(trace_sizes),
        "wall_ms": wall_ms,
    }


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = int(round((pct / 100.0) * (len(values) - 1)))
    return values[k]


def main():
    parser = argparse.ArgumentParser(description="Benchmark TraceRecorder overhead")
    parser.add_argument("--traces", type=int, default=1000)
    parser.add_argument("--rules", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output", type=str, default="output/m2_e5_trace_meta_query/trace_bench.txt"
    )
    args = parser.parse_args()

    metrics = run_benchmark(args.traces, args.rules, args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}: {v}" for k, v in metrics.items()]
    payload = "\n".join(lines)
    out_path.write_text(payload, encoding="utf-8")
    print(payload)
    print(f"Wrote benchmark results to {out_path}")


if __name__ == "__main__":
    main()
