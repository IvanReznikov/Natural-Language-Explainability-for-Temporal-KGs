#!/usr/bin/env python3
"""Micro-benchmark for query storage (M2-E6)."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from temporal_nlg.tms.query_store import QueryStore


def run(traces: int, intents: int) -> dict:
    path = Path("output/m2_e6_query_store_triggers/bench_queries.jsonl")
    store = QueryStore(path=path)

    sizes = []
    latencies = []
    start = time.perf_counter()
    for i in range(traces):
        t0 = time.perf_counter()
        store.upsert(
            query_id=f"q_{i}",
            text=f"Synthetic query {i}",
            intent=f"intent_{i % intents}",
            meta={"i": i},
            dependencies=[f"fact_{i%5}"],
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        sizes.append(len(store.get(f"q_{i}").text.encode("utf-8")))
    wall_ms = (time.perf_counter() - start) * 1000.0
    return {
        "count": traces,
        "mean_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "max_latency_ms": max(latencies),
        "mean_size_bytes": statistics.mean(sizes),
        "wall_ms": wall_ms,
    }


def percentile(values, pct):
    values = sorted(values)
    k = int(round((pct / 100.0) * (len(values) - 1)))
    return values[k]


def main():
    parser = argparse.ArgumentParser(description="Query store benchmark")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--intents", type=int, default=5)
    parser.add_argument(
        "--output", type=str, default="output/m2_e6_query_store_triggers/query_store_bench.txt"
    )
    args = parser.parse_args()
    metrics = run(args.count, args.intents)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join([f"{k}: {v}" for k, v in metrics.items()]), encoding="utf-8")
    print(metrics)
    print(f"Wrote benchmark results to {out_path}")


if __name__ == "__main__":
    main()
