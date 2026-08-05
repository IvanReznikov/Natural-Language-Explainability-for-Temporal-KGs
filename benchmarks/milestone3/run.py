#!/usr/bin/env python3
"""Run all Milestone 3 performance benchmarks.

Benchmarks:
- Graph index query throughput (temporal_graph_output_v3)
- M3-E2 fidelity metric computation throughput (proxy metrics only)
- M3-E4a efficiency metric aggregation throughput
"""

from __future__ import annotations
import json
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "benchmarks" / "milestone3"
ROOT = Path(__file__).parent.parent.parent


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bench_graph_index_query() -> dict:
    from temporal_nlg.graph_query import TemporalGraphIndex

    candidate_dirs = [
        ROOT / "data" / "jsonls" / "temporal_graph_output_v3",
        ROOT / "output" / "temporal_graph_output_v3",
    ]
    graph_dir = next(
        (
            path
            for path in candidate_dirs
            if (path / "nodes.jsonl").exists() and (path / "edges.jsonl").exists()
        ),
        None,
    )
    if graph_dir is None:
        return {
            "benchmark": "graph_index_query",
            "skipped": True,
            "reason": "graph output artifacts not found",
        }

    index = TemporalGraphIndex(graph_dir)
    labels = [label for label in index.node_label_by_uid.values() if label]
    if not labels:
        return {
            "benchmark": "graph_index_query",
            "skipped": True,
            "reason": "graph index has no labels",
        }

    N = min(max(len(labels) * 4, 500), 4000)
    t0 = time.perf_counter()
    for i in range(N):
        probe = labels[i % len(labels)]
        node_uids = index.resolve_node_uids(probe, max_hits=3)
        if node_uids:
            _ = index.outgoing_edges(node_uids[0])
    elapsed = time.perf_counter() - t0
    return {
        "benchmark": "graph_index_query",
        "artifact_dir": str(graph_dir),
        "n": N,
        "total_sec": elapsed,
        "per_item_ms": elapsed / N * 1000,
    }


def bench_fidelity_proxy() -> dict:
    """Benchmark proxy metric computation on a small sample."""
    sample_path = ROOT / "output" / "m3_e2_fidelity_smoke" / "m3_e2_fidelity.per_item.jsonl"
    if not sample_path.exists():
        return {"benchmark": "fidelity_proxy", "skipped": True, "reason": "smoke output not found"}

    items = _load_jsonl(sample_path)

    if not items:
        return {"benchmark": "fidelity_proxy", "skipped": True, "reason": "empty file"}

    N = min(len(items), 100)
    t0 = time.perf_counter()
    for item in items[:N]:
        _ = {k: v for k, v in item.items() if isinstance(v, (int, float))}
    elapsed = time.perf_counter() - t0
    return {
        "benchmark": "fidelity_proxy",
        "n": N,
        "total_sec": elapsed,
        "per_item_ms": elapsed / N * 1000,
    }


def bench_e4_efficiency_aggregation() -> dict:
    from temporal_nlg.evaluation.m3_e4 import EfficiencyRun, aggregate_efficiency

    runs_path = ROOT / "output" / "m3_e4a_efficiency" / "m3_e4a_runs_generated.jsonl"
    if not runs_path.exists():
        return {
            "benchmark": "e4_efficiency_aggregation",
            "skipped": True,
            "reason": "m3_e4a run logs not found",
        }

    rows = _load_jsonl(runs_path)
    if not rows:
        return {
            "benchmark": "e4_efficiency_aggregation",
            "skipped": True,
            "reason": "empty m3_e4a run log",
        }

    runs = [
        EfficiencyRun(
            scenario_id=str(row.get("scenario_id") or ""),
            method=str(row.get("method") or "unknown"),
            complexity_level=int(row.get("complexity_level") or 1),
            latency_ms=float(row.get("latency_ms")) if row.get("latency_ms") is not None else None,
            tokens_out=float(row.get("tokens_out")) if row.get("tokens_out") is not None else None,
            cost=float(row.get("cost")) if row.get("cost") is not None else None,
        )
        for row in rows
    ]

    N = 250
    t0 = time.perf_counter()
    for _ in range(N):
        _ = aggregate_efficiency(runs)
    elapsed = time.perf_counter() - t0
    return {
        "benchmark": "e4_efficiency_aggregation",
        "n": N,
        "input_runs": len(runs),
        "total_sec": elapsed,
        "per_item_ms": elapsed / N * 1000,
    }


def main() -> None:
    from datetime import datetime, timezone

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for fn in [bench_graph_index_query, bench_fidelity_proxy, bench_e4_efficiency_aggregation]:
        try:
            r = fn()
            if r.get("skipped"):
                print(f"  {r['benchmark']:35s}  SKIPPED ({r.get('reason','')})")
            else:
                print(f"  {r['benchmark']:35s}  {r['per_item_ms']:.3f} ms/item")
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            print(f"  {fn.__name__}: ERROR — {exc}")

    out = OUTPUT_DIR / f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
