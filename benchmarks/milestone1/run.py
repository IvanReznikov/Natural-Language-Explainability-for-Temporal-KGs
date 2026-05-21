#!/usr/bin/env python3
"""Run all Milestone 1 performance benchmarks.

Benchmarks:
- Template rendering throughput (all available template types)
- Flesch readability score computation speed
- TMS belief store insert/query throughput
"""
from __future__ import annotations
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "benchmarks" / "milestone1"


def bench_template_rendering() -> dict:
    from temporal_nlg import TemplateRenderer, TemplateType
    from temporal_nlg.data.loaders import generate_examples

    renderer = TemplateRenderer()
    template_types = [
        TemplateType.POINT_IN_TIME,
        TemplateType.INTERVAL,
        TemplateType.SEQUENCE,
        TemplateType.CAUSALITY,
        TemplateType.OVERLAP,
    ]

    facts = []
    for template_type in template_types:
        facts.extend(generate_examples(template_type, n=120))

    N = len(facts)
    t0 = time.perf_counter()
    for fact in facts:
        renderer.render(fact)
    elapsed = time.perf_counter() - t0
    return {
        "benchmark": "template_rendering",
        "template_types": [template_type.value for template_type in template_types],
        "n": N,
        "total_sec": elapsed,
        "per_item_ms": elapsed / N * 1000,
    }


def bench_flesch_score() -> dict:
    from temporal_nlg import calculate_flesch_score

    text = "The Zhengzhou–Xuzhou High-Speed Railway opened on September 10, 2016. " * 20
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        calculate_flesch_score(text)
    elapsed = time.perf_counter() - t0
    return {"benchmark": "flesch_score", "n": N, "total_sec": elapsed, "per_item_ms": elapsed / N * 1000}


def bench_tms_belief_store() -> dict:
    from temporal_nlg.tms.belief_store import Belief, BeliefStore

    store = BeliefStore()
    N = 1500

    t0 = time.perf_counter()
    for i in range(N):
        supports = []
        if i > 0:
            supports.append(f"b{i - 1}")
        belief = Belief(
            belief_id=f"b{i}",
            payload={"value": i, "kind": "synthetic"},
            supports=supports,
        )
        store.add_belief(belief)
    insert_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    for i in range(N):
        _ = store.get_belief(f"b{i}")
    query_elapsed = time.perf_counter() - t1

    total_ops = N * 2
    total_elapsed = insert_elapsed + query_elapsed
    return {
        "benchmark": "tms_belief_store_insert_query",
        "n": total_ops,
        "insert_ops": N,
        "query_ops": N,
        "insert_total_sec": insert_elapsed,
        "query_total_sec": query_elapsed,
        "total_sec": total_elapsed,
        "per_item_ms": total_elapsed / total_ops * 1000,
    }


def main() -> None:
    import json
    from datetime import datetime, timezone

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for fn in [bench_template_rendering, bench_flesch_score, bench_tms_belief_store]:
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
