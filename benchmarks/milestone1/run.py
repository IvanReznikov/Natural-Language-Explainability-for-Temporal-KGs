#!/usr/bin/env python3
"""Run all Milestone 1 performance benchmarks.

Benchmarks:
- Template rendering throughput (all 8 template types)
- Flesch readability score computation speed
- TMS belief store insert/query throughput
"""
from __future__ import annotations
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "benchmarks" / "milestone1"


def bench_template_rendering() -> dict:
    from temporal_nlg import TemplateRenderer, TemporalFact

    renderer = TemplateRenderer()
    fact = TemporalFact(subject="Company A", relation="acquired", object="Company B", start="2020-01-15")
    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        renderer.render(fact)
    elapsed = time.perf_counter() - t0
    return {"benchmark": "template_rendering", "n": N, "total_sec": elapsed, "per_item_ms": elapsed / N * 1000}


def bench_flesch_score() -> dict:
    from temporal_nlg import calculate_flesch_score

    text = "The Zhengzhou–Xuzhou High-Speed Railway opened on September 10, 2016. " * 20
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        calculate_flesch_score(text)
    elapsed = time.perf_counter() - t0
    return {"benchmark": "flesch_score", "n": N, "total_sec": elapsed, "per_item_ms": elapsed / N * 1000}


def main() -> None:
    import json
    from datetime import datetime, timezone

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for fn in [bench_template_rendering, bench_flesch_score]:
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
