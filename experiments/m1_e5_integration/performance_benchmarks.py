#!/usr/bin/env python3
"""
M1-E5: Performance benchmarks for the temporal NLG stack.

This script runs lightweight, network-free benchmarks so it is safe in
offline CI. Results are written to output/m1_e5_integration/<ts>/.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List
from uuid import uuid4

# Ensure src is importable when run directly
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from temporal_nlg.core.templates import TemplateRenderer, TemplateType
from temporal_nlg.models import HybridGenerator
from temporal_nlg.evaluation import AccuracyEvaluator
from temporal_nlg.data.loaders import generate_examples


def _latency_samples(fn: Callable[[], None], runs: int) -> List[float]:
    samples: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)  # ms
    return samples


def _summarize(samples: List[float]) -> Dict[str, float]:
    samples_sorted = sorted(samples)
    return {
        "p50_ms": statistics.median(samples_sorted),
        "p95_ms": samples_sorted[int(len(samples_sorted) * 0.95) - 1],
        "mean_ms": statistics.fmean(samples_sorted),
    }


def benchmark_template_renderer() -> Dict[str, float]:
    renderer = TemplateRenderer()
    facts = generate_examples(TemplateType.POINT_IN_TIME, n=200)
    samples = _latency_samples(lambda: renderer.render(facts.pop()), runs=100)
    return _summarize(samples)


def benchmark_hybrid_cached() -> Dict[str, float]:
    generator = HybridGenerator(enable_caching=True)
    fact = generate_examples(TemplateType.POINT_IN_TIME, n=1)[0]

    # Warm cache once
    generator.generate(fact, force_strategy="template")

    def cached_call():
        generator.generate(fact, force_strategy="template")

    samples = _latency_samples(cached_call, runs=100)
    cache_stats = generator.get_stats()
    summary = _summarize(samples)
    summary["cache_size"] = cache_stats.get("cache_size", 0)
    return summary


def benchmark_hybrid_uncached(batch_size: int = 50) -> Dict[str, float]:
    generator = HybridGenerator(enable_caching=False)
    facts = generate_examples(TemplateType.POINT_IN_TIME, n=batch_size)

    start = time.perf_counter()
    results = generator.batch_generate(facts)
    duration_ms = (time.perf_counter() - start) * 1000.0

    return {
        "batch_size": batch_size,
        "total_ms": duration_ms,
        "per_item_ms": duration_ms / batch_size,
        "strategies": [r.strategy for r in results],
    }


def benchmark_evaluator_batch() -> Dict[str, float]:
    evaluator = AccuracyEvaluator()
    facts = generate_examples(TemplateType.POINT_IN_TIME, n=50)
    renderer = TemplateRenderer()
    texts = [renderer.render(f) for f in facts]

    start = time.perf_counter()
    evaluator.batch_evaluate(facts, texts)
    duration_ms = (time.perf_counter() - start) * 1000.0
    return {
        "batch_size": len(facts),
        "total_ms": duration_ms,
        "per_item_ms": duration_ms / len(facts),
    }


def main() -> int:
    print("Running performance benchmarks (offline)...")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "template_renderer": benchmark_template_renderer(),
        "hybrid_generator_cached": benchmark_hybrid_cached(),
        "hybrid_generator_uncached": benchmark_hybrid_uncached(),
        "accuracy_evaluator_batch": benchmark_evaluator_batch(),
    }

    results_dir = Path(__file__).resolve().parents[2] / "output" / "m1_e5_integration"
    results_dir.mkdir(exist_ok=True, parents=True)
    out_path = results_dir / uuid4().hex / "performance_benchmarks.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
