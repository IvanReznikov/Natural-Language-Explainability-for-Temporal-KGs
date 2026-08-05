#!/usr/bin/env python3
"""Run all Milestone 4 integration benchmarks.

These benchmarks measure the **temporal_nlg_metta integration layer** — the cost
of bridging the M1/M2/M3 capabilities into MeTTa grounded operations. They do
NOT require the optional ``hyperon`` dependency: they measure the Python bridge
and grounded-op dispatch that any MeTTa kernel (hyperon or MORK) invokes.

The numbers feed the two-layer cost model documented in ``docs/RESULTS_M4.md``:
  program_time = (MeTTa reduction steps) * kernel_time_per_step
               + (grounded calls)        * python_time_per_call
MORK accelerates the first term; the second is fixed and measured here.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "output" / "benchmarks" / "milestone4"


def _stats(samples_ms: list[float]) -> dict:
    samples_ms = sorted(samples_ms)
    n = len(samples_ms)
    return {
        "n": n,
        "mean_ms": statistics.fmean(samples_ms),
        "p50_ms": samples_ms[n // 2],
        "p95_ms": samples_ms[min(n - 1, int(n * 0.95))],
        "p99_ms": samples_ms[min(n - 1, int(n * 0.99))],
        "min_ms": samples_ms[0],
        "max_ms": samples_ms[-1],
    }


def _bench(fn, *, iterations: int, warmup: int = 50) -> list[float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


# ----------------------------------------------------------------------
# B1 — Bridge construction cost (M1/M2 eager; M3 lazy)
# ----------------------------------------------------------------------
def bench_bridge_construction() -> dict:
    from temporal_nlg_metta import TemporalBridge
    from temporal_nlg_metta.config import MettaConfig

    cfg = MettaConfig(graph_dir=Path("/nonexistent"))  # M3 stays lazy/unbuilt
    samples = _bench(lambda: TemporalBridge(config=cfg), iterations=100, warmup=10)
    return {"bridge_construction_ms": _stats(samples)}


# ----------------------------------------------------------------------
# B2 — Per-grounded-op latency (the Python layer every kernel invokes)
# ----------------------------------------------------------------------
def bench_grounded_op_latency() -> dict:
    from temporal_nlg_metta import TemporalBridge
    from temporal_nlg_metta.config import MettaConfig

    bridge = TemporalBridge(config=MettaConfig(graph_dir=Path("/nonexistent")))
    import json as _json

    pit_content = _json.dumps({"entity": "Einstein", "event": "was born", "date": "1879"})

    ops: dict[str, callable] = {
        "nlg-fact(template)": lambda: bridge.nlg_fact("point_in_time", pit_content, "template"),
        "nlg-readability": lambda: bridge.nlg_readability("Einstein was born in 1879."),
        "tms-start-trace": lambda: bridge.start_trace(),
        "tms-add-belief": lambda: bridge.add_belief("bench_b", "{}", "[]", "[]"),
        "tms-rules-fired": lambda: bridge.rules_fired(),
        "tms-contradictions": lambda: bridge.contradictions(),
        "tms-active-beliefs": lambda: bridge.active_beliefs(),
        "tms-explain(miss)": lambda: bridge.explain_belief("absent_id"),
    }

    results: dict[str, dict] = {}
    for name, fn in ops.items():
        bridge.reset()
        if name.startswith("tms-record") or name == "tms-rules-fired":
            bridge.start_trace("bench")
        samples = _bench(fn, iterations=500, warmup=100)
        results[name] = _stats(samples)
    return {"grounded_op_latency_ms": results}


# ----------------------------------------------------------------------
# B3 — JSON marshalling overhead (the string-return convention)
# ----------------------------------------------------------------------
def bench_json_marshalling() -> dict:
    from temporal_nlg_metta.atoms import _dumps

    small = {"text": "Einstein was born in 1879.", "strategy": "template", "confidence": 0.9}
    large = {
        "evidence": [
            {
                "source": f"node_{i}",
                "target": f"node_{i + 1}",
                "relation": "caused",
                "start": 1900 + i,
            }
            for i in range(20)
        ],
        "answer_text": "A long natural-language answer " * 10,
    }

    return {
        "json_dumps_ms": {
            "small_dict": _stats(_bench(lambda: _dumps(small), iterations=2000, warmup=500)),
            "large_dict": _stats(_bench(lambda: _dumps(large), iterations=2000, warmup=500)),
        }
    }


# ----------------------------------------------------------------------
# B4 — TMS throughput (record_rule rules/sec)
# ----------------------------------------------------------------------
def bench_tms_throughput() -> dict:
    from temporal_nlg_metta import TemporalBridge
    from temporal_nlg_metta.config import MettaConfig

    bridge = TemporalBridge(config=MettaConfig(graph_dir=Path("/nonexistent")))
    import json as _json

    inputs = _json.dumps([{"fact_id": "f"}])
    conclusion = _json.dumps({"fact_id": "c", "value": "v"})

    def record_many(n: int):
        bridge.reset()
        bridge.start_trace("bench")
        for i in range(n):
            bridge.record_rule(f"r{i}", "rule", inputs, conclusion, 0.9)

    # Warm up, then time a batch of 1000 rule firings.
    record_many(100)
    t0 = time.perf_counter()
    record_many(1000)
    elapsed_s = time.perf_counter() - t0
    return {
        "tms_throughput": {
            "rules_per_sec": 1000.0 / elapsed_s,
            "per_rule_us": elapsed_s * 1e6 / 1000.0,
        }
    }


# ----------------------------------------------------------------------
# B5 — Operation registration cost (register_with onto a runner)
# ----------------------------------------------------------------------
def bench_registration() -> dict:
    from temporal_nlg_metta import TemporalBridge
    from temporal_nlg_metta.atoms import available_tokens
    from temporal_nlg_metta.config import MettaConfig

    bridge = TemporalBridge(config=MettaConfig(graph_dir=Path("/nonexistent")))
    token_count = len(available_tokens())

    # Without hyperon we can't call register_with; measure the spec build cost
    # instead (the work done per registration regardless of kernel).
    from temporal_nlg_metta.atoms import all_specs

    samples = _bench(lambda: all_specs(), iterations=500, warmup=100)
    return {
        "registration": {
            "token_count": token_count,
            "spec_build_ms": _stats(samples),
        }
    }


# ----------------------------------------------------------------------
# B6 — M3 graph query latency (one real end-to-end query)
# ----------------------------------------------------------------------
def bench_m3_query() -> dict | None:
    graph_dir = ROOT / "data" / "jsonls" / "temporal_graph_output_v3"
    if not (graph_dir / "nodes.jsonl").exists():
        return None

    from temporal_nlg_metta import TemporalBridge

    bridge = TemporalBridge()
    question = "What likely caused the Model T price drop in 1913?"

    # Warm up the lazy pipeline + embeddings on the first call.
    bridge.answer(question)

    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        bridge.answer(question)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"m3_query_ms": _stats(samples)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="M4 integration benchmarks")
    parser.add_argument(
        "--m3",
        action="store_true",
        help="Also run the B6 M3 graph-query benchmark (slow; loads model servers).",
    )
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("M4 integration benchmarks (Python bridge layer)")
    print("=" * 60)

    all_results: dict = {}

    print("B1: bridge construction...")
    all_results.update(bench_bridge_construction())
    print("B2: grounded-op latency...")
    all_results.update(bench_grounded_op_latency())
    print("B3: JSON marshalling...")
    all_results.update(bench_json_marshalling())
    print("B4: TMS throughput...")
    all_results.update(bench_tms_throughput())
    print("B5: registration spec build...")
    all_results.update(bench_registration())

    if args.m3:
        print("B6: M3 graph query...")
        m3 = bench_m3_query()
        if m3 is not None:
            all_results.update(m3)
        else:
            all_results["m3_query_ms"] = None
            print("  (skipped — graph artifacts not found)")
    else:
        all_results["m3_query_ms"] = None
        print("B6: M3 graph query (skipped — pass --m3 to enable)")

    out_file = OUTPUT_DIR / "m4_benchmarks.json"
    out_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_file}")

    # Console summary of the headline numbers.
    print("\nHeadline numbers:")
    bc = all_results["bridge_construction_ms"]
    print(f"  bridge construction (M1/M2 eager):   {bc['p50_ms']:.3f} ms p50")
    ops = all_results["grounded_op_latency_ms"]
    cheap = [v["p50_ms"] for v in ops.values()]
    print(f"  grounded-op latency (min..max p50): {min(cheap):.4f} .. {max(cheap):.4f} ms")
    tt = all_results["tms_throughput"]
    print(
        f"  TMS record_rule throughput:          {tt['rules_per_sec']:.0f} rules/sec "
        f"({tt['per_rule_us']:.2f} us/rule)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
