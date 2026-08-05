#!/usr/bin/env python3
"""MORK HTTP vs hyperon benchmarks (M4) — the harness behind the RESULTS_M4 table.

Measures the two kernels on the operation each is designed for, and writes the
raw numbers to ``output/benchmarks/milestone4/m4_mork_http.json`` so the table
in ``docs/RESULTS_M4.md`` is reproducible:

* **MORK** (pure-Rust atomspace): pattern-matched export latency, single-edge
  upload latency, batch upload throughput. Requires a live MORK HTTP server
  (default ``http://127.0.0.1:8000``, override with ``MORK_SERVER_URL``).
* **hyperon** (MeTTa evaluator): full ``metta.run`` round-trip for a grounded
  temporal op — MeTTa parse + evaluation + Python FFI per call.

The comparison is intentionally asymmetric (MORK cannot host Python grounded
ops; hyperon is not an atomspace server) — the artifact records exactly what
was measured, and the doc presents it with that caveat.

Benchmark namespaces use unique prefixes and are cleared afterwards, so the
shared atomspace is left as it was found. Missing server / hyperon are
recorded as ``null`` sections rather than failures.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import uuid
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
        "min_ms": samples_ms[0],
        "max_ms": samples_ms[-1],
    }


def _bench(fn, *, iterations: int, warmup: int = 5) -> list[float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


# ----------------------------------------------------------------------
# MORK HTTP atomspace benchmarks
# ----------------------------------------------------------------------
def bench_mork_http() -> dict | None:
    from temporal_nlg_metta import MORKHttpRunner, mork_http_available

    if not mork_http_available():
        print("  MORK HTTP server not reachable — recording null section")
        return None

    runner = MORKHttpRunner(timeout_s=10.0)
    ns = f"m4bench_{uuid.uuid4().hex[:8]}"
    pattern = f"({ns} $e $s $r $t $y)"

    def wait_export(pat: str, tpl: str, timeout_s: float = 10.0) -> str:
        deadline = time.monotonic() + timeout_s
        out = ""
        while time.monotonic() < deadline:
            out = runner.export(pat, tpl)
            if out.strip():
                return out
            time.sleep(0.1)
        return out

    try:
        # B-M1: pattern-matched export latency over the shared `edge` space.
        export_stats = _stats(
            _bench(
                lambda: runner.export("(edge $e $s $r $t $y)", "($e $s $r $t $y)"), iterations=50
            )
        )

        # B-M2: single-edge upload latency (synchronous ACK per upload).
        i = 0

        def upload_one():
            nonlocal i
            i += 1
            runner.upload(pattern, pattern, f"({ns} b{i} S{i} rel{i} T{i} {2000 + (i % 50)})")

        upload_stats = _stats(_bench(upload_one, iterations=50))

        # B-M3: batch upload throughput — 100 edges in one body, timed to the
        # point the data is visible to queries (ACK + async commit).
        batch = "\n".join(f"({ns} e{j} S{j} rel{j} T{j} {2000 + (j % 50)})" for j in range(100))
        t0 = time.perf_counter()
        runner.upload(pattern, pattern, batch)
        visible = wait_export(f"({ns} e99 $s $r $t $y)", "($s $r $t $y)")
        elapsed = time.perf_counter() - t0
        if not visible.strip():
            print("  WARNING: batch upload not visible after 10s; throughput is a lower bound")
        batch_stats = {
            "edges": 100,
            "elapsed_s": elapsed,
            "edges_per_sec": 100.0 / elapsed,
        }
        return {
            "export_query_ms": export_stats,
            "single_edge_upload_ms": upload_stats,
            "batch_upload": batch_stats,
        }
    finally:
        runner.clear(pattern)


# ----------------------------------------------------------------------
# hyperon MeTTa-eval benchmarks (same ops as the M4 bridge benchmarks, but
# timed through the full MeTTa parse + eval + FFI round-trip)
# ----------------------------------------------------------------------
def bench_hyperon_eval() -> dict | None:
    from temporal_nlg_metta import TemporalBridge, hyperon_available, make_metta_runner
    from temporal_nlg_metta.config import MettaConfig

    if not hyperon_available():
        print("  hyperon not installed — recording null section")
        return None

    bridge = TemporalBridge(config=MettaConfig(graph_dir=Path("/nonexistent")))
    metta = make_metta_runner(bridge=bridge)

    simple = _stats(_bench(lambda: metta.run("!(tms-contradictions)"), iterations=30, warmup=3))
    nlg = _stats(
        _bench(
            lambda: metta.run(
                '!(nlg-fact "point_in_time" '
                '"{\\"entity\\":\\"Einstein\\",\\"event\\":\\"was born\\",\\"date\\":\\"1879\\"}" '
                '"template")'
            ),
            iterations=30,
            warmup=3,
        )
    )
    return {
        "simple_grounded_op_eval_ms": simple,
        "nlg_fact_template_eval_ms": nlg,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("MORK HTTP vs hyperon benchmarks (M4)")
    print("=" * 60)

    print("MORK HTTP atomspace...")
    mork = bench_mork_http()
    print("hyperon MeTTa eval...")
    hyperon = bench_hyperon_eval()

    from temporal_nlg_metta import MORKHttpRunner

    result = {
        "meta": {
            "mork_server_url": MORKHttpRunner.DEFAULT_URL,
            "python_version": sys.version.split()[0],
            "hyperon_version": _hyperon_version(),
            "caveat": (
                "Asymmetric by design: MORK rows measure raw atomspace "
                "upload/export (no Python ops can run on MORK); hyperon rows "
                "measure full MeTTa parse+eval+FFI round-trips."
            ),
        },
        "mork_http": mork,
        "hyperon_eval": hyperon,
    }

    out_file = OUTPUT_DIR / "m4_mork_http.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out_file}")

    if mork:
        print(f"  MORK export p50:        {mork['export_query_ms']['p50_ms']:.2f} ms")
        print(f"  MORK upload p50:        {mork['single_edge_upload_ms']['p50_ms']:.2f} ms")
        print(f"  MORK batch throughput:  {mork['batch_upload']['edges_per_sec']:.0f} edges/sec")
    if hyperon:
        print(f"  hyperon eval p50:       {hyperon['simple_grounded_op_eval_ms']['p50_ms']:.2f} ms")
        print(f"  hyperon nlg-fact p50:   {hyperon['nlg_fact_template_eval_ms']['p50_ms']:.2f} ms")
    return 0


def _hyperon_version() -> str | None:
    try:
        import hyperon

        return getattr(hyperon, "__version__", "unknown")
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
