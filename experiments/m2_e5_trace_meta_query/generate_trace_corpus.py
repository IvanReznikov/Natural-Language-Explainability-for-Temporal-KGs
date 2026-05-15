#!/usr/bin/env python3
"""Generate a synthetic trace corpus for M2-E5 regression tests."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

from temporal_nlg.tms.trace import TraceRecorder


class IncrementalClock:
    """Deterministic clock to keep synthetic traces reproducible."""

    def __init__(self, start: float = 0.0, step: float = 0.001):
        self.value = start
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def synthesize_trace(recorder: TraceRecorder, rng: random.Random, idx: int) -> Dict:
    trace = recorder.start_query(f"q_{idx}", meta={"user": f"user_{idx % 5}", "intent": "demo"})
    rule_count = rng.randint(1, 4)

    for rule_idx in range(rule_count):
        recorder.record_rule_firing(
            trace,
            rule_id=f"rule_{rule_idx}",
            rule_name=f"rule_{rule_idx}",
            inputs=[{"fact_id": f"f_{idx}_{rule_idx}", "value": rng.randint(1, 100)}],
            conclusion={"fact_id": f"g_{idx}_{rule_idx}", "value": rng.randint(1, 100)},
            confidence=round(rng.uniform(0.7, 1.0), 2),
            latency_ms=round(rng.uniform(0.1, 2.5), 3),
            meta={"depth": rule_idx},
        )

    recorder.complete_query(trace)
    return trace.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic traces for M2-E5")
    parser.add_argument("--count", type=int, default=1000, help="Number of queries to synthesize")
    parser.add_argument(
        "--output",
        type=str,
        default="output/m2_e5_trace_meta_query/synthetic_traces.jsonl",
        help="Path to the JSONL file to write",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    clock = IncrementalClock()
    recorder = TraceRecorder(
        sampling_rate=1.0,
        max_overhead_ms=1000.0,
        time_fn=clock,
        perf_fn=clock,
        rand_fn=rng.random,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(args.count):
            payload = synthesize_trace(recorder, rng, idx)
            f.write(record_trace_to_json(payload))
            f.write("\n")

    print(f"Wrote {args.count} synthetic traces to {out_path}")


def record_trace_to_json(trace_dict: Dict) -> str:
    """Serialize a trace dictionary to JSON with stable formatting."""

    import json

    return json.dumps(trace_dict, sort_keys=True)


if __name__ == "__main__":
    main()

