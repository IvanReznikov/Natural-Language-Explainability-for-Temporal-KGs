#!/usr/bin/env python3
"""Generate synthetic QueryTrace JSONL aligned to a queries.jsonl file."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable, List, Dict

Rule = Dict[str, object]
TraceRecord = Dict[str, object]

INTENT_RULES = {
    "medical": ("r_med", "extract_medical_facts", "symptom"),
    "financial": ("r_fin", "compute_financial_outcome", "definition"),
    "historical": ("r_hist", "retrieve_historical_context", "event"),
    "science": ("r_sci", "derive_scientific_fact", "mechanism"),
}


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_rule(query: dict, now: float) -> Rule:
    intent = query.get("intent", "unknown")
    base_rule_id, rule_name, input_key = INTENT_RULES.get(intent, ("r_generic", "generic_rule", "topic"))
    qid = query.get("query_id", "q_unknown")

    required_facts = query.get("required_facts") or []
    if not required_facts:
        required_facts = ["intent"]

    conclusions = []
    for fact in required_facts:
        conclusions.append({"fact_id": fact, "value": f"val_{fact}_{qid}"})

    # ensure intent fact exists for intent matching
    if "intent" not in [c["fact_id"] for c in conclusions]:
        conclusions.append({"fact_id": "intent", "value": intent})

    return {
        "rule_id": f"{base_rule_id}_{qid}",
        "rule_name": rule_name,
        "inputs": [
            {"fact_id": input_key, "value": query.get("text", "")},
        ],
        "conclusion": conclusions[0],
        "fired_at": now + random.uniform(0.2, 0.8),
        "confidence": round(random.uniform(0.85, 0.99), 3),
        "latency_ms": round(random.uniform(3.0, 12.0), 2),
        "meta": {"gen": "synthetic", "extra_conclusions": conclusions[1:]},
    }


def make_trace(query: dict, base_ts: float) -> TraceRecord:
    qid = query.get("query_id", "q_unknown")
    started = base_ts
    completed = started + random.uniform(0.8, 1.6)
    rule = make_rule(query, started)
    return {
        "query_id": qid,
        "started_at": started,
        "completed_at": completed,
        "instrumentation_overhead_ms": round(random.uniform(0.5, 2.0), 2),
        "dropped": False,
        "over_budget": False,
        "meta": {"gen": "synthetic"},
        "rule_traces": [rule],
    }


def generate_traces(queries: List[dict], seed: int) -> List[TraceRecord]:
    rng = random.Random(seed)
    now = time.time()
    traces = []
    for i, q in enumerate(queries):
        base_ts = now + i * 2.0 + rng.uniform(0, 0.5)
        random.seed(seed + i)  # keep per-query reproducibility
        traces.append(make_trace(q, base_ts))
    return traces


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic traces for M2-E7")
    parser.add_argument("--queries", type=str, default="experiments/m2_e7_harness/input/queries.jsonl")
    parser.add_argument("--output", type=str, default="experiments/m2_e7_harness/input/trace.jsonl")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    queries = load_jsonl(Path(args.queries))
    traces = generate_traces(queries, args.seed)
    save_jsonl(Path(args.output), traces)
    print(f"Wrote {len(traces)} traces to {args.output}")


if __name__ == "__main__":
    main()

