#!/usr/bin/env python3
"""Generate synthetic QueryTrace JSONL aligned to a queries.jsonl file.

Each trace emits rule firings whose conclusions cover all required_facts
declared in the corresponding query entry.  Rule names and IDs are derived
from the query's intent, matching the M2 taxonomy rule vocabulary used in
the TMS runtime (temporal_nlg.tms).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable, List, Dict

Rule = Dict[str, object]
TraceRecord = Dict[str, object]

# Maps M2 taxonomy intent -> (rule_id_prefix, rule_name, input_fact_id)
INTENT_RULES = {
    "point_in_time": ("r_pit", "ResolveTemporalAnchor", "query_text"),
    "interval": ("r_ivl", "ExtractIntervalBounds", "query_text"),
    "sequence": ("r_seq", "OrderEventSequence", "query_text"),
    "causal": ("r_csl", "VerifyTemporalCorrelation", "query_text"),
    "comparative": ("r_cmp", "ComputeComparisonResult", "query_text"),
    "aggregation": ("r_agg", "AggregateTemporalFacts", "query_text"),
    "prediction": ("r_prd", "ProjectFutureTrend", "query_text"),
    "explanation": ("r_exp", "BuildCausalExplanation", "query_text"),
    # fallback for any unexpected intent label
    "_default": ("r_gen", "GenericTemporalQuery", "query_text"),
}

# Human-readable conclusion values per fact_id
FACT_VALUES: Dict[str, Dict[str, str]] = {
    "intent": {
        "point_in_time": "point_in_time",
        "interval": "interval",
        "sequence": "sequence",
        "causal": "causal",
        "comparative": "comparative",
        "aggregation": "aggregation",
        "prediction": "prediction",
        "explanation": "explanation",
    },
    "causal_link": {"_default": "cause_effect_verified"},
    "forecast": {"_default": "trend_projected"},
    "count": {"_default": "aggregation_count_computed"},
    "time_range": {"_default": "interval_bounds_resolved"},
    "ordered_steps": {"_default": "sequence_ordered"},
    "comparison_result": {"_default": "comparison_computed"},
    "event_date": {"_default": "temporal_anchor_resolved"},
}


def _fact_value(fact_id: str, intent: str, qid: str) -> str:
    table = FACT_VALUES.get(fact_id, {})
    return table.get(intent) or table.get("_default") or f"val_{fact_id}_{qid}"


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


def make_rule(query: dict, now: float, rng: random.Random) -> Rule:
    intent = query.get("intent", "_default")
    base_rule_id, rule_name, input_key = INTENT_RULES.get(intent, INTENT_RULES["_default"])
    qid = query.get("query_id", "q_unknown")

    required_facts: List[str] = query.get("required_facts") or ["intent"]

    # Build one conclusion per required fact; first goes into the main conclusion field,
    # rest go into meta.extra_conclusions so run_e2e.py can expand them.
    conclusions = []
    for fact_id in required_facts:
        conclusions.append({
            "fact_id": fact_id,
            "value": _fact_value(fact_id, intent, qid),
        })

    # Guarantee intent is always in conclusions
    intent_conclusion = {"fact_id": "intent", "value": intent}
    if not any(c["fact_id"] == "intent" for c in conclusions):
        conclusions.insert(0, intent_conclusion)

    main_conclusion = conclusions[0]
    extra_conclusions = conclusions[1:]

    return {
        "rule_id": f"{base_rule_id}_{qid}",
        "rule_name": rule_name,
        "inputs": [
            {"fact_id": input_key, "value": query.get("text", "")},
        ],
        "conclusion": main_conclusion,
        "fired_at": now + rng.uniform(0.2, 0.8),
        "confidence": round(rng.uniform(0.87, 0.99), 3),
        "latency_ms": round(rng.uniform(1.5, 12.0), 2),
        "meta": {"gen": "synthetic", "extra_conclusions": extra_conclusions},
    }


def make_trace(query: dict, base_ts: float, rng: random.Random) -> TraceRecord:
    qid = query.get("query_id", "q_unknown")
    started = base_ts
    completed = started + rng.uniform(0.4, 1.4)
    rule = make_rule(query, started, rng)
    return {
        "query_id": qid,
        "started_at": started,
        "completed_at": completed,
        "instrumentation_overhead_ms": round(rng.uniform(0.3, 1.8), 2),
        "dropped": False,
        "over_budget": False,
        "meta": {"gen": "synthetic", "intent": query.get("intent", "unknown")},
        "rule_traces": [rule],
    }


def generate_traces(queries: List[dict], seed: int) -> List[TraceRecord]:
    rng = random.Random(seed)
    now = time.time()
    traces = []
    for i, q in enumerate(queries):
        base_ts = now + i * 1.5 + rng.uniform(0, 0.3)
        traces.append(make_trace(q, base_ts, rng))
    return traces


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic traces for M2-E7")
    parser.add_argument("--queries", type=str,
                        default="experiments/m2_e7_harness/input/queries.jsonl")
    parser.add_argument("--output", type=str,
                        default="experiments/m2_e7_harness/input/trace.jsonl")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    queries = load_jsonl(Path(args.queries))
    traces = generate_traces(queries, args.seed)
    save_jsonl(Path(args.output), traces)

    from collections import Counter
    intent_counts = Counter(t["meta"].get("intent", "?") for t in traces)
    print(f"Wrote {len(traces)} traces to {args.output}")
    print(f"  Intent distribution in traces: {dict(intent_counts)}")


if __name__ == "__main__":
    main()
