#!/usr/bin/env python3
"""Generate a realistic E2E query corpus for M2-E7 aligned to the M2 intent taxonomy.

Intents match the 8-class label set used throughout M2:
  point_in_time, interval, sequence, causal, comparative,
  aggregation, prediction, explanation

Queries are drawn from the annotated corpus (experiments/m2_e2_intent/data/annotated_queries.jsonl)
when available, then topped-up with template-generated examples so the corpus always
reaches --count entries.

Required facts per query are derived from the intent:
  - all queries require "intent"
  - causal / explanation additionally require "causal_link"
  - prediction additionally requires "forecast"
  - aggregation additionally requires "count"
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

# Canonical M2 taxonomy intents
INTENTS = [
    "point_in_time",
    "interval",
    "sequence",
    "causal",
    "comparative",
    "aggregation",
    "prediction",
    "explanation",
]

# Template queries per intent (used when the gold corpus runs out)
TEMPLATES: Dict[str, List[str]] = {
    "point_in_time": [
        "When did the {event} happen?",
        "On what date was {event} announced?",
        "At what point in time did {metric} peak?",
        "When was the {entity} policy implemented?",
        "On which exact day did {entity} cross the threshold?",
    ],
    "interval": [
        "Show all events between {year_a} and {year_b} for {entity}.",
        "List everything that occurred during {period} in {domain}.",
        "Retrieve all data from {year_a} to {year_b}.",
        "What happened across {entity} during {period}?",
        "Show the timeline from {event_a} to {event_b}.",
    ],
    "sequence": [
        "What was the order of steps that led to {event}?",
        "Walk me through the sequence of the {event}.",
        "In what order did the {entity} complications develop?",
        "Trace the pathway from {event_a} to {event_b}.",
        "What sequence of decisions led to {outcome}?",
    ],
    "causal": [
        "What caused the {event}?",
        "Why did {metric} drop in {period}?",
        "What triggered the {entity} investigation?",
        "Which upstream failures led to {event}?",
        "What drove the {metric} compression in {entity}?",
    ],
    "comparative": [
        "Compare the timelines for {entity_a} and {entity_b}.",
        "How did the rollout differ between {entity_a} and {entity_b}?",
        "Contrast {metric} before and after {event}.",
        "Compare {metric} across {entity_a}, {entity_b}, and {entity_c}.",
        "How does this {period} performance rank against last year?",
    ],
    "aggregation": [
        "How many {entity} occurred each {period_unit} in {year_a}?",
        "What is the total count of {metric}?",
        "Count the number of {entity} per {period_unit}.",
        "What is the distribution of {entity} by category?",
        "How frequently did {entity} report this issue?",
    ],
    "prediction": [
        "Given events {event_a} and {event_b}, what is likely to happen next?",
        "Forecast {metric} for {period}.",
        "What would happen if we delayed {event} by two weeks?",
        "Predict the {metric} rate for next quarter.",
        "Under current constraints, when will {entity} reach the target?",
    ],
    "explanation": [
        "Explain the causal chain leading to {event}.",
        "Why should we believe this timeline of {entity}?",
        "Walk through the reasoning behind the {entity} plan.",
        "Explain why {metric} improved despite {condition}.",
        "Justify the decision to {action} at this point.",
    ],
}

FILL_VALUES = {
    "event": ["the market crash", "the product recall", "the system outage", "the policy change", "the merger"],
    "event_a": ["the initial alert", "contract signing", "the prototype review", "Phase 1"],
    "event_b": ["full containment", "first revenue recognition", "production handoff", "Phase 3"],
    "metric": ["revenue", "ICU admissions", "error rate", "churn rate", "latency"],
    "entity": ["the regulatory body", "the supply chain", "the clinical trial", "the platform", "the team"],
    "entity_a": ["Company A", "Region East", "Treatment Group 1", "the legacy system"],
    "entity_b": ["Company B", "Region West", "Treatment Group 2", "the new system"],
    "entity_c": ["Company C", "Region North", "Control Group"],
    "period": ["Q2 2025", "the fiscal year", "the trial period", "H1 2024", "the rollout window"],
    "period_unit": ["month", "week", "quarter", "day"],
    "year_a": ["2022", "2023", "2020"],
    "year_b": ["2024", "2025", "2022"],
    "domain": ["healthcare", "logistics", "financial services", "infrastructure"],
    "outcome": ["the recall", "the outage", "the successful launch", "market exit"],
    "condition": ["stable admission volumes", "constant headcount", "rising traffic"],
    "action": ["deprecate the legacy API", "scale out the cluster", "halt the trial"],
}


def _fill(template: str, rng: random.Random) -> str:
    """Replace all {placeholders} with random fill values."""
    import re
    keys = re.findall(r"\{(\w+)\}", template)
    result = template
    for key in keys:
        choices = FILL_VALUES.get(key, [key])
        result = result.replace(f"{{{key}}}", rng.choice(choices), 1)
    return result


def _required_facts(intent: str) -> List[str]:
    base = ["intent"]
    extras = {
        "causal": ["causal_link"],
        "explanation": ["causal_link"],
        "prediction": ["forecast"],
        "aggregation": ["count"],
        "interval": ["time_range"],
        "sequence": ["ordered_steps"],
        "comparative": ["comparison_result"],
        "point_in_time": ["event_date"],
    }
    return base + extras.get(intent, [])


def load_gold_queries(gold_path: Path, max_per_intent: int, rng: random.Random) -> List[dict]:
    """Load real queries from annotated_queries.jsonl, sampling up to max_per_intent per label."""
    if not gold_path.exists():
        return []
    by_intent: Dict[str, List[dict]] = {intent: [] for intent in INTENTS}
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for intent in row.get("intents", []):
                if intent in by_intent and len(by_intent[intent]) < max_per_intent:
                    by_intent[intent].append(row)
                    break  # assign to first matching intent only

    out = []
    for intent, rows in by_intent.items():
        rng.shuffle(rows)
        for row in rows:
            primary_intent = next((i for i in row.get("intents", []) if i in INTENTS), intent)
            out.append({
                "query_id": row.get("id", f"gold_{len(out)}"),
                "text": row.get("query", ""),
                "intent": primary_intent,
                "required_facts": _required_facts(primary_intent),
                "max_latency_ms": 15.0,
                "source": "gold",
            })
    return out


def generate_synthetic(count: int, rng: random.Random, start_idx: int = 0) -> List[dict]:
    """Generate template-based synthetic queries cycling through all intents."""
    out = []
    for i in range(count):
        intent = INTENTS[(start_idx + i) % len(INTENTS)]
        templates = TEMPLATES[intent]
        text = _fill(rng.choice(templates), rng)
        out.append({
            "query_id": f"syn_{start_idx + i:05d}",
            "text": text,
            "intent": intent,
            "required_facts": _required_facts(intent),
            "max_latency_ms": 15.0,
            "source": "synthetic",
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate E2E query corpus for M2-E7")
    parser.add_argument("--count", type=int, default=2210,
                        help="Total number of queries to emit")
    parser.add_argument("--gold", type=str,
                        default="experiments/m2_e2_intent/data/annotated_queries.jsonl",
                        help="Path to annotated_queries.jsonl for real query sourcing")
    parser.add_argument("--max-gold-per-intent", type=int, default=200,
                        help="Max gold queries to include per intent class")
    parser.add_argument("--output", type=str,
                        default="experiments/m2_e7_harness/input/queries.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load gold first
    gold = load_gold_queries(Path(args.gold), args.max_gold_per_intent, rng)
    rng.shuffle(gold)

    # Top-up with synthetic
    needed = max(0, args.count - len(gold))
    synthetic = generate_synthetic(needed, rng, start_idx=len(gold))

    all_queries = gold + synthetic
    # Re-assign sequential IDs while preserving source tag
    for idx, q in enumerate(all_queries):
        if q.get("source") == "synthetic":
            q["query_id"] = f"syn_{idx:05d}"

    all_queries = all_queries[: args.count]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q) + "\n")

    # Summary
    from collections import Counter
    intent_counts = Counter(q["intent"] for q in all_queries)
    source_counts = Counter(q.get("source", "?") for q in all_queries)
    print(f"Wrote {len(all_queries)} queries to {out_path}")
    print(f"  Source breakdown: {dict(source_counts)}")
    print(f"  Intent distribution: {dict(intent_counts)}")


if __name__ == "__main__":
    main()
