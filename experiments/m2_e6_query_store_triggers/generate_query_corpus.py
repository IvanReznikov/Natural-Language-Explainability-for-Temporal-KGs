#!/usr/bin/env python3
"""Generate synthetic queries/results for M2-E6 storage tests."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List

from temporal_nlg.tms.query_store import QueryStore
from temporal_nlg.tms.result_store import ResultStore


def main():
    parser = argparse.ArgumentParser(description="Generate query/result corpora")
    parser.add_argument("--queries", type=str, default="output/m2_e6_query_store_triggers/queries.jsonl")
    parser.add_argument("--results", type=str, default="output/m2_e6_query_store_triggers/results.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    q_store = QueryStore(path=Path(args.queries))
    r_store = ResultStore(path=Path(args.results))

    intents = ["medical", "financial", "historical"]

    for idx in range(args.count):
        intent = intents[idx % len(intents)]
        qid = f"q_{idx}"
        text = f"Query number {idx} about {intent}"
        deps = [f"fact_{idx%10}"]
        q_store.upsert(qid, text, intent=intent, meta={"k": idx}, dependencies=deps)

        rid = f"r_{idx}"
        res = [{"value": rng.randint(1, 100)}]
        r_store.upsert(
            result_id=rid,
            query_id=qid,
            results=res,
            freshness={"generated_at": time.time()},
            dependent_facts=deps,
            invalidation_rules=["update_rule"],
        )

    print(f"Wrote {args.count} queries to {args.queries} and results to {args.results}")


if __name__ == "__main__":
    main()

