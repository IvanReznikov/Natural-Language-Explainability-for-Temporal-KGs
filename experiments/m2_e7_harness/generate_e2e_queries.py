#!/usr/bin/env python3
"""Generate a synthetic E2E query corpus for M2-E7."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

INTENTS = ["medical", "financial", "historical", "science"]


def main():
    parser = argparse.ArgumentParser(description="Generate E2E query corpus")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=str, default="experiments/m2_e7_harness/input/queries.jsonl")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.count):
            intent = INTENTS[i % len(INTENTS)]
            payload = {
                "query_id": f"q{i}",
                "text": f"Example query {i} about {intent}",
                "intent": intent,
                "expected": "ok",  # marker for basic success expectation
            }
            f.write(json.dumps(payload) + "\n")
    print(f"Wrote {args.count} queries to {out_path}")


if __name__ == "__main__":
    main()

