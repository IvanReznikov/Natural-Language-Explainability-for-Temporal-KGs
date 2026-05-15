#!/usr/bin/env python3
"""Demo: mark results stale when dependent facts change (M2-E6)."""
from __future__ import annotations

from pathlib import Path

from temporal_nlg.tms.result_store import ResultStore


def main():
    store_path = Path("output/m2_e6_query_store_triggers/change_results.jsonl")
    store = ResultStore(path=store_path)

    # Seed two results with different dependencies
    store.upsert(
        result_id="r_a",
        query_id="q_a",
        results=[{"value": 1}],
        dependent_facts=["f1", "f2"],
        invalidation_rules=["rule_x"],
    )
    store.upsert(
        result_id="r_b",
        query_id="q_b",
        results=[{"value": 2}],
        dependent_facts=["f3"],
        invalidation_rules=["rule_y"],
    )

    # Simulate fact change and rule firing
    touched = {"f2"}
    fired_rules = {"rule_y"}

    changed_facts = store.mark_stale_by_facts(touched)
    changed_rules = store.mark_stale_by_rules(fired_rules)

    print({
        "stale_by_facts": [r.result_id for r in changed_facts],
        "stale_by_rules": [r.result_id for r in changed_rules],
        "active": [r.result_id for r in store.active_results()],
    })
    print(f"Results persisted at {store_path}")


if __name__ == "__main__":
    main()

