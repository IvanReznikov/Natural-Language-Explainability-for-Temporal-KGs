#!/usr/bin/env python3
"""Trigger → store queries → reify results → mark stale (M2-E6)."""

from pathlib import Path

from temporal_nlg.tms.query_store import QueryStore
from temporal_nlg.tms.result_store import ResultStore
from temporal_nlg.tms.trigger_engine import TriggerContext, TriggerEngine, TriggerRule


def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "m2_e6_trigger_chain" / "example_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    q_path = out_dir / "queries.jsonl"
    r_path = out_dir / "results.jsonl"

    store = QueryStore(path=q_path)
    rstore = ResultStore(path=r_path)

    def pred_high_temp(ctx: TriggerContext) -> bool:
        return ctx.facts.get("temperature", 0) > 101

    def factory(ctx: TriggerContext):
        return {
            "query_id": f"q_temp_{ctx.context_id}",
            "text": "What antibiotics treat this infection?",
            "intent": "medical",
            "dependencies": ["temperature_fact"],
        }

    rules = [
        TriggerRule("r_high_temp", "High fever triggers treatment query", pred_high_temp, factory)
    ]
    engine = TriggerEngine(store)

    ctxs = [
        TriggerContext("c1", {"temperature": 102}, {}),
        TriggerContext("c2", {"temperature": 99}, {}),
    ]

    triggered_all = []
    for ctx in ctxs:
        triggered = engine.evaluate(ctx, rules)
        triggered_all.extend(triggered)

    for qid in triggered_all:
        rstore.upsert(
            result_id=f"res_{qid}",
            query_id=qid,
            results=[{"value": "demo_result"}],
            freshness={"generated_at": "now"},
            dependent_facts=["temperature_fact"],
            invalidation_rules=["rule_temp_update"],
        )

    stale = rstore.mark_stale_by_facts({"temperature_fact"})

    print(
        {
            "triggered": triggered_all,
            "stale_results": [r.result_id for r in stale],
            "active_results": [r.result_id for r in rstore.active_results()],
            "query_file": str(q_path),
            "result_file": str(r_path),
        }
    )


if __name__ == "__main__":
    main()
