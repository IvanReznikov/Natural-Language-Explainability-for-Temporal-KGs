#!/usr/bin/env python3
"""Demonstrate trigger engine issuing queries (M2-E6)."""

from __future__ import annotations

from pathlib import Path

from temporal_nlg.tms.query_store import QueryStore
from temporal_nlg.tms.trigger_engine import TriggerEngine, TriggerRule, TriggerContext


def main():
    store = QueryStore(path=Path("output/m2_e6_query_store_triggers/trigger_queries.jsonl"))

    def pred_high_temp(ctx: TriggerContext) -> bool:
        return ctx.facts.get("temperature", 0) > 101

    def factory(ctx: TriggerContext):
        return {
            "query_id": f"q_temp_{ctx.context_id}",
            "text": "What antibiotics treat this infection?",
            "intent": "medical",
            "dependencies": ["temperature_fact"],
        }

    def pred_stock_drop(ctx: TriggerContext) -> bool:
        return ctx.facts.get("stock_drop", 0) > 0.1

    def factory_stock(ctx: TriggerContext):
        return {
            "query_id": f"q_stock_{ctx.context_id}",
            "text": "Explain the stock decline",
            "intent": "financial",
            "dependencies": ["market_fact"],
        }

    rules = [
        TriggerRule("r_high_temp", "High fever triggers treatment query", pred_high_temp, factory),
        TriggerRule("r_stock", "Stock drop triggers analysis", pred_stock_drop, factory_stock),
    ]

    engine = TriggerEngine(store)
    ctxs = [
        TriggerContext("c1", {"temperature": 102}, {}),
        TriggerContext("c2", {"stock_drop": 0.15}, {}),
        TriggerContext("c3", {"temperature": 99}, {}),
    ]

    for ctx in ctxs:
        triggered = engine.evaluate(ctx, rules)
        print({"context": ctx.context_id, "triggered": triggered})

    print(f"Stored {store.stats()['count']} triggered queries at {store.path}")


if __name__ == "__main__":
    main()
