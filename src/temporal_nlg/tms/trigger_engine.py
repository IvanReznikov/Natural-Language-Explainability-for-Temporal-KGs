"""Rule-based query triggering for M2-E6."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .query_store import QueryStore


@dataclass
class TriggerContext:
    context_id: str
    facts: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass
class TriggerRule:
    rule_id: str
    description: str
    predicate: Callable[[TriggerContext], bool]
    query_factory: Callable[[TriggerContext], Dict[str, Any]]


class TriggerEngine:
    def __init__(self, store: QueryStore, max_latency_ms: float = 500.0):
        self.store = store
        self.max_latency_ms = max_latency_ms

    def evaluate(self, ctx: TriggerContext, rules: List[TriggerRule]) -> List[str]:
        triggered: List[str] = []
        start = time.perf_counter()
        for rule in rules:
            if rule.predicate(ctx):
                payload = rule.query_factory(ctx)
                qid = payload.get("query_id") or f"trig_{rule.rule_id}_{ctx.context_id}"
                self.store.upsert(
                    query_id=qid,
                    text=payload.get("text", ""),
                    intent=payload.get("intent", "unknown"),
                    meta=payload.get("meta", {}),
                    dependencies=payload.get("dependencies", []),
                    user_id=payload.get("user_id"),
                )
                triggered.append(qid)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > self.max_latency_ms:
            # Keep this lightweight; callers can log or handle
            pass
        return triggered


__all__ = ["TriggerEngine", "TriggerRule", "TriggerContext"]
