"""Trace instrumentation for rule-based query workflows (M2-E5).

Captures rule firing details, tracks instrumentation overhead, and supports
sampling to keep tracing lightweight.
"""
from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class RuleTrace:
    """Represents a single rule firing."""

    rule_id: str
    rule_name: str
    inputs: List[Dict[str, Any]]
    conclusion: Dict[str, Any]
    fired_at: float
    confidence: float = 1.0
    latency_ms: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleTrace":
        return cls(
            rule_id=data.get("rule_id", ""),
            rule_name=data.get("rule_name", ""),
            inputs=data.get("inputs", []) or [],
            conclusion=data.get("conclusion", {}) or {},
            fired_at=data.get("fired_at", 0.0),
            confidence=data.get("confidence", 1.0),
            latency_ms=data.get("latency_ms"),
            meta=data.get("meta", {}) or {},
        )


@dataclass
class QueryTrace:
    """Aggregates all rule firings for a query execution."""

    query_id: str
    started_at: float
    meta: Dict[str, Any] = field(default_factory=dict)
    rule_traces: List[RuleTrace] = field(default_factory=list)
    completed_at: Optional[float] = None
    instrumentation_overhead_ms: float = 0.0
    dropped: bool = False
    over_budget: bool = False

    def duration_ms(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms(),
            "instrumentation_overhead_ms": self.instrumentation_overhead_ms,
            "dropped": self.dropped,
            "over_budget": self.over_budget,
            "meta": self.meta,
            "rule_traces": [rt.to_dict() for rt in self.rule_traces],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryTrace":
        qt = cls(
            query_id=data.get("query_id", ""),
            started_at=data.get("started_at", 0.0),
            meta=data.get("meta", {}) or {},
            rule_traces=[RuleTrace.from_dict(rt) for rt in data.get("rule_traces", [])],
            completed_at=data.get("completed_at"),
            instrumentation_overhead_ms=data.get("instrumentation_overhead_ms", 0.0),
            dropped=data.get("dropped", False),
            over_budget=data.get("over_budget", False),
        )
        return qt


class TraceRecorder:
    """Captures trace events with sampling and overhead guardrails."""

    def __init__(
        self,
        sampling_rate: float = 1.0,
        max_overhead_ms: float = 5.0,
        time_fn: Optional[Callable[[], float]] = None,
        perf_fn: Optional[Callable[[], float]] = None,
        rand_fn: Optional[Callable[[], float]] = None,
    ):
        if sampling_rate < 0 or sampling_rate > 1:
            raise ValueError("sampling_rate must be within [0, 1]")
        self.sampling_rate = sampling_rate
        self.max_overhead_ms = max_overhead_ms
        self.time_fn = time_fn or time.time
        self.perf_fn = perf_fn or time.perf_counter
        self.rand_fn = rand_fn or random.random

    def start_query(self, query_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> QueryTrace:
        sampled = self.rand_fn() <= self.sampling_rate
        qid = query_id or f"q_{uuid.uuid4().hex[:8]}"
        return QueryTrace(query_id=qid, started_at=self.time_fn(), meta=meta or {}, dropped=not sampled)

    def complete_query(self, trace: QueryTrace) -> QueryTrace:
        if trace.completed_at is None:
            trace.completed_at = self.time_fn()
        return trace

    def record_rule_firing(
        self,
        trace: QueryTrace,
        *,
        rule_id: str,
        rule_name: str,
        inputs: List[Dict[str, Any]],
        conclusion: Dict[str, Any],
        confidence: float = 1.0,
        latency_ms: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[RuleTrace]:
        if trace.dropped:
            return None

        start_perf = self.perf_fn()
        rule_trace = RuleTrace(
            rule_id=rule_id,
            rule_name=rule_name,
            inputs=inputs,
            conclusion=conclusion,
            fired_at=self.time_fn(),
            confidence=confidence,
            latency_ms=latency_ms,
            meta=meta or {},
        )
        trace.rule_traces.append(rule_trace)
        self._accumulate_overhead(trace, start_perf)
        return rule_trace

    def to_json(self, trace: QueryTrace) -> str:
        return json.dumps(trace.to_dict(), indent=2)

    def dump(self, trace: QueryTrace, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(trace))
        return path

    @contextmanager
    def session(self, query_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
        """Context manager that auto-completes the query trace."""
        trace = self.start_query(query_id, meta)
        try:
            yield trace
        finally:
            self.complete_query(trace)

    def _accumulate_overhead(self, trace: QueryTrace, start_perf: float):
        elapsed_ms = (self.perf_fn() - start_perf) * 1000.0
        trace.instrumentation_overhead_ms += elapsed_ms
        trace.over_budget = trace.instrumentation_overhead_ms > self.max_overhead_ms


__all__ = ["RuleTrace", "QueryTrace", "TraceRecorder"]
