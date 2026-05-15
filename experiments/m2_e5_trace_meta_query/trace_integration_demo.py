#!/usr/bin/env python3
"""Example wiring of TraceRecorder into a simple rule engine (M2-E5).

This simulates how to instrument a rule evaluation pipeline: start a trace
session, record each firing, and save the trace to JSONL.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from temporal_nlg.tms.trace import TraceRecorder


class DummyRule:
    def __init__(self, rule_id: str):
        self.rule_id = rule_id

    def name(self) -> str:
        return self.rule_id

    def apply(self, facts: Dict[str, int]) -> Dict[str, int] | None:
        # Fire when the sum of values is even; return a derived fact
        s = sum(facts.values())
        if s % 2 == 0:
            return {"fact_id": f"g_{self.rule_id}", "value": s // 2}
        return None


def run_query(query_id: str, facts: Dict[str, int], out_path: Path) -> None:
    recorder = TraceRecorder()
    rules = [DummyRule("rule_0"), DummyRule("rule_1"), DummyRule("rule_2")]

    with recorder.session(query_id, meta={"demo": True}) as trace:
        for rule in rules:
            conclusion = rule.apply(facts)
            if conclusion:
                recorder.record_rule_firing(
                    trace,
                    rule_id=rule.rule_id,
                    rule_name=rule.name(),
                    inputs=[{"fact_id": k, "value": v} for k, v in facts.items()],
                    conclusion=conclusion,
                    confidence=0.9,
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(recorder.to_json(trace), encoding="utf-8")
    print(f"Saved trace for {query_id} to {out_path}")


if __name__ == "__main__":
    run_query("demo_q1", {"a": 2, "b": 4}, Path("output/m2_e5_trace_meta_query/demo_trace.json"))

