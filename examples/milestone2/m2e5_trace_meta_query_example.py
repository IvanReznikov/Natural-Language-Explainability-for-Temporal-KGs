#!/usr/bin/env python3
"""Run meta-queries over the curated M2-E5 sample traces."""
import json
from pathlib import Path

from temporal_nlg.tms.meta_query import contradictions, explain_fact, influential_facts, rules_fired, why_not_fired
from temporal_nlg.tms.trace import QueryTrace


def load_traces(path: Path):
    traces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            traces.append(QueryTrace.from_dict(json.loads(line)))
    return traces


def main():
    root = Path(__file__).resolve().parents[2]
    trace_path = root / "experiments" / "m2_e5" / "output" / "small_traces.jsonl"
    traces = load_traces(trace_path)

    print({"sample_file": str(trace_path), "num_traces": len(traces)})

    for t in traces[:3]:
        rid_list = rules_fired(t)
        contrad = contradictions(t)
        infl = influential_facts(t, top_k=3)
        why_missing = why_not_fired(t, ["rule_0", "rule_1"])
        fact_to_explain = t.rule_traces[0].conclusion.get("fact_id") if t.rule_traces else None
        explanation = explain_fact(t, fact_to_explain) if fact_to_explain else None

        print({
            "query_id": t.query_id,
            "rules": rid_list,
            "contradictions": contrad,
            "influential": infl,
            "why_not": why_missing,
            "example_fact": fact_to_explain,
            "explanation": explanation,
        })


if __name__ == "__main__":
    main()
