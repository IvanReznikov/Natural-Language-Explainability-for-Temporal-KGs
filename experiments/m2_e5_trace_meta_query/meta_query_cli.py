#!/usr/bin/env python3
"""CLI for meta-queries over M2-E5 trace JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

from temporal_nlg.tms.trace import QueryTrace
from temporal_nlg.tms.meta_query import (
    rules_fired,
    why_not_fired,
    influential_facts,
    explain_fact,
    contradictions,
)


def load_traces(path: Path) -> List[QueryTrace]:
    traces: List[QueryTrace] = []
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        # Single JSON object
        traces.append(QueryTrace.from_dict(json.loads(text)))
        return traces

    for line in text.splitlines():
        if not line.strip():
            continue
        traces.append(QueryTrace.from_dict(json.loads(line)))
    return traces


def cmd_list_rules(traces: List[QueryTrace], args: argparse.Namespace):
    for t in traces:
        print(json.dumps({"query_id": t.query_id, "rules": rules_fired(t)}))


def cmd_explain_fact(traces: List[QueryTrace], args: argparse.Namespace):
    fact_id = args.fact
    for t in traces:
        explanation = explain_fact(t, fact_id)
        print(json.dumps({"query_id": t.query_id, "fact": fact_id, "explanation": explanation}))


def cmd_contradictions(traces: List[QueryTrace], args: argparse.Namespace):
    for t in traces:
        print(json.dumps({"query_id": t.query_id, "contradictions": contradictions(t)}))


def cmd_influential(traces: List[QueryTrace], args: argparse.Namespace):
    topk = args.top_k
    for t in traces:
        print(json.dumps({"query_id": t.query_id, "influential": influential_facts(t, topk)}))


def cmd_why_not(traces: List[QueryTrace], args: argparse.Namespace):
    expected = args.rule
    for t in traces:
        print(json.dumps({"query_id": t.query_id, "why_not": why_not_fired(t, expected)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meta-queries over trace JSONL")
    parser.add_argument("trace_file", type=str, help="Path to trace JSONL file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_rules = subparsers.add_parser("list-rules", help="List rules fired per query")
    sp_rules.set_defaults(func=cmd_list_rules)

    sp_explain = subparsers.add_parser("explain-fact", help="Explain how a fact was derived")
    sp_explain.add_argument("fact", type=str, help="Fact ID to explain")
    sp_explain.set_defaults(func=cmd_explain_fact)

    sp_contra = subparsers.add_parser("contradictions", help="Detect contradictions")
    sp_contra.set_defaults(func=cmd_contradictions)

    sp_infl = subparsers.add_parser("influential", help="Most influential facts")
    sp_infl.add_argument("--top-k", type=int, default=5)
    sp_infl.set_defaults(func=cmd_influential)

    sp_why = subparsers.add_parser("why-not", help="Why rules did not fire")
    sp_why.add_argument("rule", nargs="+", help="Rule IDs expected to fire")
    sp_why.set_defaults(func=cmd_why_not)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    traces = load_traces(Path(args.trace_file))
    args.func(traces, args)


if __name__ == "__main__":
    main()
