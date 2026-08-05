#!/usr/bin/env python3
"""Run end-to-end M2-E7 harness over a query corpus and traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def build_results_from_traces(traces: List[dict]) -> List[dict]:
    results: List[dict] = []
    for tr in traces:
        qid = tr.get("query_id")
        raw_rules = tr.get("rule_traces", []) or []
        expanded_rules: List[dict] = []
        intent_val = None

        for rule in raw_rules:
            if not isinstance(rule, dict):
                continue
            expanded_rules.append(rule)

            conc = rule.get("conclusion") if isinstance(rule, dict) else None
            if isinstance(conc, dict) and conc.get("fact_id") == "intent":
                intent_val = conc.get("value", intent_val)

            extras = []
            meta = rule.get("meta") if isinstance(rule, dict) else None
            if isinstance(meta, dict):
                extras = meta.get("extra_conclusions") or []
            for extra in extras:
                if not isinstance(extra, dict):
                    continue
                expanded_rules.append(
                    {
                        "rule_id": f"{rule.get('rule_id', 'r')}_extra",
                        "rule_name": rule.get("rule_name", "extra"),
                        "inputs": rule.get("inputs", []),
                        "conclusion": extra,
                        "fired_at": rule.get("fired_at"),
                        "confidence": rule.get("confidence", 1.0),
                        "latency_ms": rule.get("latency_ms"),
                        "meta": {"gen": "synthetic_extra"},
                    }
                )
                if extra.get("fact_id") == "intent":
                    intent_val = extra.get("value", intent_val)

        results.append({"query_id": qid, "intent": intent_val, "rule_traces": expanded_rules})
    return results


def _collect_fact_ids(rule_traces: List[dict]) -> List[str]:
    fact_ids: List[str] = []
    for rt in rule_traces:
        conc = rt.get("conclusion") if isinstance(rt, dict) else None
        fid = conc.get("fact_id") if isinstance(conc, dict) else None
        if fid:
            fact_ids.append(fid)
    return fact_ids


def build_expected_map(
    queries: List[dict],
    *,
    default_required_facts: Iterable[str] = ("intent",),
    default_max_latency_ms: float | None = 15.0,
) -> dict:
    expected = {}
    for q in queries:
        qid = q.get("query_id")
        if not qid:
            continue
        expected[qid] = {
            "intent": q.get("intent"),
            "required_facts": q.get("required_facts", list(default_required_facts)),
            "max_latency_ms": q.get("max_latency_ms", default_max_latency_ms),
        }
    return expected


# util helpers


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def eval_results(results: List[dict], expected_map: dict) -> Tuple[int, int, List[dict]]:
    ok, total = 0, len(results)
    failures = []
    for r in results:
        qid = r.get("query_id")
        exp = expected_map.get(qid)
        if exp is None:
            failures.append({"query_id": qid, "reason": "missing expected entry"})
            continue

        reasons = []

        exp_intent = exp.get("intent")
        if exp_intent is not None and r.get("intent") != exp_intent:
            reasons.append(f"intent mismatch: got {r.get('intent')} expected {exp_intent}")

        required_facts = exp.get("required_facts") or []
        if required_facts:
            fact_ids = set(_collect_fact_ids(r.get("rule_traces", []) or []))
            missing = [fid for fid in required_facts if fid not in fact_ids]
            if missing:
                reasons.append(f"missing facts: {missing}")

        max_latency = exp.get("max_latency_ms")
        if max_latency is not None:
            over = [
                rt
                for rt in r.get("rule_traces", []) or []
                if isinstance(rt, dict)
                and rt.get("latency_ms")
                and rt.get("latency_ms") > max_latency
            ]
            if over:
                reasons.append(f"latency_ms over {max_latency}")

        if reasons:
            failures.append({"query_id": qid, "reason": "; ".join(reasons)})
        else:
            ok += 1
    return ok, total, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M2-E7 E2E harness")
    parser.add_argument(
        "--queries", type=str, default="experiments/m2_e7_harness/input/queries.jsonl"
    )
    parser.add_argument("--trace", type=str, default="experiments/m2_e7_harness/input/trace.jsonl")
    parser.add_argument("--output", type=str, default="output/m2_e7_harness/results.jsonl")
    parser.add_argument("--report", type=str, default="output/m2_e7_harness/report.json")
    args = parser.parse_args()

    queries = load_jsonl(Path(args.queries))
    traces = load_jsonl(Path(args.trace))

    results = build_results_from_traces(traces)

    save_jsonl(Path(args.output), results)

    expected = build_expected_map(queries)
    ok, total, failures = eval_results(results, expected)

    report = {
        "total": total,
        "ok": ok,
        "fail": len(failures),
        "failures": failures,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
