#!/usr/bin/env python3
"""M3-E4b: Consistency Maintenance Under Fact Revision."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional

from temporal_nlg.evaluation.m3_e4 import (
    ConsistencyFact,
    ConsistencyRevision,
    ConsistencyResult,
    aggregate_consistency,
)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def export_revisions(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_jsonl(dataset))
    rng = random.Random(args.seed)

    if args.n_facts <= 0:
        raise SystemExit("--n-facts must be > 0")

    sample = rows if len(rows) <= args.n_facts else rng.sample(rows, args.n_facts)

    facts: List[dict] = []
    revisions: List[dict] = []
    rev_types = ["date_correction", "add_causal", "contradiction", "removal"]

    for idx, r in enumerate(sample):
        rid = str(r.get("id") or f"row_{idx}")
        fact = ConsistencyFact(
            fact_id=rid,
            domain=str(r.get("domain") or "unknown"),
            query=str(r.get("query") or r.get("question") or ""),
            base_explanation=str(r.get("gold_answer") or ""),
            gold_facts=r.get("gold_facts"),
        )
        facts.append(fact.model_dump())

        for rt in rev_types:
            revisions.append(
                ConsistencyRevision(
                    revision_id=f"{rid}.{rt}",
                    fact_id=rid,
                    revision_type=rt,  # placeholder delta to be filled later
                    delta={},
                ).model_dump()
            )

    facts_path = out_dir / "m3_e4b_facts.jsonl"
    revs_path = out_dir / "m3_e4b_revisions.jsonl"
    _write_jsonl(facts_path, facts)
    _write_jsonl(revs_path, revisions)

    template_path = out_dir / "m3_e4b_results_template.csv"
    with template_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "revision_id",
                "method",
                "update_accuracy",
                "contradiction_detected",
                "coherence_rating_1_5",
                "resolution_time_sec",
            ],
        )
        w.writeheader()
        for r in revisions:
            w.writerow(
                {
                    "revision_id": r["revision_id"],
                    "method": "",
                    "update_accuracy": "",
                    "contradiction_detected": "",
                    "coherence_rating_1_5": "",
                    "resolution_time_sec": "",
                }
            )

    print(f"Wrote facts: {facts_path}")
    print(f"Wrote revisions: {revs_path}")
    print(f"Wrote results template: {template_path}")


def _read_results(path: Path) -> List[ConsistencyResult]:
    out: List[ConsistencyResult] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    out.append(ConsistencyResult(**row))
                except Exception:
                    continue
        return out

    for obj in _iter_jsonl(path):
        try:
            out.append(ConsistencyResult(**obj))
        except Exception:
            continue
    return out


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _read_results(Path(args.results))
    summary = aggregate_consistency(results)

    (out_dir / "m3_e4b_consistency.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote summary: {out_dir / 'm3_e4b_consistency.summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_exp = sub.add_parser("export")
    ap_exp.add_argument("--dataset", type=str, required=True)
    ap_exp.add_argument("--output-dir", type=str, required=True)
    ap_exp.add_argument("--n-facts", type=int, default=1000)
    ap_exp.add_argument("--seed", type=int, default=13)
    ap_exp.set_defaults(func=export_revisions)

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--results", type=str, required=True, help="CSV or JSONL results file")
    ap_an.add_argument("--output-dir", type=str, required=True)
    ap_an.set_defaults(func=analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
