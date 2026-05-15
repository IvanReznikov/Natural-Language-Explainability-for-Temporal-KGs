#!/usr/bin/env python3
"""M3-E3b: Utility Assessment via Real Task Performance.

This tooling exports a pool of per-domain tasks from the project JSONL dataset
and supports the 3-condition study design:
- with_explanation: show explanation text
- without_explanation: show underlying facts (gold_facts) but no explanation
- control: show only the user query

It also provides an analyzer that computes success-rate improvement, confidence
delta, time reduction, and expert agreement (if provided).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from temporal_nlg.evaluation.m3_e3 import UtilityResponse


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


def _load_predictions(path: Path) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    for obj in _iter_jsonl(path):
        rid = obj.get("id")
        if not rid:
            continue
        text = obj.get("prediction")
        if text is None:
            text = obj.get("generated_text")
        if text is None:
            text = obj.get("output")
        if text is None:
            text = obj.get("text")
        if text is None:
            continue
        preds[str(rid)] = str(text)
    return preds


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _copy_web_assets(out_dir: Path) -> None:
    src = Path(__file__).parent / "web" / "utility.html"
    if not src.exists():
        return
    (out_dir / "web").mkdir(parents=True, exist_ok=True)
    (out_dir / "web" / "utility.html").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def export_tasks(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds: Optional[Dict[str, str]] = None
    if args.predictions:
        preds = _load_predictions(Path(args.predictions))

    rows = list(_iter_jsonl(dataset_path))
    rng = random.Random(args.seed)

    # Build a pool per domain.
    by_domain: Dict[str, List[dict]] = {}
    for r in rows:
        dom = str(r.get("domain") or "unknown")
        by_domain.setdefault(dom, []).append(r)

    tasks: List[dict] = []
    for dom, group in sorted(by_domain.items()):
        k = min(len(group), args.tasks_per_domain)
        chosen = group if len(group) <= k else rng.sample(group, k)
        for r in chosen:
            rid = str(r.get("id"))
            explanation_text = None
            if preds is not None:
                explanation_text = preds.get(rid)
            if explanation_text is None:
                explanation_text = str(r.get("gold_answer") or "")

            tasks.append(
                {
                    "task_id": rid,
                    "domain": dom,
                    "prompt": str(r.get("query") or r.get("question") or ""),
                    "gold_answer": str(r.get("gold_answer") or ""),
                    "gold_facts": r.get("gold_facts") or [],
                    "explanation_text": explanation_text,
                }
            )

    tasks_path = out_dir / "m3_e3b_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)

    # Assignment plan template.
    assign_path = out_dir / "m3_e3b_assignments.csv"
    with assign_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["participant_id", "domain", "condition", "task_ids"],
        )
        w.writeheader()

    _copy_web_assets(out_dir)

    print(f"Wrote tasks: {tasks_path}")
    print(f"Wrote assignments template: {assign_path}")
    print(f"Web UI: {out_dir / 'web' / 'utility.html'}")


def _read_utility_responses(path: Path) -> List[UtilityResponse]:
    out: List[UtilityResponse] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    out.append(UtilityResponse(**row))
                except Exception:
                    continue
        return out

    for obj in _iter_jsonl(path):
        try:
            out.append(UtilityResponse(**obj))
        except Exception:
            continue
    return out


def _read_tasks(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for obj in _iter_jsonl(path):
        tid = obj.get("task_id")
        if tid:
            out[str(tid)] = obj
    return out


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = _read_tasks(Path(args.tasks))
    responses = _read_utility_responses(Path(args.responses))

    # Auto-compute success if missing (exact string match).
    coerced: List[dict] = []
    for r in responses:
        rr = r.model_copy(deep=True)
        t = tasks.get(rr.task_id)
        if t is not None and rr.success is None:
            gold = str(t.get("gold_answer") or "").strip()
            # UI stores user answer under an extra field; tolerate it.
            user_answer = getattr(rr, "answer", None)
            if user_answer is None:
                user_answer = ""
            ua = str(user_answer).strip()
            rr.success = bool(gold and ua and ua == gold)
        coerced.append(rr.model_dump())

    # Use aggregator from evaluation module.
    from temporal_nlg.evaluation.m3_e3 import aggregate_utility

    summary = aggregate_utility([UtilityResponse(**x) for x in coerced])

    # Success criteria checks.
    crit = {}
    imp = summary.get("success_improvement_vs_without")
    if imp is not None:
        crit["success_improvement_ge_0_20"] = float(imp) >= 0.20
    else:
        crit["success_improvement_ge_0_20"] = None

    conf_delta = summary.get("confidence_delta_vs_without")
    if conf_delta is not None:
        crit["confidence_delta_ge_0_3"] = float(conf_delta) >= 0.30
    else:
        crit["confidence_delta_ge_0_3"] = None

    time_red = summary.get("time_reduction_vs_without")
    if time_red is not None:
        crit["time_reduction_ge_0_15"] = float(time_red) >= 0.15
    else:
        crit["time_reduction_ge_0_15"] = None

    agree = summary.get("expert_agreement_mean")
    if agree is not None:
        crit["expert_agreement_ge_0_70"] = float(agree) >= 0.70
    else:
        crit["expert_agreement_ge_0_70"] = None

    summary["success_criteria"] = crit

    (out_dir / "m3_e3b_utility.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "m3_e3b_utility.responses.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in coerced),
        encoding="utf-8",
    )

    print(f"Wrote summary: {out_dir / 'm3_e3b_utility.summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_exp = sub.add_parser("export")
    ap_exp.add_argument("--dataset", type=str, required=True)
    ap_exp.add_argument("--predictions", type=str, default=None)
    ap_exp.add_argument("--output-dir", type=str, required=True)
    ap_exp.add_argument("--tasks-per-domain", type=int, default=50)
    ap_exp.add_argument("--seed", type=int, default=13)
    ap_exp.set_defaults(func=export_tasks)

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--tasks", type=str, required=True, help="m3_e3b_tasks.jsonl")
    ap_an.add_argument("--responses", type=str, required=True, help="Response JSONL or CSV")
    ap_an.add_argument("--output-dir", type=str, required=True)
    ap_an.set_defaults(func=analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
