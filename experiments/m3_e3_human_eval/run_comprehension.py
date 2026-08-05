#!/usr/bin/env python3
"""M3-E3a: Comprehension Assessment Framework.

This script supports two steps:
1) export: sample explanation items and produce a task JSONL + a minimal web UI.
2) analyze: ingest response JSONL/CSV and compute the M3-E3a aggregate metrics.

The web UI is intentionally lightweight: it reads the exported tasks JSONL and
lets participants download a response JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from temporal_nlg.evaluation.m3_e3 import (
    ComprehensionResponse,
    ComprehensionTask,
    ExplanationItem,
    bucket_from_time_scope,
    aggregate_comprehension,
    score_responses_against_tasks,
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


def _extract_years_from_gold_facts(gold_facts: Any) -> List[int]:
    years: List[int] = []
    if not isinstance(gold_facts, list):
        return years
    for f in gold_facts:
        if not isinstance(f, dict):
            continue
        for k in ("start", "end", "date", "timestamp", "time"):
            v = f.get(k)
            if not v:
                continue
            s = str(v)
            if len(s) >= 4 and s[:4].isdigit():
                y = int(s[:4])
                if 1000 <= y <= 2999:
                    years.append(y)
    # Dedup, preserve order
    out: List[int] = []
    seen = set()
    for y in years:
        if y in seen:
            continue
        seen.add(y)
        out.append(y)
    return out


def _extract_entities_from_gold_facts(gold_facts: Any) -> List[str]:
    ents: List[str] = []
    if not isinstance(gold_facts, list):
        return ents
    for f in gold_facts:
        if not isinstance(f, dict):
            continue
        for k in ("subject", "object", "value"):
            v = f.get(k)
            if not v:
                continue
            s = str(v).strip()
            if not s:
                continue
            ents.append(s)
    out: List[str] = []
    seen = set()
    for e in ents:
        el = e.lower()
        if el in seen:
            continue
        seen.add(el)
        out.append(e)
    return out


def _make_questions(record: dict, explanation_id: str, seed: int, n_questions: int) -> List[dict]:
    """Heuristic question generation.

    Goal: provide a usable default question set for MCQ/fill-in-blank while still
    allowing rubric-scored questions (timeline/inference).
    """

    rng = random.Random(seed)
    bucket = bucket_from_time_scope(record.get("time_scope"))

    years = _extract_years_from_gold_facts(record.get("gold_facts"))
    ents = _extract_entities_from_gold_facts(record.get("gold_facts"))

    questions: List[dict] = []

    def _qid(i: int) -> str:
        return f"{explanation_id}.q{i:02d}"

    i = 1

    # MCQ: timestamp/year where possible.
    if years and i <= n_questions:
        y = years[0]
        distractors = sorted({y, y - 1, y + 1, y - 5, y + 5})
        opts = [str(v) for v in distractors if 1000 <= v <= 2999]
        rng.shuffle(opts)
        questions.append(
            {
                "question_id": _qid(i),
                "question_type": "mcq",
                "prompt": "Which year best matches the key time reference in the explanation?",
                "options": opts,
                "correct_answer": str(y),
            }
        )
        i += 1

    # Fill-in-blank: entity mention.
    if ents and i <= n_questions:
        e = ents[0]
        questions.append(
            {
                "question_id": _qid(i),
                "question_type": "fill_blank",
                "prompt": "Fill in the blank with an entity mentioned in the explanation: ________",
                "correct_answer": str(e),
                "rubric": "Exact match expected (case-sensitive by default).",
            }
        )
        i += 1

    # Sequence bucket: timeline drawing (rubric).
    if bucket == "sequence" and i <= n_questions:
        questions.append(
            {
                "question_id": _qid(i),
                "question_type": "timeline",
                "prompt": "List the events in chronological order (earliest → latest).",
                "rubric": "Score 1 if ordering is correct; partial credit allowed if scorer provides it.",
            }
        )
        i += 1

    # Causal bucket: inference question (rubric).
    if bucket == "causal" and i <= n_questions:
        questions.append(
            {
                "question_id": _qid(i),
                "question_type": "inference",
                "prompt": "Based on the explanation, what is the most plausible effect if the key cause did NOT happen?",
                "rubric": "Rubric-scored by expert/annotator; provide 0..1 in response file.",
            }
        )
        i += 1

    # Top up with generic MCQs if needed.
    while i <= n_questions:
        questions.append(
            {
                "question_id": _qid(i),
                "question_type": "mcq",
                "prompt": "Which statement is most consistent with the explanation?",
                "options": [
                    "Statement A (author this)",
                    "Statement B (author this)",
                    "Statement C (author this)",
                    "Statement D (author this)",
                ],
                "correct_answer": "Statement A (author this)",
                "rubric": "Replace placeholder statements and set correct_answer.",
            }
        )
        i += 1

    return questions


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _copy_web_assets(out_dir: Path) -> None:
    src = Path(__file__).parent / "web" / "comprehension.html"
    if not src.exists():
        return
    (out_dir / "web").mkdir(parents=True, exist_ok=True)
    (out_dir / "web" / "comprehension.html").write_text(
        src.read_text(encoding="utf-8"), encoding="utf-8"
    )


def export_tasks(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds: Optional[Dict[str, str]] = None
    if args.predictions:
        preds = _load_predictions(Path(args.predictions))

    rows = list(_iter_jsonl(dataset_path))

    rng = random.Random(args.seed)
    if args.n_items <= 0:
        raise SystemExit("--n-items must be > 0")
    if len(rows) <= args.n_items:
        sample = rows
    else:
        sample = rng.sample(rows, args.n_items)

    tasks_out: List[dict] = []
    for idx, r in enumerate(sample):
        rid = str(r.get("id") or f"row_{idx}")
        explanation_id = rid
        explanation_text = None
        if preds is not None:
            explanation_text = preds.get(rid)
        if explanation_text is None:
            explanation_text = str(r.get("gold_answer") or "")

        item = ExplanationItem(
            explanation_id=explanation_id,
            domain=str(r.get("domain") or "unknown"),
            bucket=bucket_from_time_scope(r.get("time_scope")),
            query=str(r.get("query") or r.get("question") or ""),
            explanation_text=explanation_text,
            gold_context=r.get("gold_facts"),
        )

        questions = _make_questions(
            r,
            explanation_id=explanation_id,
            seed=args.seed + idx,
            n_questions=args.questions_per_item,
        )
        task = ComprehensionTask(item=item, questions=[q for q in questions])
        tasks_out.append(task.model_dump())

    tasks_path = out_dir / "m3_e3a_tasks.jsonl"
    _write_jsonl(tasks_path, tasks_out)

    # Response template (CSV) for manual collection or sanity checking.
    template_path = out_dir / "m3_e3a_response_template.csv"
    with template_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "explanation_id",
                "question_id",
                "question_type",
                "answer",
                "score",
                "response_time_sec",
                "domain",
                "bucket",
            ],
        )
        w.writeheader()
        for t in tasks_out:
            it = t["item"]
            for q in t["questions"]:
                w.writerow(
                    {
                        "participant_id": "",
                        "explanation_id": it["explanation_id"],
                        "question_id": q["question_id"],
                        "question_type": q["question_type"],
                        "answer": "",
                        "score": "",
                        "response_time_sec": "",
                        "domain": it.get("domain"),
                        "bucket": it.get("bucket"),
                    }
                )

    _copy_web_assets(out_dir)

    print(f"Wrote tasks: {tasks_path}")
    print(f"Wrote response template: {template_path}")
    print(f"Web UI: {out_dir / 'web' / 'comprehension.html'}")


def _read_responses(path: Path) -> List[ComprehensionResponse]:
    if path.suffix.lower() == ".csv":
        out: List[ComprehensionResponse] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    out.append(ComprehensionResponse(**row))
                except Exception:
                    continue
        return out

    out = []
    for obj in _iter_jsonl(path):
        try:
            out.append(ComprehensionResponse(**obj))
        except Exception:
            continue
    return out


def _read_tasks(path: Path) -> List[ComprehensionTask]:
    out: List[ComprehensionTask] = []
    for obj in _iter_jsonl(path):
        try:
            out.append(ComprehensionTask(**obj))
        except Exception:
            continue
    return out


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = _read_tasks(Path(args.tasks))
    responses = _read_responses(Path(args.responses))

    scored = score_responses_against_tasks(responses, tasks)
    summary = aggregate_comprehension(scored)

    # Convenience: compute per-bucket minimum and overall thresholds.
    overall_ok = None
    if summary.get("overall_accuracy") is not None:
        overall_ok = float(summary["overall_accuracy"]) >= 0.80

    per_bucket_ok: Dict[str, Optional[bool]] = {}
    for b, v in (summary.get("accuracy_by_bucket") or {}).items():
        if v is None:
            per_bucket_ok[b] = None
        else:
            per_bucket_ok[b] = float(v) >= 0.70

    summary["success_criteria"] = {
        "overall_accuracy_ge_0_80": overall_ok,
        "per_bucket_accuracy_ge_0_70": per_bucket_ok,
        "mean_response_time_sec_le_30": (summary.get("mean_response_time_sec") is not None)
        and (float(summary["mean_response_time_sec"]) <= 30.0),
    }

    (out_dir / "m3_e3a_comprehension.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Persist scored responses for audit.
    scored_path = out_dir / "m3_e3a_comprehension.scored_responses.jsonl"
    _write_jsonl(scored_path, [r.model_dump() for r in scored])

    print(f"Wrote summary: {out_dir / 'm3_e3a_comprehension.summary.json'}")
    print(f"Wrote scored responses: {scored_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_exp = sub.add_parser("export")
    ap_exp.add_argument("--dataset", type=str, required=True)
    ap_exp.add_argument("--predictions", type=str, default=None)
    ap_exp.add_argument("--output-dir", type=str, required=True)
    ap_exp.add_argument("--n-items", type=int, default=50, help="Number of explanations to export")
    ap_exp.add_argument("--questions-per-item", type=int, default=5)
    ap_exp.add_argument("--seed", type=int, default=13)
    ap_exp.set_defaults(func=export_tasks)

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--tasks", type=str, required=True, help="m3_e3a_tasks.jsonl")
    ap_an.add_argument("--responses", type=str, required=True, help="Response JSONL or CSV")
    ap_an.add_argument("--output-dir", type=str, required=True)
    ap_an.set_defaults(func=analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
