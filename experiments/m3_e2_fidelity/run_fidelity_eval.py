#!/usr/bin/env python3
"""Run M3-E2 fidelity evaluation over temporal_graph-style JSONL.

Usage (gold-as-pred baseline):
  python experiments/m3_e2_fidelity/run_fidelity_eval.py \
    --dataset data/jsonls/temporal_graph.jsonl \
    --output-dir output/m3_e2_fidelity

Usage (with model predictions):
  python experiments/m3_e2_fidelity/run_fidelity_eval.py \
    --dataset data/jsonls/temporal_graph.jsonl \
    --predictions path/to/preds.jsonl \
    --output-dir output/m3_e2_fidelity

Predictions JSONL format:
- Must contain an `id` matching the dataset record `id`.
- Must contain the generated explanation under one of: `prediction`, `generated_text`, `output`, `text`.
"""

from __future__ import annotations

import argparse
import json
import random
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from temporal_nlg.evaluation.m3_e2_fidelity import M3E2FidelityEvaluator, aggregate_by_bucket
from temporal_nlg.evaluation.m3_e2_human_loop import M3E2HumanLoopLLMScorer, coerce_unit_score


def _load_dotenv_if_present(repo_root: Path) -> None:
    """Best-effort .env loader.

    We avoid extra dependencies (python-dotenv) and do not print secrets.
    Only sets env vars that are not already present in the process.
    """

    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            # Strip matching quotes.
            if (len(value) >= 2) and ((value[0] == value[-1]) and value[0] in {"\"", "'"}):
                value = value[1:-1]

            if key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Silent best-effort; caller may still provide env vars via shell.
        return


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


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


def _bucket(time_scope: str) -> str:
    t = str(time_scope or "").strip().lower()
    if t in {"point", "timestamp", "date"}:
        return "point"
    if t in {"interval", "range", "duration"}:
        return "interval"
    if t in {"sequence", "ordering", "order"}:
        return "sequence"
    if t in {"causal", "causality"}:
        return "causal"
    if t in {"overlap"}:
        return "overlap"
    return "other"


def _sample_stratified(rows: Sequence[dict], n: int, seed: int, stratify_domain: bool) -> List[dict]:
    if n <= 0:
        return []

    rng = random.Random(seed)

    if not stratify_domain:
        if len(rows) <= n:
            return list(rows)
        return rng.sample(list(rows), n)

    by_domain: Dict[str, List[dict]] = {}
    for r in rows:
        by_domain.setdefault(str(r.get("domain") or "unknown"), []).append(r)

    domains = sorted(by_domain.keys())
    if not domains:
        return []

    base = n // len(domains)
    rem = n % len(domains)

    sampled: List[dict] = []
    for idx, dom in enumerate(domains):
        k = base + (1 if idx < rem else 0)
        group = by_domain[dom]
        if not group:
            continue
        if len(group) <= k:
            sampled.extend(group)
        else:
            sampled.extend(rng.sample(group, k))

    # If we under-filled due to small groups, top up globally.
    if len(sampled) < n:
        remaining = [r for r in rows if r not in sampled]
        need = n - len(sampled)
        if remaining:
            sampled.extend(rng.sample(remaining, min(need, len(remaining))))

    return sampled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Path to temporal_graph JSONL")
    ap.add_argument("--predictions", type=str, default=None, help="Optional predictions JSONL with {id, prediction}")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--n-per-type", type=int, default=100)
    ap.add_argument(
        "--point-per-domain",
        type=int,
        default=None,
        help="If set, sample this many point examples per domain (e.g., 20). Overrides --n-per-type for point.",
    )
    ap.add_argument(
        "--max-domains",
        type=int,
        default=5,
        help="When using --point-per-domain, limit point sampling to the top-N most frequent domains (default: 5).",
    )
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-domain-stratify", action="store_true", help="Disable domain stratification")

    ap.add_argument(
        "--human-loop",
        type=str,
        default="none",
        choices=["none", "human", "llm"],
        help="Fill judgement-only metrics via human annotation or LLM proxy.",
    )
    ap.add_argument(
        "--human-loop-input",
        type=str,
        default=None,
        help="JSONL with {id, ambiguity_resolution, causal_link_accuracy, confidence_calibration, narrative_consistency, notes}.",
    )
    ap.add_argument(
        "--human-loop-export",
        type=str,
        default=None,
        help="Where to write the human annotation task JSONL (defaults inside output-dir).",
    )
    ap.add_argument("--llm-model", type=str, default="gpt-4.1-nano")
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-max-tokens", type=int, default=200)

    args = ap.parse_args()

    # Ensure OPENAI_API_KEY from .env is available for --human-loop llm.
    _load_dotenv_if_present(Path.cwd())

    dataset_path = Path(args.dataset)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds: Optional[Dict[str, str]] = None
    if args.predictions:
        preds = _load_predictions(Path(args.predictions))

    evaluator = M3E2FidelityEvaluator()

    # Stream the dataset and bucket rows.
    buckets: Dict[str, List[dict]] = {"point": [], "interval": [], "sequence": [], "causal": [], "overlap": [], "other": []}
    for r in _iter_jsonl(dataset_path):
        buckets[_bucket(r.get("time_scope"))].append(r)

    selected: List[dict] = []

    # Point: optionally enforce per-domain quota (per M3-E2a).
    if args.point_per_domain is not None and not args.no_domain_stratify:
        rng = random.Random(args.seed + 101)
        by_domain: Dict[str, List[dict]] = {}
        for r in buckets["point"]:
            by_domain.setdefault(str(r.get("domain") or "unknown"), []).append(r)

        # M3-E2a expects 20 examples per domain for ~5 domains. If the dataset has
        # many domains, default to the top-N domains to keep the point set size
        # around 100 unless explicitly overridden.
        max_domains = max(0, int(args.max_domains))
        domain_items = sorted(by_domain.items(), key=lambda kv: len(kv[1]), reverse=True)
        if max_domains:
            domain_items = domain_items[:max_domains]

        point_selected: List[dict] = []
        for dom, group in domain_items:
            k = max(0, int(args.point_per_domain))
            if k == 0:
                continue
            if len(group) <= k:
                point_selected.extend(group)
            else:
                point_selected.extend(rng.sample(group, k))
        selected.extend(point_selected)
        point_n = len(point_selected)
    else:
        chosen = _sample_stratified(
            buckets["point"],
            n=args.n_per_type,
            seed=args.seed + hash("point") % 10000,
            stratify_domain=not args.no_domain_stratify,
        )
        selected.extend(chosen)
        point_n = len(chosen)

    # Remaining buckets use n-per-type sampling.
    for b in ["interval", "sequence", "causal"]:
        chosen = _sample_stratified(
            buckets[b],
            n=args.n_per_type,
            seed=args.seed + hash(b) % 10000,
            stratify_domain=not args.no_domain_stratify,
        )
        selected.extend(chosen)

    per_item_rows: List[dict] = []
    task_rows: List[dict] = []
    missing_pred = 0
    for r in selected:
        rid = str(r.get("id"))
        if preds is None:
            pred_text = str(r.get("gold_answer") or "")
        else:
            pred_text = preds.get(rid)
            if pred_text is None:
                missing_pred += 1
                pred_text = ""

        metrics = evaluator.evaluate_example(r, prediction_text=pred_text)
        metrics["has_prediction"] = bool(pred_text.strip())
        per_item_rows.append(metrics)

        if args.human_loop == "human":
            task_rows.append(
                {
                    "id": rid,
                    "bucket": metrics.get("bucket"),
                    "domain": r.get("domain"),
                    "query": r.get("query"),
                    "prediction": pred_text,
                    "gold_facts": r.get("gold_facts"),
                    "ambiguity_resolution": None,
                    "causal_link_accuracy": None,
                    "confidence_calibration": None,
                    "narrative_consistency": None,
                    "notes": None,
                }
            )

    # Human loop: export tasks.
    human_export_path: Optional[Path] = None
    if args.human_loop == "human":
        human_export_path = Path(args.human_loop_export) if args.human_loop_export else (out_dir / "m3_e2_human_tasks.jsonl")
        with human_export_path.open("w", encoding="utf-8") as f:
            for row in task_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Merge human annotations if provided.
    merged_human = 0
    if args.human_loop_input:
        annotations: Dict[str, dict] = {}
        for obj in _iter_jsonl(Path(args.human_loop_input)):
            if obj.get("id"):
                annotations[str(obj["id"])] = obj

        for row in per_item_rows:
            rid = str(row.get("id"))
            ann = annotations.get(rid)
            if not ann:
                continue
            for k in ["ambiguity_resolution", "causal_link_accuracy", "confidence_calibration", "narrative_consistency"]:
                if k in ann:
                    row[k] = coerce_unit_score(ann.get(k))
            if "notes" in ann:
                row["human_notes"] = ann.get("notes")
            merged_human += 1

    # LLM loop: fill judgement metrics via LangChain.
    llm_scored = 0
    if args.human_loop == "llm":
        llm = M3E2HumanLoopLLMScorer(model=args.llm_model, temperature=args.llm_temperature, max_tokens=args.llm_max_tokens)
        id_to_record = {str(r.get("id")): r for r in selected}
        for row in per_item_rows:
            if not row.get("has_prediction"):
                continue
            rid = str(row.get("id"))
            record = id_to_record.get(rid)
            if not record:
                continue
            pred_text = ""
            if preds is None:
                pred_text = str(record.get("gold_answer") or "")
            else:
                pred_text = str(preds.get(rid) or "")

            extra = llm.score(record=record, prediction_text=pred_text, bucket=str(row.get("bucket") or ""))
            if extra:
                for k in ["ambiguity_resolution", "causal_link_accuracy", "confidence_calibration", "narrative_consistency", "notes"]:
                    if k in extra:
                        if k == "notes":
                            row["llm_notes"] = extra.get(k)
                        else:
                            row[k] = coerce_unit_score(extra.get(k))
                llm_scored += 1

        llm_path = out_dir / "m3_e2_fidelity.human_loop_llm.jsonl"
        with llm_path.open("w", encoding="utf-8") as f:
            for row in per_item_rows:
                out = {"id": row.get("id"), "bucket": row.get("bucket")}
                for k in ["ambiguity_resolution", "causal_link_accuracy", "confidence_calibration", "narrative_consistency", "llm_notes"]:
                    if k in row:
                        out[k] = row.get(k)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary = {
        "dataset": str(dataset_path),
        "predictions": str(Path(args.predictions)) if args.predictions else None,
        "n_per_type": args.n_per_type,
        "point_n": point_n,
        "point_per_domain": args.point_per_domain,
        "max_domains": args.max_domains,
        "seed": args.seed,
        "missing_predictions": missing_pred if preds is not None else 0,
        "human_loop": args.human_loop,
        "llm": {
            "model": args.llm_model,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
        }
        if args.human_loop == "llm"
        else None,
        "human_export_path": str(human_export_path) if human_export_path else None,
        "merged_human": merged_human,
        "llm_scored": llm_scored,
        "aggregate_by_bucket": aggregate_by_bucket(per_item_rows),
    }

    per_item_path = out_dir / "m3_e2_fidelity.per_item.jsonl"
    with per_item_path.open("w", encoding="utf-8") as f:
        for row in per_item_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = out_dir / "m3_e2_fidelity.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if preds is None:
        print("NOTE: No --predictions provided; evaluated gold_answer as a baseline.")
    if missing_pred:
        print(f"WARNING: Missing predictions for {missing_pred} selected items.")
    if human_export_path:
        print(f"Wrote: {human_export_path}")

    print(f"Wrote: {per_item_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()

