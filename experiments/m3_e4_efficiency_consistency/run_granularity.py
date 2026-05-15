#!/usr/bin/env python3
"""M3-E4d: Robustness to Temporal Granularity Changes."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from temporal_nlg.evaluation.m3_e2_fidelity import M3E2FidelityEvaluator
from temporal_nlg.evaluation.m3_e4 import (
    GranularityScenario,
    GranularityVariant,
    aggregate_granularity,
    bucket_from_time_scope,
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


def _load_granularity_predictions(path: Optional[Path]) -> Dict[Tuple[str, str], str]:
    if path is None:
        return {}
    preds: Dict[Tuple[str, str], str] = {}
    for obj in _iter_jsonl(path):
        rid = obj.get("id") or obj.get("scenario_id")
        gran = obj.get("granularity")
        if not rid or not gran:
            continue
        text = obj.get("prediction") or obj.get("generated_text") or obj.get("output") or obj.get("text")
        if text is None:
            continue
        preds[(str(rid), str(gran))] = str(text)
    return preds


def _load_scenarios(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for obj in _iter_jsonl(path):
        sid = obj.get("scenario_id") or obj.get("id")
        if sid:
            out[str(sid)] = obj
    return out


def _quality_proxy_from_metrics(metrics: Dict[str, float]) -> Optional[float]:
    parts = []
    for k in ("context_relevance", "entity_coverage", "unnecessary_detail_score"):
        v = metrics.get(k)
        if v is not None:
            parts.append(float(v))
    return sum(parts) / len(parts) if parts else None


def export_variants(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_jsonl(dataset))
    rng = random.Random(args.seed)

    if args.n_scenarios <= 0:
        raise SystemExit("--n-scenarios must be > 0")

    sample = rows if len(rows) <= args.n_scenarios else rng.sample(rows, args.n_scenarios)

    scenarios: List[dict] = []
    variants: List[dict] = []
    granularities = ["seconds", "minutes", "hours", "days", "months", "years", "decades"]
    preds = _load_granularity_predictions(Path(args.predictions)) if args.predictions else {}

    for idx, r in enumerate(sample):
        rid = str(r.get("id") or f"row_{idx}")
        scenarios.append(
            GranularityScenario(
                scenario_id=rid,
                domain=str(r.get("domain") or "unknown"),
                bucket=bucket_from_time_scope(r.get("time_scope")),
                query=str(r.get("query") or r.get("question") or ""),
                gold_facts=r.get("gold_facts"),
            ).model_dump()
        )

        base_text = str(r.get("gold_answer") or "")
        for g in granularities:
            text = preds.get((rid, g)) or base_text
            variants.append(
                GranularityVariant(
                    scenario_id=rid,
                    granularity=g,  # placeholder text for now
                    text=text,
                    quality_score=None,
                    length_chars=len(text),
                ).model_dump()
            )

    scenarios_path = out_dir / "m3_e4d_scenarios.jsonl"
    variants_path = out_dir / "m3_e4d_scaled.jsonl"
    _write_jsonl(scenarios_path, scenarios)
    _write_jsonl(variants_path, variants)

    template_path = out_dir / "m3_e4d_scores_template.csv"
    with template_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["scenario_id", "granularity", "quality_score", "length_chars"],
        )
        w.writeheader()
        for v in variants:
            w.writerow(
                {
                    "scenario_id": v["scenario_id"],
                    "granularity": v["granularity"],
                    "quality_score": v.get("quality_score") or "",
                    "length_chars": v.get("length_chars") or "",
                }
            )

    print(f"Wrote scenarios: {scenarios_path}")
    print(f"Wrote variants: {variants_path}")
    print(f"Wrote scores template: {template_path}")


def _read_variants(path: Path) -> List[GranularityVariant]:
    out: List[GranularityVariant] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    out.append(GranularityVariant(**row))
                except Exception:
                    continue
        return out

    for obj in _iter_jsonl(path):
        try:
            out.append(GranularityVariant(**obj))
        except Exception:
            continue
    return out


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = _read_variants(Path(args.variants))
    scenarios = _load_scenarios(Path(args.scenarios)) if args.scenarios else {}
    evaluator = M3E2FidelityEvaluator() if scenarios else None

    coerced: List[GranularityVariant] = []
    for v in variants:
        vv = v.model_copy(deep=True)
        if vv.quality_score is None and evaluator is not None:
            record = scenarios.get(str(vv.scenario_id))
            if record:
                metrics = evaluator.evaluate_example(record, prediction_text=str(vv.text))
                vv.quality_score = _quality_proxy_from_metrics(metrics)
        coerced.append(vv)

    summary = aggregate_granularity(coerced)

    (out_dir / "m3_e4d_granularity.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote summary: {out_dir / 'm3_e4d_granularity.summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_exp = sub.add_parser("export")
    ap_exp.add_argument("--dataset", type=str, required=True)
    ap_exp.add_argument("--predictions", type=str, default=None, help="JSONL with {id, granularity, prediction}")
    ap_exp.add_argument("--output-dir", type=str, required=True)
    ap_exp.add_argument("--n-scenarios", type=int, default=200)
    ap_exp.add_argument("--seed", type=int, default=13)
    ap_exp.set_defaults(func=export_variants)

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--variants", type=str, required=True, help="CSV or JSONL variants file")
    ap_an.add_argument("--scenarios", type=str, default=None, help="m3_e4d_scenarios.jsonl (for quality proxy)")
    ap_an.add_argument("--output-dir", type=str, required=True)
    ap_an.set_defaults(func=analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
