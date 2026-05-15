#!/usr/bin/env python3
"""M3-E4a: Generation Efficiency Benchmarking.

Exports scenario sets and ingests run logs to compute latency/throughput metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from temporal_nlg.evaluation.m3_e2_fidelity import M3E2FidelityEvaluator
from temporal_nlg.evaluation.m3_e4 import EfficiencyScenario, EfficiencyRun, aggregate_efficiency, bucket_from_time_scope


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


def _load_predictions(path: Optional[Path]) -> Dict[Tuple[str, Optional[str]], str]:
    if path is None:
        return {}
    preds: Dict[Tuple[str, Optional[str]], str] = {}
    for obj in _iter_jsonl(path):
        rid = obj.get("id") or obj.get("scenario_id")
        if not rid:
            continue
        method = obj.get("method")
        text = obj.get("prediction") or obj.get("generated_text") or obj.get("output") or obj.get("text")
        if text is None:
            continue
        preds[(str(rid), str(method) if method is not None else None)] = str(text)
    return preds


def _prediction_for(preds: Dict[Tuple[str, Optional[str]], str], scenario_id: str, method: Optional[str]) -> Optional[str]:
    if not preds:
        return None
    key = (str(scenario_id), str(method) if method is not None else None)
    if key in preds:
        return preds[key]
    return preds.get((str(scenario_id), None))


def _load_scenarios(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for obj in _iter_jsonl(path):
        sid = obj.get("scenario_id") or obj.get("id")
        if sid:
            out[str(sid)] = obj
    return out


def _quality_proxy_from_metrics(metrics: Dict[str, Any]) -> Optional[float]:
    parts = []
    for k in ("context_relevance", "entity_coverage", "unnecessary_detail_score"):
        v = metrics.get(k)
        if v is not None:
            parts.append(float(v))
    return sum(parts) / len(parts) if parts else None


def _complexity_level(record: dict) -> int:
    gold_facts = record.get("gold_facts")
    n = len(gold_facts) if isinstance(gold_facts, list) else 0
    level = 1 + min(4, n // 3)
    return max(1, min(5, int(level)))


def export_scenarios(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = _load_predictions(Path(args.predictions)) if args.predictions else {}
    rows = list(_iter_jsonl(dataset))
    rng = random.Random(args.seed)

    if args.n_scenarios <= 0:
        raise SystemExit("--n-scenarios must be > 0")

    sample = rows if len(rows) <= args.n_scenarios else rng.sample(rows, args.n_scenarios)

    scenarios: List[dict] = []
    for idx, r in enumerate(sample):
        rid = str(r.get("id") or f"row_{idx}")
        text = preds.get(rid) or str(r.get("gold_answer") or "")
        scenario = EfficiencyScenario(
            scenario_id=rid,
            domain=str(r.get("domain") or "unknown"),
            bucket=bucket_from_time_scope(r.get("time_scope")),
            complexity_level=_complexity_level(r),
            time_scope=str(r.get("time_scope") or ""),
            query=str(r.get("query") or r.get("question") or ""),
            explanation_text=text,
            gold_facts=r.get("gold_facts"),
        )
        scenarios.append(scenario.model_dump())

    scenarios_path = out_dir / "m3_e4a_scenarios.jsonl"
    _write_jsonl(scenarios_path, scenarios)

    template_path = out_dir / "m3_e4a_runs_template.csv"
    with template_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_id",
                "method",
                "complexity_level",
                "latency_ms",
                "tokens_out",
                "cost",
                "quality_proxy",
            ],
        )
        w.writeheader()
        for s in scenarios:
            w.writerow(
                {
                    "scenario_id": s["scenario_id"],
                    "method": "",
                    "complexity_level": s["complexity_level"],
                    "latency_ms": "",
                    "tokens_out": "",
                    "cost": "",
                    "quality_proxy": "",
                }
            )

    print(f"Wrote scenarios: {scenarios_path}")
    print(f"Wrote runs template: {template_path}")


def _read_runs(path: Path) -> List[EfficiencyRun]:
    out: List[EfficiencyRun] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    out.append(EfficiencyRun(**row))
                except Exception:
                    continue
        return out

    for obj in _iter_jsonl(path):
        try:
            out.append(EfficiencyRun(**obj))
        except Exception:
            continue
    return out


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _read_runs(Path(args.runs))

    scenarios = _load_scenarios(Path(args.scenarios)) if args.scenarios else {}
    preds = _load_predictions(Path(args.predictions)) if args.predictions else {}
    evaluator = M3E2FidelityEvaluator() if scenarios else None

    if not runs and scenarios:
        runs = [
            EfficiencyRun(
                scenario_id=sid,
                method=str(args.default_method),
                complexity_level=int(rec.get("complexity_level") or 3),
            )
            for sid, rec in scenarios.items()
        ]

    coerced: List[EfficiencyRun] = []
    for r in runs:
        rr = r.model_copy(deep=True)
        if rr.quality_proxy is None and evaluator is not None:
            record = scenarios.get(str(rr.scenario_id))
            if record:
                pred_text = _prediction_for(preds, rr.scenario_id, rr.method) or record.get("explanation_text") or ""
                metrics = evaluator.evaluate_example(record, prediction_text=str(pred_text))
                rr.quality_proxy = _quality_proxy_from_metrics(metrics)
        coerced.append(rr)

    summary = aggregate_efficiency(coerced)

    (out_dir / "m3_e4a_efficiency.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote summary: {out_dir / 'm3_e4a_efficiency.summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_exp = sub.add_parser("export")
    ap_exp.add_argument("--dataset", type=str, required=True)
    ap_exp.add_argument("--predictions", type=str, default=None)
    ap_exp.add_argument("--output-dir", type=str, required=True)
    ap_exp.add_argument("--n-scenarios", type=int, default=1000)
    ap_exp.add_argument("--seed", type=int, default=13)
    ap_exp.set_defaults(func=export_scenarios)

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--runs", type=str, required=True, help="CSV or JSONL runs file")
    ap_an.add_argument("--scenarios", type=str, default=None, help="m3_e4a_scenarios.jsonl (for quality proxy)")
    ap_an.add_argument("--predictions", type=str, default=None, help="Optional JSONL with {id, method?, prediction}")
    ap_an.add_argument("--default-method", type=str, default="baseline", help="Method name when runs file is empty")
    ap_an.add_argument("--output-dir", type=str, required=True)
    ap_an.set_defaults(func=analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
