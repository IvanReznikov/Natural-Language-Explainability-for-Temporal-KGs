#!/usr/bin/env python3
"""M3-E3c: Cognitive Load Assessment.

This tooling exports a scenario set (with multiple explanation conditions) and
analyzes NASA TLX + optional auxiliary measures.

Conditions are free-form strings; the experimental plan suggests:
- dense_text
- structured_narrative
- timeline_plus_text
- interactive

The export step writes a scenario JSONL and a minimal static HTML survey.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from temporal_nlg.evaluation.m3_e3 import CognitiveLoadResponse, aggregate_cognitive_load


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


def _copy_web_assets(out_dir: Path) -> None:
    src = Path(__file__).parent / "web" / "cognitive_load.html"
    if not src.exists():
        return
    (out_dir / "web").mkdir(parents=True, exist_ok=True)
    (out_dir / "web" / "cognitive_load.html").write_text(
        src.read_text(encoding="utf-8"), encoding="utf-8"
    )


def export_scenarios(args: argparse.Namespace) -> None:
    """Export a scenario set.

    By default we derive one text per condition from a single explanation string.
    Users can later edit the scenario JSONL to provide truly distinct formats.
    """

    dataset_path = Path(args.dataset)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_jsonl(dataset_path))
    rng = random.Random(args.seed)
    if args.n_scenarios <= 0:
        raise SystemExit("--n-scenarios must be > 0")
    sample = rows if len(rows) <= args.n_scenarios else rng.sample(rows, args.n_scenarios)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if not conditions:
        raise SystemExit("--conditions must contain at least one condition")

    scenarios: List[dict] = []
    for i, r in enumerate(sample):
        sid = str(r.get("id") or f"scenario_{i}")
        base_text = str(r.get("gold_answer") or "")
        variants = {c: base_text for c in conditions}
        scenarios.append(
            {
                "scenario_id": sid,
                "domain": str(r.get("domain") or "unknown"),
                "query": str(r.get("query") or r.get("question") or ""),
                "conditions": variants,
            }
        )

    path = out_dir / "m3_e3c_scenarios.jsonl"
    _write_jsonl(path, scenarios)

    # Empty response template.
    template = out_dir / "m3_e3c_response_template.csv"
    with template.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "condition",
                "tlx_mental",
                "tlx_physical",
                "tlx_temporal",
                "tlx_performance",
                "tlx_effort",
                "tlx_frustration",
                "mental_effort_0_10",
                "retention_score_0_1",
                "attention_on_key_ratio_0_1",
            ],
        )
        w.writeheader()

    _copy_web_assets(out_dir)

    print(f"Wrote scenarios: {path}")
    print(f"Web UI: {out_dir / 'web' / 'cognitive_load.html'}")


def _read_responses(path: Path) -> List[CognitiveLoadResponse]:
    out: List[CognitiveLoadResponse] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    out.append(CognitiveLoadResponse(**row))
                except Exception:
                    continue
        return out

    for obj in _iter_jsonl(path):
        try:
            out.append(CognitiveLoadResponse(**obj))
        except Exception:
            continue
    return out


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    responses = _read_responses(Path(args.responses))
    summary = aggregate_cognitive_load(responses)

    # Success criteria check: TLX < 50 for all conditions where tlx is present.
    tlx = summary.get("tlx_mean_by_condition") or {}
    tlx_ok: Dict[str, Optional[bool]] = {}
    for cond, v in tlx.items():
        if v is None:
            tlx_ok[cond] = None
        else:
            tlx_ok[cond] = float(v) < 50.0

    summary["success_criteria"] = {"tlx_lt_50_by_condition": tlx_ok}

    (out_dir / "m3_e3c_cognitive_load.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote summary: {out_dir / 'm3_e3c_cognitive_load.summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_exp = sub.add_parser("export")
    ap_exp.add_argument("--dataset", type=str, required=True)
    ap_exp.add_argument("--output-dir", type=str, required=True)
    ap_exp.add_argument("--n-scenarios", type=int, default=40)
    ap_exp.add_argument(
        "--conditions",
        type=str,
        default="dense_text,structured_narrative,timeline_plus_text,interactive",
    )
    ap_exp.add_argument("--seed", type=int, default=13)
    ap_exp.set_defaults(func=export_scenarios)

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--responses", type=str, required=True, help="Response JSONL or CSV")
    ap_an.add_argument("--output-dir", type=str, required=True)
    ap_an.set_defaults(func=analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
