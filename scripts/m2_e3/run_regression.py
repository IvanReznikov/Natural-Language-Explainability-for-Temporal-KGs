#!/usr/bin/env python3
"""
scripts/m2_e3/run_regression.py
===============================
Run the M2-E3 parser regression test suite against the dedicated regression set.

This script covers the cases previously flagged by the reviewer:
  - AGG: standard, verb-trigger, and question-form variants
  - PREDICT: bare-year date vs quarter period
  - POINT: parametric date cases (date not in query text)
  - EXPLANATION/SEQUENCE vs CAUSAL ambiguity
  - Known reviewer failure cases (q612, q097, q178, q742, q1317)

Usage
-----
    # Rules-only (fast, no GPU required)
    python -X utf8 scripts/m2_e3/run_regression.py --rules-only

    # With Qwen model (requires adapter)
    python -X utf8 scripts/m2_e3/run_regression.py

Output
------
  Prints a per-example pass/fail table with expected vs. actual canonical query.
  Exits with code 0 if all tests pass, 1 if any fail.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windows console fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.m2_e3_construct.run_construct import (
    build_mappings_for_row,
    build_template_improved,
)
from experiments.m2_e3_parse.run_parse import (
    load_jsonl,
    load_qwen_parser_bundle,
    parse_row,
    parse_row_rules,
)


from scripts.m2_e3.eval_construct_from_preds import _lenient_cq_match

# ── CQ evaluation ──────────────────────────────────────────────────────────────


def _eval_cq(pred_cq: str, gold_cq: str) -> tuple[bool, bool]:
    """Return (exact_match, lenient_match)."""
    exact = pred_cq.strip() == gold_cq.strip()
    lenient = _lenient_cq_match(pred_cq, gold_cq)
    return exact, lenient


def _intent_ok(pred: Dict, gold: Dict) -> bool:
    pred_i = pred.get("intent_labels") or []
    gold_i = gold.get("intent_labels") or []
    # Primary intent must match
    return bool(pred_i) and bool(gold_i) and pred_i[0] == gold_i[0]


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--regression-data",
        type=Path,
        default=ROOT / "experiments" / "m2_e3_parse" / "data" / "regression_set.jsonl",
        help="Path to regression JSONL file",
    )
    ap.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "experiments" / "m2_e3_parse" / "data" / "temporal_queries_merged.jsonl",
        help="Gold JSONL for entity mapping normalisation",
    )
    ap.add_argument(
        "--adapter-dir",
        type=Path,
        default=ROOT / "experiments" / "m2_e3_parse" / "artifacts" / "qwen_parser_lora",
        help="Qwen LoRA adapter directory",
    )
    ap.add_argument(
        "--rules-only", action="store_true", help="Skip Qwen model; use rule-based parser only"
    )
    args = ap.parse_args()

    regression_path = args.regression_data.resolve()
    if not regression_path.exists():
        print(f"ERROR: Regression file not found: {regression_path}", file=sys.stderr)
        sys.exit(1)

    regression_rows: List[Dict] = load_jsonl(regression_path)
    print(f"Loaded {len(regression_rows)} regression cases from {regression_path}\n")

    # Load Qwen model
    qwen_bundle: Optional[Dict[str, Any]] = None
    if not args.rules_only:
        adapter_dir = args.adapter_dir.resolve()
        if (adapter_dir / "adapter_config.json").exists():
            print(f"[Setup] Loading Qwen LoRA adapter from {adapter_dir} ...")
            qwen_bundle = load_qwen_parser_bundle(adapter_dir)
            if qwen_bundle:
                print(f"  Adapter loaded  [QWEN MODEL]  device={qwen_bundle['device']}\n")
            else:
                print("  WARNING: adapter load failed — falling back to rules-only\n")
        else:
            print(f"[Setup] Qwen adapter not found — falling back to rules-only\n")

    parser_label = "[QWEN MODEL]" if qwen_bundle else "[RULES]"

    # Load entity mappings
    gold_path = args.gold.resolve()
    gold_mappings: Dict[str, str] = {}
    row_mappings: Dict[str, Dict[str, str]] = {}
    if gold_path.exists():
        for gr in load_jsonl(gold_path):
            rm = build_mappings_for_row(gr)
            row_mappings[gr["id"]] = rm
            gold_mappings.update(rm)

    # ── Run regression ────────────────────────────────────────────────────────
    results: List[Dict] = []
    col_w = 38

    header = (
        f"{'ID':<22} {'Category':<28} {'Intent':6} {'Exact':6} {'Lenient':7}  "
        f"{'Pred CQ (truncated)':<{col_w}} {'Gold CQ (truncated)':<{col_w}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for row in regression_rows:
        qid = row["id"]
        category = row.get("category", "")
        gold_cq = row.get("canonical_query", "")

        # Parse
        if qwen_bundle:
            bundles: Dict[str, Any] = {"intent": None, "parser": qwen_bundle}
            pred, _ = parse_row(row, bundles, threshold=0.25, fallback_on_error=True)
        else:
            pred = parse_row_rules(row)

        pred_intents = pred.get("intent_labels") or []
        pred_frame = pred.get("frame") or {}

        # Build canonical query
        pred_cq = build_template_improved(
            frame=pred_frame,
            intents=pred_intents,
            qid=qid,
            row_mappings=row_mappings,
            global_mappings=gold_mappings,
        )

        intent_match = _intent_ok(pred, row)
        exact, lenient = _eval_cq(pred_cq, gold_cq)

        def _trunc(s: str, w: int = col_w) -> str:
            return (s[: w - 1] + "…") if len(s) > w else s

        intent_sym = "✓" if intent_match else "✗"
        exact_sym = "✓" if exact else "✗"
        lenient_sym = "✓" if lenient else "✗"

        print(
            f"{qid:<22} {category:<28} {intent_sym:<6} {exact_sym:<6} {lenient_sym:<7}  "
            f"{_trunc(pred_cq):<{col_w}} {_trunc(gold_cq):<{col_w}}"
        )

        results.append(
            {
                "id": qid,
                "category": category,
                "intent_ok": intent_match,
                "exact": exact,
                "lenient": lenient,
                "pred_cq": pred_cq,
                "gold_cq": gold_cq,
                "note": row.get("note", ""),
            }
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    n_intent = sum(1 for r in results if r["intent_ok"])
    n_exact = sum(1 for r in results if r["exact"])
    n_lenient = sum(1 for r in results if r["lenient"])

    print(sep)
    print(f"\nParser mode : {parser_label}")
    print(f"Total cases : {n}")
    print(f"Intent OK   : {n_intent}/{n} = {100*n_intent/n:.1f}%")
    print(f"Exact match : {n_exact}/{n} = {100*n_exact/n:.1f}%")
    print(f"Lenient     : {n_lenient}/{n} = {100*n_lenient/n:.1f}%\n")

    # Per-category summary
    categories = {}
    for r in results:
        cat = r["category"].split("_")[0]  # group by prefix AGG, PREDICT, POINT, etc.
        categories.setdefault(cat, []).append(r)

    print("Per-category summary:")
    print(f"  {'Category':<12}  {'Intent':7}  {'Exact':7}  {'Lenient':8}  Count")
    print("  " + "-" * 50)
    for cat, rows in sorted(categories.items()):
        ni = sum(1 for r in rows if r["intent_ok"])
        ne = sum(1 for r in rows if r["exact"])
        nl = sum(1 for r in rows if r["lenient"])
        print(
            f"  {cat:<12}  {ni}/{len(rows):>2}       {ne}/{len(rows):>2}       {nl}/{len(rows):>2}        {len(rows)}"
        )

    # Failures
    failures = [r for r in results if not r["lenient"]]
    if failures:
        print(f"\n{len(failures)} lenient-match failures:")
        for r in failures:
            print(f"  [{r['id']}]  {r['note']}")
            print(f"    Pred: {r['pred_cq']}")
            print(f"    Gold: {r['gold_cq']}")

    # Exit code
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
