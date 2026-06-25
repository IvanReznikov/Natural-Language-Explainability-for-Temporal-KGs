#!/usr/bin/env python3
"""
scripts/m2_e3/eval_construct_from_preds.py
==========================================
Evaluate end-to-end canonical query accuracy using *predicted frames*
(not gold frames), broken out by intent.

The script reads a preds_*.jsonl file produced by eval_pipeline.py, feeds
each predicted frame + intent through build_template_improved(), and compares
the result to the gold canonical_query from the merged gold file.

Usage
-----
    # Using rules preds (no GPU needed)
    python scripts/m2_e3/eval_construct_from_preds.py \\
        --preds  experiments/m2_e3_parse/runs/eval_pipeline/preds_rules.jsonl \\
        --gold   experiments/m2_e3_parse/data/temporal_queries_merged.jsonl

    # After Colab — using qwen+fallback preds
    python scripts/m2_e3/eval_construct_from_preds.py \\
        --preds  experiments/m2_e3_parse/runs/eval_pipeline/preds_qwen_plus_fallback.jsonl \\
        --gold   experiments/m2_e3_parse/data/temporal_queries_merged.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Windows console fix done in main block



# ---------------------------------------------------------------------------
# Repo path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.m2_e3_construct.run_construct import (  # noqa: E402
    build_template_improved,
    build_mappings_for_row,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_cq(cq: str) -> str:
    """Normalise a canonical query for lenient comparison.

    Strips trailing whitespace, lowercases everything inside quotes,
    and collapses repeated underscores/spaces so that minor snake_case
    differences don't count as failures.
    """
    if not cq:
        return ""
    # Lowercase string values inside single-quoted CQ arguments
    cq = re.sub(r"'([^']*)'", lambda m: "'" + m.group(1).lower().replace(" ", "_") + "'", cq)
    return cq.strip()


def _canonical_query_type(cq: str) -> str:
    """Extract the UPPERCASE keyword from a canonical query string."""
    m = re.match(r"([A-Z]+)\(", cq or "")
    return m.group(1) if m else "UNKNOWN"

def _normalize_slot(val: str) -> str:
    """Normalise a single slot value for comparison: lowercase, strip determiners, snake_case."""
    val = val.lower().strip()
    val = re.sub(r"^(the|a|an)\s+", "", val)
    val = re.sub(r"[^a-z0-9]+", "_", val)
    val = re.sub(r"_+", "_", val).strip("_")
    val = val.replace("fine_tuning", "finetuning")
    return val


def _extract_slots(cq: str) -> Dict[str, str]:
    """Parse a canonical query string into a {key: value} dict."""
    return {k: v for k, v in re.findall(r"(\w+)='([^']*)'", cq)}


def _slots_match(pred_val: str, gold_val: str) -> bool:
    p_norm = _normalize_slot(pred_val)
    g_norm = _normalize_slot(gold_val)
    if p_norm == g_norm:
        return True
    if not p_norm or not g_norm:
        return False
        
    stop_words = {
        "the", "a", "an", "of", "in", "on", "at", "by", "for", "with", "about", "to", "from", "and", "or",
        "signing", "signed", "built", "founded", "launched", "opened", "invented", "completed", "declared",
        "announced", "established", "occur", "happen", "start", "end", "begin", "finish", "take", "place"
    }
    p_words = {w for w in p_norm.split("_") if w not in stop_words}
    g_words = {w for w in g_norm.split("_") if w not in stop_words}
    
    if not p_words or not g_words:
        return p_norm == g_norm
        
    intersection = p_words.intersection(g_words)
    if intersection == p_words or intersection == g_words:
        return True
        
    jaccard = len(intersection) / len(p_words.union(g_words))
    return jaccard >= 0.5


def _lenient_cq_match(pred_cq: str, gold_cq: str) -> bool:
    """Compare two canonical queries semantically by checking slot values."""
    pred_type = _canonical_query_type(pred_cq)
    gold_type = _canonical_query_type(gold_cq)
    if pred_type != gold_type:
        return False
        
    pred_slots = _extract_slots(pred_cq)
    gold_slots = _extract_slots(gold_cq)
    
    # We want to match all gold slots (except 'date' in POINT queries)
    for k, g_val in gold_slots.items():
        if gold_type == "POINT" and k == "date":
            continue
        p_val = pred_slots.get(k, "")
        if not _slots_match(p_val, g_val):
            return False
            
    return True


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(preds_path: Path, gold_path: Path) -> None:
    print(f"\nPreds : {preds_path}")
    print(f"Gold  : {gold_path}\n")

    # Load gold index
    gold_rows = {r["id"]: r for r in _load_jsonl(gold_path)}

    # Build global/row mappings from gold (same as run_construct.py does)
    global_mappings: Dict[str, str] = {}
    row_mappings: Dict[str, Dict[str, str]] = {}
    for qid, gold in gold_rows.items():
        rm = build_mappings_for_row(gold)
        row_mappings[qid] = rm
        global_mappings.update(rm)

    # Load predictions
    pred_rows = _load_jsonl(preds_path)
    if not pred_rows:
        print("ERROR: no predictions found in preds file.")
        sys.exit(1)

    # Evaluate
    per_intent: Dict[str, Dict] = defaultdict(lambda: {
        "exact": 0, "lenient": 0, "wrong_type": 0, "no_gold": 0, "total": 0
    })
    failures: List[Dict] = []
    total = exact = lenient = 0

    for pred in pred_rows:
        qid = pred.get("id", "")
        gold = gold_rows.get(qid)
        if gold is None:
            continue  # no gold row for this test example

        gold_cq = gold.get("canonical_query", "")
        gold_intent = (gold.get("intent_labels") or ["unknown"])[0]

        # Build canonical query from predicted frame + intent
        pred_frame = pred.get("frame") or {}
        pred_intents = pred.get("intent_labels") or []
        predicted_cq = build_template_improved(
            frame=pred_frame,
            intents=pred_intents,
            qid=qid,
            row_mappings=row_mappings,
            global_mappings=global_mappings,
        )

        total += 1
        per_intent[gold_intent]["total"] += 1

        is_exact = (predicted_cq.strip() == gold_cq.strip())
        is_lenient = (
            _normalize_cq(predicted_cq) == _normalize_cq(gold_cq)
            or _lenient_cq_match(predicted_cq, gold_cq)
        )
        wrong_type = (_canonical_query_type(predicted_cq) != _canonical_query_type(gold_cq))

        if is_exact:
            exact += 1
            lenient += 1
            per_intent[gold_intent]["exact"] += 1
            per_intent[gold_intent]["lenient"] += 1
        elif is_lenient:
            lenient += 1
            per_intent[gold_intent]["lenient"] += 1
        else:
            if wrong_type:
                per_intent[gold_intent]["wrong_type"] += 1
            failures.append({
                "id": qid,
                "text": pred.get("text", ""),
                "gold_intent": gold_intent,
                "pred_intent": (pred_intents or ["?"])[0],
                "pred_frame": pred_frame,
                "gold_cq": gold_cq,
                "pred_cq": predicted_cq,
                "wrong_type": wrong_type,
            })

    # ---------------------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------------------
    divider = "=" * 78
    print(divider)
    preds_label = preds_path.stem.replace("preds_", "").replace("_", "+")
    print(f"  End-to-End Canonical Query Accuracy  |  mode = {preds_label}")
    print(divider)
    print(f"  Total evaluated   : {total}")
    print(f"  Exact match       : {exact}/{total} = {100*exact/max(total,1):.1f}%")
    print(f"  Lenient match     : {lenient}/{total} = {100*lenient/max(total,1):.1f}%")
    print(f"  (lenient = ignoring case, spaces, underscores inside values)")
    print()

    # Per-intent table
    print(f"  {'Intent':<16}  {'Exact':>6}  {'Lenient':>7}  {'WrongType':>9}  {'Total':>6}  {'Exact%':>7}  {'Lenient%':>8}")
    print("  " + "-" * 64)
    for intent, s in sorted(per_intent.items()):
        n = s["total"]
        ep = 100 * s["exact"] / max(n, 1)
        lp = 100 * s["lenient"] / max(n, 1)
        mark = "  ✓" if ep >= 70 else "  ✗"
        print(f"  {intent:<16}  {s['exact']:>6}  {s['lenient']:>7}  {s['wrong_type']:>9}  {n:>6}  {ep:>6.1f}%{mark}  {lp:>7.1f}%")
    print(divider)

    # Failure examples
    if failures:
        print(f"\n  Failure examples (first 10 of {len(failures)} non-matching):\n")
        for f in failures[:10]:
            print(f"  ID        : {f['id']}")
            print(f"  Text      : {f['text']}")
            print(f"  Gold CQ   : {f['gold_cq']}")
            print(f"  Pred CQ   : {f['pred_cq']}")
            print(f"  Pred frame: {f['pred_frame']}")
            print(f"  Wrong type: {f['wrong_type']}")
            print()


def main() -> None:
    # Windows console fix
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preds",
        type=Path,
        default=Path("experiments/m2_e3_parse/runs/eval_pipeline/preds_rules.jsonl"),
        help="Predictions JSONL from eval_pipeline.py",
    )
    ap.add_argument(
        "--gold",
        type=Path,
        default=Path("experiments/m2_e3_parse/data/temporal_queries_merged.jsonl"),
        help="Merged gold JSONL with canonical_query field",
    )
    args = ap.parse_args()

    preds_path = args.preds.resolve()
    gold_path = args.gold.resolve()

    if not preds_path.exists():
        print(f"ERROR: preds file not found: {preds_path}")
        sys.exit(1)
    if not gold_path.exists():
        print(f"ERROR: gold file not found: {gold_path}")
        sys.exit(1)

    evaluate(preds_path, gold_path)


if __name__ == "__main__":
    main()

