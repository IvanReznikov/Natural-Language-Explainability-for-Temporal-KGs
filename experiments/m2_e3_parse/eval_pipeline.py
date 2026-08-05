"""
experiments/m2_e3_parse/eval_pipeline.py
==========================================
M2 milestone evaluation script.

Runs the three required pipeline modes against the test split and produces
the comparative table that is the key deliverable for M2 sign-off:

    Mode            | Intent-F1 | Frame-F1 | Span-F1 | Fallback%
    ----------------|-----------|----------|---------|----------
    rules-only      |   ...     |   ...    |   ...   |    0%
    qwen-only       |   ...     |   ...    |   ...   |    0%
    qwen+fallback   |   ...     |   ...    |   ...   |   ...%

Usage
-----
    # After Colab training — supply the adapter dir
    python experiments/m2_e3_parse/eval_pipeline.py \\
        --test-data  experiments/m2_e3_parse/data/splits_qwen/test_prompts.jsonl \\
        --gold-data  experiments/m2_e3_parse/data/temporal_queries_merged.jsonl \\
        --adapter-dir experiments/m2_e3_parse/artifacts/qwen_parser_lora \\
        --output-dir  experiments/m2_e3_parse/runs/eval_pipeline

    # Rules-only (no GPU needed — useful for baseline comparison)
    python experiments/m2_e3_parse/eval_pipeline.py \\
        --test-data  experiments/m2_e3_parse/data/splits_qwen/test_prompts.jsonl \\
        --gold-data  experiments/m2_e3_parse/data/temporal_queries_merged.jsonl \\
        --rules-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_pipeline")

# ---------------------------------------------------------------------------
# Add repo root to sys.path so we can import from experiments/ and scripts/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.m2_e3_parse.run_parse import (  # noqa: E402
    load_jsonl,
    load_qwen_parser_bundle,
    parse_row,
    parse_row_rules,
    save_jsonl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_gold_map(gold_path: Path) -> Dict[str, Dict]:
    """Return id → row dict from the merged gold file."""
    rows = load_jsonl(gold_path)
    return {r["id"]: r for r in rows}


def read_test_rows(test_path: Path) -> List[Dict]:
    """
    Test prompts written by prepare_qwen_data.py have a different schema
    (messages / target / id).  Reconstruct minimal gold-compatible rows
    from the prompt file so we can run the full pipeline.
    """
    rows = []
    with test_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Prompt dicts have 'messages' and 'target'
            if "messages" in obj:
                text = next((m["content"] for m in obj["messages"] if m["role"] == "user"), "")
                rows.append({"id": obj.get("id", ""), "text": text, "_prompt_row": obj})
            else:
                rows.append(obj)
    return rows


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _span_key(span: Dict) -> Tuple[str, int, int]:
    return str(span.get("label", "")), int(span.get("start", 0)), int(span.get("end", 0))


def _f1(gold: List, pred: List) -> float:
    gs, ps = set(gold), set(pred)
    tp = len(gs & ps)
    if tp == 0:
        return 0.0
    p = tp / len(ps) if ps else 0.0
    r = tp / len(gs) if gs else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _intent_f1_micro(gold_rows: Dict[str, Dict], preds: List[Dict]) -> float:
    """Micro-F1 over intent labels (raw label space)."""
    tp = fp = fn = 0
    for pred in preds:
        qid = pred["id"]
        gold = gold_rows.get(qid, {})
        g_set = set(gold.get("intent_labels", []))
        p_set = set(pred.get("intent_labels", []))
        tp += len(g_set & p_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _intent_accuracy(gold_rows: Dict[str, Dict], preds: List[Dict]) -> float:
    """Exact primary-intent accuracy (first label only)."""
    correct = total = 0
    for pred in preds:
        qid = pred["id"]
        gold = gold_rows.get(qid, {})
        g_intent = (gold.get("intent_labels") or [""])[0]
        p_intent = (pred.get("intent_labels") or [""])[0]
        total += 1
        if g_intent == p_intent:
            correct += 1
    return correct / total if total else 0.0


def _per_intent_accuracy(gold_rows: Dict[str, Dict], preds: List[Dict]) -> Dict[str, Dict]:
    stats: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred in preds:
        qid = pred["id"]
        gold = gold_rows.get(qid, {})
        g_intent = (gold.get("intent_labels") or ["unknown"])[0]
        p_intent = (pred.get("intent_labels") or [""])[0]
        stats[g_intent]["total"] += 1
        if g_intent == p_intent:
            stats[g_intent]["correct"] += 1
    result = {}
    for intent, s in sorted(stats.items()):
        acc = s["correct"] / max(s["total"], 1)
        result[intent] = {"correct": s["correct"], "total": s["total"], "accuracy": round(acc, 4)}
    return result


def _normalize_val(val: Any) -> str:
    import re

    s = str(val).lower().strip()
    s = re.sub(r"\b(the|a|an)\b", "", s)
    s = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()?]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _frame_f1(gold_rows: Dict[str, Dict], preds: List[Dict]) -> float:
    """Macro-average frame key-value F1."""
    scores = []
    for pred in preds:
        qid = pred["id"]
        gold = gold_rows.get(qid, {})
        g_frame = gold.get("frame") or {}
        p_frame = pred.get("frame") or {}
        # Treat each (key, value) pair as a token, normalising string values
        g_pairs = [(k, _normalize_val(v)) for k, v in g_frame.items()]
        p_pairs = [(k, _normalize_val(v)) for k, v in p_frame.items()]
        scores.append(_f1(g_pairs, p_pairs))
    return sum(scores) / len(scores) if scores else 0.0


def _span_f1(gold_rows: Dict[str, Dict], preds: List[Dict]) -> float:
    scores = []
    for pred in preds:
        qid = pred["id"]
        gold = gold_rows.get(qid, {})
        g_spans = [_span_key(s) for s in gold.get("spans", [])]
        p_spans = [_span_key(s) for s in pred.get("spans", [])]
        scores.append(_f1(g_spans, p_spans))
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_mode(
    mode: str,
    rows: List[Dict],
    gold_rows: Dict[str, Dict],
    qwen_bundle: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict]]:
    """Run one pipeline mode and return metrics dict and predictions list."""
    bundles: Dict[str, Any] = {"intent": None, "parser": None}
    use_fallback = mode.endswith("+fallback")

    if mode.startswith("qwen") and qwen_bundle is not None:
        bundles["parser"] = qwen_bundle

    preds: List[Dict] = []
    fallback_count = 0
    t0 = time.perf_counter()

    for row in rows:
        if mode == "rules":
            pred = parse_row_rules(row)
            preds.append(pred)
        else:
            pred, just = parse_row(
                row,
                bundles,
                threshold=0.25,
                fallback_on_error=use_fallback,
            )
            if "fallback_rules_on_error" in just.get("notes", []):
                fallback_count += 1
            preds.append(pred)

    elapsed = time.perf_counter() - t0

    metrics = {
        "mode": mode,
        "examples": len(preds),
        "elapsed_s": round(elapsed, 2),
        "intent_accuracy": round(_intent_accuracy(gold_rows, preds), 4),
        "intent_micro_f1": round(_intent_f1_micro(gold_rows, preds), 4),
        "frame_f1": round(_frame_f1(gold_rows, preds), 4),
        "span_f1": round(_span_f1(gold_rows, preds), 4),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(preds), 4) if preds else 0.0,
        "per_intent": _per_intent_accuracy(gold_rows, preds),
    }
    return metrics, preds


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

MODES_ORDER = ["rules", "qwen", "qwen+fallback"]


def print_summary_table(all_metrics: List[Dict]) -> None:
    cols = ["mode", "intent_accuracy", "intent_micro_f1", "frame_f1", "span_f1", "fallback_rate"]
    widths = [16, 14, 16, 10, 10, 12]
    headers = ["Mode", "Intent-Acc", "Intent-µF1", "Frame-F1", "Span-F1", "Fallback%"]

    header_line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)

    log.info("\n" + "=" * len(header_line))
    log.info("M2 Pipeline Evaluation Summary")
    log.info("=" * len(header_line))
    log.info("%s", header_line)
    log.info("%s", sep)

    for m in all_metrics:
        row_vals = [
            m.get("mode", ""),
            f"{m.get('intent_accuracy', 0):.3f}",
            f"{m.get('intent_micro_f1', 0):.3f}",
            f"{m.get('frame_f1', 0):.3f}",
            f"{m.get('span_f1', 0):.3f}",
            f"{m.get('fallback_rate', 0):.1%}",
        ]
        log.info("%s", "  ".join(f"{v:<{w}}" for v, w in zip(row_vals, widths)))

    log.info("=" * len(header_line))


def print_per_intent_table(metrics: Dict, label: str) -> None:
    per_intent = metrics.get("per_intent", {})
    if not per_intent:
        return
    log.info("\nPer-intent accuracy — %s", label)
    log.info("  %-14s  %8s  %8s  %8s", "Intent", "Correct", "Total", "Acc%")
    log.info("  %s", "-" * 44)
    for intent, s in sorted(per_intent.items()):
        mark = "  ✓" if s["accuracy"] >= 0.70 else "  ✗"
        log.info(
            "  %-14s  %8d  %8d  %7.1f%%%s",
            intent,
            s["correct"],
            s["total"],
            s["accuracy"] * 100,
            mark,
        )


def print_frame_field_breakdown(
    preds: List[Dict], gold_rows: Dict[str, Dict], test_rows: List[Dict], label: str
) -> None:
    import re
    from collections import defaultdict

    def extract_year(val: Any) -> Optional[str]:
        if not val:
            return None
        m = re.search(r"\b\d{4}\b", str(val))
        return m.group(0) if m else None

    def is_extractive(gold_val: Any, question: str) -> bool:
        if not gold_val or not question:
            return False
        g_norm = _normalize_val(gold_val)
        q_norm = _normalize_val(question)
        if not g_norm:
            return False
        year = extract_year(gold_val)
        if year and year in q_norm:
            return True
        return g_norm in q_norm

    test_texts = {}
    for r in test_rows:
        qid = r.get("id")
        text = r.get("text", "")
        if not text and "_prompt_row" in r:
            text = next(
                (m["content"] for m in r["_prompt_row"].get("messages", []) if m["role"] == "user"),
                "",
            )
        if qid:
            test_texts[qid] = text

    field_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    time_stats = {
        "extractive_total": 0,
        "extractive_correct": 0,
        "parametric_total": 0,
        "parametric_correct": 0,
        "year_match_total": 0,
        "year_match_correct": 0,
    }

    for pred in preds:
        qid = pred.get("id")
        if not qid or qid not in gold_rows:
            continue
        gold = gold_rows[qid]
        gold_frame = gold.get("frame") or {}
        pred_frame = pred.get("frame") or {}
        question = test_texts.get(qid, "")

        for field, gold_val in gold_frame.items():
            field_stats[field]["total"] += 1
            pred_val = pred_frame.get(field)

            is_correct = _normalize_val(pred_val) == _normalize_val(gold_val)
            if is_correct:
                field_stats[field]["correct"] += 1

            if field in ["time", "date", "period"]:
                g_year = extract_year(gold_val)
                p_year = extract_year(pred_val)
                if g_year:
                    time_stats["year_match_total"] += 1
                    if g_year == p_year:
                        time_stats["year_match_correct"] += 1

                if is_extractive(gold_val, question):
                    time_stats["extractive_total"] += 1
                    if is_correct:
                        time_stats["extractive_correct"] += 1
                else:
                    time_stats["parametric_total"] += 1
                    if is_correct:
                        time_stats["parametric_correct"] += 1

    log.info("\nFrame-field accuracy breakdown — %s", label)
    log.info("  %-20s  %8s  %8s  %8s  %8s", "Field", "Correct", "Total", "Acc%", "Miss%")
    log.info("  %s", "-" * 60)
    for field, s in sorted(field_stats.items(), key=lambda x: -x[1]["total"]):
        acc = 100 * s["correct"] / max(s["total"], 1)
        miss = 100 - acc
        mark = "  <-- paraphrase hotspot" if acc < 90 else ""
        log.info(
            "  %-20s  %8d  %8d  %7.1f%%  %7.1f%%%s",
            field,
            s["correct"],
            s["total"],
            acc,
            miss,
            mark,
        )

    if (
        time_stats["extractive_total"]
        or time_stats["parametric_total"]
        or time_stats["year_match_total"]
    ):
        log.info("\nTemporal / Date Extraction Analysis — %s", label)
        log.info("  %s", "=" * 60)
        if time_stats["extractive_total"]:
            ext_acc = 100 * time_stats["extractive_correct"] / time_stats["extractive_total"]
            log.info(
                "  Extractive time slots  : %d/%d = %.1f%%",
                time_stats["extractive_correct"],
                time_stats["extractive_total"],
                ext_acc,
            )
        if time_stats["parametric_total"]:
            param_acc = 100 * time_stats["parametric_correct"] / time_stats["parametric_total"]
            log.info(
                "  Parametric (Factoid)   : %d/%d = %.1f%%",
                time_stats["parametric_correct"],
                time_stats["parametric_total"],
                param_acc,
            )
        if time_stats["year_match_total"]:
            year_acc = 100 * time_stats["year_match_correct"] / time_stats["year_match_total"]
            log.info(
                "  Year-only matching     : %d/%d = %.1f%%",
                time_stats["year_match_correct"],
                time_stats["year_match_total"],
                year_acc,
            )
        log.info("  %s", "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate M2 parser pipeline across rules / Qwen / Qwen+fallback modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--test-data",
        type=Path,
        default=Path("experiments/m2_e3_parse/data/splits_qwen/test_prompts.jsonl"),
        help="Test split (prompt-format or raw gold jsonl)",
    )
    ap.add_argument(
        "--gold-data",
        type=Path,
        default=Path("experiments/m2_e3_parse/data/temporal_queries_merged.jsonl"),
        help="Merged gold file for metric computation",
    )
    ap.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("experiments/m2_e3_parse/artifacts/qwen_parser_lora"),
        help="Qwen LoRA adapter directory (optional — omit if adapter not yet trained)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/m2_e3_parse/runs/eval_pipeline"),
        help="Where to write per-mode preds + summary JSON",
    )
    ap.add_argument(
        "--rules-only",
        action="store_true",
        help="Run only the rule-based baseline (no model needed)",
    )
    ap.add_argument(
        "--modes",
        nargs="+",
        default=None,
        choices=["rules", "qwen", "qwen+fallback"],
        help="Specific modes to run (default: all available)",
    )
    args = ap.parse_args(argv)

    # Resolve paths
    test_path = args.test_data.resolve()
    gold_path = args.gold_data.resolve()
    output_dir = args.output_dir.resolve()
    adapter_dir = args.adapter_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not test_path.exists():
        log.error("Test data not found: %s", test_path)
        log.error("Run: python scripts/m2_e3/prepare_qwen_data.py")
        return 1
    if not gold_path.exists():
        log.error("Gold data not found: %s", gold_path)
        return 1

    log.info("Test data  : %s", test_path)
    log.info("Gold data  : %s", gold_path)
    log.info("Output dir : %s", output_dir)

    # Load data
    log.info("Loading test rows...")
    test_rows = read_test_rows(test_path)
    log.info("  %d test rows loaded", len(test_rows))

    log.info("Loading gold index...")
    gold_rows = read_gold_map(gold_path)
    log.info("  %d gold rows indexed", len(gold_rows))

    # Determine which modes to run
    if args.rules_only:
        modes_to_run = ["rules"]
    elif args.modes:
        modes_to_run = args.modes
    else:
        modes_to_run = MODES_ORDER

    # Load Qwen bundle if any non-rules mode is requested
    qwen_bundle = None
    if any(m.startswith("qwen") for m in modes_to_run):
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            log.info("Loading Qwen LoRA adapter from %s ...", adapter_dir)
            qwen_bundle = load_qwen_parser_bundle(adapter_dir)
            if qwen_bundle is None:
                log.warning("Failed to load Qwen adapter — qwen modes will be skipped.")
            else:
                log.info("  Qwen adapter loaded (device: %s)", qwen_bundle["device"])
        else:
            log.warning("Adapter dir not found or missing adapter_config.json: %s", adapter_dir)
            log.warning("Qwen modes will be skipped. Train the model first via Colab.")
            modes_to_run = [m for m in modes_to_run if not m.startswith("qwen")]

    if not modes_to_run:
        log.error("No modes to run.")
        return 1

    # Run each mode
    all_metrics: List[Dict] = []
    mode_preds: Dict[str, List[Dict]] = {}
    for mode in modes_to_run:
        if mode.startswith("qwen") and qwen_bundle is None:
            log.warning("Skipping mode '%s' — Qwen bundle not available.", mode)
            continue
        log.info("\n[MODE: %s]  running %d examples...", mode.upper(), len(test_rows))
        metrics, preds = run_mode(mode, test_rows, gold_rows, qwen_bundle)
        all_metrics.append(metrics)
        mode_preds[mode] = preds

        # Save predictions
        mode_slug = mode.replace("+", "_plus_")
        save_jsonl(output_dir / f"preds_{mode_slug}.jsonl", preds)
        log.info("  intent_accuracy = %.3f", metrics["intent_accuracy"])
        log.info("  intent_micro_f1 = %.3f", metrics["intent_micro_f1"])
        log.info("  frame_f1        = %.3f", metrics["frame_f1"])
        log.info("  span_f1         = %.3f", metrics["span_f1"])
        if metrics["fallback_rate"] > 0:
            log.info(
                "  fallback_rate   = %.1f%%  (%d/%d)",
                metrics["fallback_rate"] * 100,
                metrics["fallback_count"],
                metrics["examples"],
            )

    # Summary table
    print_summary_table(all_metrics)

    # Per-intent breakdown for best mode
    best = max(all_metrics, key=lambda m: m.get("intent_accuracy", 0))
    best_mode = best["mode"]
    print_per_intent_table(best, f"best mode = {best_mode}")
    if best_mode in mode_preds:
        print_frame_field_breakdown(
            mode_preds[best_mode], gold_rows, test_rows, f"best mode = {best_mode}"
        )

    # Save summary JSON
    summary = {
        "modes": all_metrics,
        "test_examples": len(test_rows),
        "gold_examples": len(gold_rows),
    }
    summary_path = output_dir / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log.info("\nSummary written to: %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
