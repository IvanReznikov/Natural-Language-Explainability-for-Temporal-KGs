import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def span_key(span: Dict) -> Tuple[str, int, int]:
    return str(span.get("label", "")), int(span.get("start", 0)), int(span.get("end", 0))


def f1(gold: List[Tuple], pred: List[Tuple]) -> float:
    gold_set, pred_set = set(gold), set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    gold_rows = {row["id"]: row for row in load_jsonl(args.gold)}
    pred_rows = {row["id"]: row for row in load_jsonl(args.pred)}

    span_scores = []
    frame_matches = 0
    total = 0

    for qid, gold_row in gold_rows.items():
        if qid not in pred_rows:
            continue
        pred_row = pred_rows[qid]
        total += 1

        gold_spans = [span_key(s) for s in gold_row.get("spans", [])]
        pred_spans = [span_key(s) for s in pred_row.get("spans", [])]
        span_scores.append(f1(gold_spans, pred_spans))

        if gold_row.get("frame", {}) == pred_row.get("frame", {}):
            frame_matches += 1

    span_f1 = sum(span_scores) / len(span_scores) if span_scores else 0.0
    frame_exact = frame_matches / total if total else 0.0

    # Calculate intent classification metrics
    all_intent_labels = set()
    for row in gold_rows.values():
        all_intent_labels.update(row.get("intent_labels", []))
    intent_labels_set = sorted(list(all_intent_labels))

    tp_c = {lbl: 0 for lbl in intent_labels_set}
    fp_c = {lbl: 0 for lbl in intent_labels_set}
    fn_c = {lbl: 0 for lbl in intent_labels_set}

    for qid, gold_row in gold_rows.items():
        if qid not in pred_rows:
            continue
        pred_row = pred_rows[qid]

        gold_intents = set(gold_row.get("intent_labels", []))
        pred_intents = set(pred_row.get("intent_labels", []))

        for lbl in intent_labels_set:
            g_has = lbl in gold_intents
            p_has = lbl in pred_intents
            if g_has and p_has:
                tp_c[lbl] += 1
            elif p_has and not g_has:
                fp_c[lbl] += 1
            elif g_has and not p_has:
                fn_c[lbl] += 1

    per_intent_metrics = {}
    macro_f1_sum = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for lbl in intent_labels_set:
        tp = tp_c[lbl]
        fp = fp_c[lbl]
        fn = fn_c[lbl]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_intent_metrics[lbl] = {"precision": prec, "recall": rec, "f1": f1_val, "support": tp + fn}
        macro_f1_sum += f1_val

    macro_f1 = macro_f1_sum / len(intent_labels_set) if intent_labels_set else 0.0
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    metrics = {
        "examples": total,
        "span_f1": span_f1,
        "frame_exact": frame_exact,
        "intent_micro_f1": micro_f1,
        "intent_macro_f1": macro_f1,
        "per_intent_metrics": per_intent_metrics
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
