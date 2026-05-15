import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def span_key(span: Dict) -> Tuple[str, int, int]:
    return span.get("label"), span.get("start"), span.get("end")


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

    metrics = {
        "examples": total,
        "span_f1": span_f1,
        "frame_exact": frame_exact,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
