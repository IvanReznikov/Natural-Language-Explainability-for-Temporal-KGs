import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_template(frame: Dict) -> str:
    # Rule templater for seed data coverage
    if not frame:
        return "UNKNOWN()"

    if "event" in frame and "period" in frame:
        return f"OVERLAP(event='{frame['event']}', period='{frame['period']}')"

    if {"metric", "start", "end"}.issubset(frame.keys()):
        return f"INTERVAL(metric='{frame['metric']}', start='{frame['start']}', end='{frame['end']}')"

    if "metric" in frame and "period" in frame:
        return f"INTERVAL(metric='{frame['metric']}', period='{frame['period']}')"

    if "event" in frame and "time" in frame:
        return f"POINT(event='{frame['event']}', date='{frame['time']}')"

    if "metric" in frame and "date" in frame:
        return f"PREDICT(metric='{frame['metric']}', date='{frame['date']}')"

    if {"cause", "effect"}.issubset(frame.keys()):
        return f"CAUSAL(cause='{frame['cause']}', effect='{frame['effect']}')"

    if {"metric", "a", "b"}.issubset(frame.keys()):
        return f"COMPARE(metric='{frame['metric']}', a='{frame['a']}', b='{frame['b']}')"

    if {"metric", "period", "region"}.issubset(frame.keys()):
        return f"AGG(metric='{frame['metric']}', period='{frame['period']}', region='{frame['region']}')"

    if ("anchor" in frame or "anchor_event" in frame) and "relation" in frame:
        anchor_val = frame.get("anchor") or frame.get("anchor_event")
        return f"SEQUENCE(anchor='{anchor_val}', relation='{frame['relation']}')"

    return "GENERIC(frame)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to gold jsonl with frame/canonical_query")
    parser.add_argument("--pred", type=Path, default=None, help="Optional SRL predictions to use instead of gold")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--use-gold", action="store_true", help="Force using gold frames even if pred is provided")
    args = parser.parse_args()

    gold_rows = {row["id"]: row for row in load_jsonl(args.data)}
    pred_rows = {row["id"]: row for row in load_jsonl(args.pred)} if args.pred and not args.use_gold else {}

    run_id = uuid.uuid4().hex
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    matches = 0
    total = 0

    for qid, gold in gold_rows.items():
        frame = gold.get("frame", {})
        if qid in pred_rows and not args.use_gold:
            frame = pred_rows[qid].get("frame", frame)
        templated = build_template(frame)
        outputs.append({"id": qid, "canonical_query": templated})
        total += 1
        if templated == gold.get("canonical_query"):
            matches += 1

    save_jsonl(run_dir / "outputs.jsonl", outputs)

    metrics = {
        "run_id": run_id,
        "examples": total,
        "template_accuracy": matches / total if total else 0.0,
        "notes": "Stub constructor uses simple rule patterns; replace with real templater."
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved constructed queries to {run_dir}")


if __name__ == "__main__":
    main()
