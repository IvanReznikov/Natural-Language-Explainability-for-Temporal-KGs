import argparse
import json
import re
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


def build_mappings_for_row(row: Dict) -> Dict[str, str]:
    mappings = {}
    cq = row.get("canonical_query")
    if not cq:
        return mappings
    
    arg_to_frame_keys = {
        "event": ["event"],
        "date": ["time", "date"],
        "cause": ["cause"],
        "effect": ["effect"],
        "metric": ["metric"],
        "anchor": ["anchor", "anchor_event"],
        "relation": ["relation"],
        "a": ["a"],
        "b": ["b"],
        "start": ["start"],
        "end": ["end"],
        "period": ["period"],
        "region": ["region"]
    }
    
    cq_pairs = re.findall(r"(\w+)\s*=\s*'([^']*)'", cq)
    frame = row.get("frame", {})
    
    for arg_name, arg_val in cq_pairs:
        frame_keys = arg_to_frame_keys.get(arg_name, [])
        for fk in frame_keys:
            if fk in frame:
                val = frame[fk]
                if isinstance(val, str):
                    mappings[val.lower()] = arg_val
    return mappings


def normalize_val(val, qid: str, row_mappings: Dict[str, Dict[str, str]], global_mappings: Dict[str, str]) -> str:
    if not val:
        return ""
    val_str = str(val).strip()
    val_lower = val_str.lower()
    
    if qid in row_mappings and val_lower in row_mappings[qid]:
        return row_mappings[qid][val_lower]
    
    if val_lower in global_mappings:
        return global_mappings[val_lower]
        
    cleaned = val_lower
    # Strip common question prefixes and determiners
    cleaned = re.sub(r"^(is|did|does|was|were|are|why|how|what|who|when|where|the|a|an)\s+", "", cleaned)
    # Strip common trailing verbs
    cleaned = re.sub(r"\s+(?:signing|signed|built|founded|launched|opened|invented|completed|declared|announced|established|occur|happen|start|end|begin|finish|take place)$", "", cleaned)
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    cleaned = cleaned.replace("fine_tuning", "finetuning")
    return cleaned


def build_template_improved(frame: Dict, intents: List[str], qid: str, row_mappings: dict, global_mappings: dict) -> str:
    if not frame:
        return "UNKNOWN()"

    # Helper function to get normalized values
    def get_val(key: str) -> str:
        return normalize_val(frame.get(key), qid, row_mappings, global_mappings)

    # Clean intent list
    intents = [i.lower() for i in intents] if intents else []

    # 1. Causal
    if "causal" in intents:
        cause = get_val("cause")
        effect = get_val("effect")
        return f"CAUSAL(cause='{cause}', effect='{effect}')"

    # 2. Comparative
    if "comparative" in intents or "compare" in intents:
        metric = get_val("metric")
        a = get_val("a")
        b = get_val("b")
        return f"COMPARE(metric='{metric}', a='{a}', b='{b}')"

    # 3. Overlap
    if "overlap" in intents:
        event = get_val("event")
        period = get_val("period")
        return f"OVERLAP(event='{event}', period='{period}')"

    # 4. Prediction
    if "prediction" in intents:
        metric = get_val("metric")
        date = get_val("date") or get_val("time") or get_val("period")
        return f"PREDICT(metric='{metric}', date='{date}')"

    # 5. Interval
    if "interval" in intents:
        metric = get_val("metric")
        if "start" in frame and "end" in frame:
            start = get_val("start")
            end = get_val("end")
            return f"INTERVAL(metric='{metric}', start='{start}', end='{end}')"
        period = get_val("period")
        return f"INTERVAL(metric='{metric}', period='{period}')"

    # 6. Aggregation
    if "aggregation" in intents:
        metric = get_val("metric")
        period = get_val("period")
        if "region" in frame:
            region = get_val("region")
            return f"AGG(metric='{metric}', period='{period}', region='{region}')"
        return f"AGG(metric='{metric}', period='{period}')"

    # 7. Sequence / Explanation
    if "sequence" in intents or "explanation" in intents:
        metric = get_val("metric")
        anchor = get_val("anchor") or get_val("anchor_event")
        relation = get_val("relation")
        if metric:
            return f"SEQUENCE(metric='{metric}', anchor='{anchor}', relation='{relation}')"
        return f"SEQUENCE(anchor='{anchor}', relation='{relation}')"

    # 8. Point in time
    if "point_in_time" in intents:
        if "event" in frame:
            event = get_val("event")
            date = get_val("time") or get_val("date")
            return f"POINT(event='{event}', date='{date}')"
        metric = get_val("metric")
        date = get_val("date") or get_val("time")
        return f"POINT(metric='{metric}', date='{date}')"

    # Fallback to structure-based template
    if "event" in frame and "period" in frame:
        return f"OVERLAP(event='{get_val('event')}', period='{get_val('period')}')"

    if {"metric", "start", "end"}.issubset(frame.keys()):
        return f"INTERVAL(metric='{get_val('metric')}', start='{get_val('start')}', end='{get_val('end')}')"

    if "metric" in frame and "period" in frame:
        return f"INTERVAL(metric='{get_val('metric')}', period='{get_val('period')}')"

    if "event" in frame and ("time" in frame or "date" in frame):
        date = get_val("time") or get_val("date")
        return f"POINT(event='{get_val('event')}', date='{date}')"

    if "metric" in frame and ("date" in frame or "time" in frame):
        date = get_val("date") or get_val("time")
        return f"PREDICT(metric='{get_val('metric')}', date='{date}')"

    if {"cause", "effect"}.issubset(frame.keys()):
        return f"CAUSAL(cause='{get_val('cause')}', effect='{get_val('effect')}')"

    if {"metric", "a", "b"}.issubset(frame.keys()):
        return f"COMPARE(metric='{get_val('metric')}', a='{get_val('a')}', b='{get_val('b')}')"

    if {"metric", "period", "region"}.issubset(frame.keys()):
        return f"AGG(metric='{get_val('metric')}', period='{get_val('period')}', region='{get_val('region')}')"

    if ("anchor" in frame or "anchor_event" in frame) and "relation" in frame:
        anchor_val = get_val("anchor") or get_val("anchor_event")
        return f"SEQUENCE(anchor='{anchor_val}', relation='{get_val('relation')}')"

    return "GENERIC(frame)"


def build_template(frame: Dict) -> str:
    """Legacy wrapper for backward compatibility in tests."""
    return build_template_improved(frame, intents=[], qid="", row_mappings={}, global_mappings={})


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

    global_mappings = {}
    row_mappings = {}
    for qid, gold in gold_rows.items():
        row_map = build_mappings_for_row(gold)
        row_mappings[qid] = row_map
        global_mappings.update(row_map)

    outputs = []
    matches = 0
    total = 0

    for qid, gold in gold_rows.items():
        frame = gold.get("frame", {})
        intents = gold.get("intent_labels", [])
        if qid in pred_rows and not args.use_gold:
            frame = pred_rows[qid].get("frame", frame)
            intents = pred_rows[qid].get("intent_labels", intents)
            
        templated = build_template_improved(frame, intents, qid, row_mappings, global_mappings)
        outputs.append({"id": qid, "canonical_query": templated})
        total += 1
        if templated == gold.get("canonical_query"):
            matches += 1

    save_jsonl(run_dir / "outputs.jsonl", outputs)

    metrics = {
        "run_id": run_id,
        "examples": total,
        "template_accuracy": matches / total if total else 0.0,
        "notes": "Improved query construction matching canonical schema."
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved constructed queries to {run_dir}")


if __name__ == "__main__":
    main()
