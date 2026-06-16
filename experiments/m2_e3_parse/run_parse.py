import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.m2_e3.consistency import validate as validate_prediction


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def find_year_spans(text: str) -> List[Dict]:
    spans: List[Dict] = []
    for m in re.finditer(r"\b(?:19|20)\d{2}\b", text):
        spans.append({"label": "year", "start": m.start(), "end": m.end(), "text": m.group(0)})
    return spans


def find_quarter_spans(text: str, lower: str) -> List[Tuple[Dict, str]]:
    """Return quarter spans and normalized period values."""
    spans: List[Tuple[Dict, str]] = []
    for m in re.finditer(r"\b(q[1-4])\s*(20\d{2})\b", lower):
        qpart, year = m.group(1).upper(), m.group(2)
        start, end = m.start(), m.end()
        spans.append(({"label": "period", "start": start, "end": end, "text": text[start:end]}, f"{year}-{qpart}"))
    # Also catch forms like 2023 Q4
    for m in re.finditer(r"\b(20\d{2})\s*(q[1-4])\b", lower):
        year, qpart = m.group(1), m.group(2).upper()
        start, end = m.start(), m.end()
        spans.append(({"label": "period", "start": start, "end": end, "text": text[start:end]}, f"{year}-{qpart}"))
    return spans


def snake(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.lower())).strip("_")


def extract_region(text: str, lower: str) -> Optional[Dict]:
    m = re.search(r"\bin\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b", text)
    if not m:
        return None
    start, end = m.start(1), m.end(1)
    return {"label": "region", "start": start, "end": end, "text": text[start:end]}


def extract_metric_span(text: str, lower: str) -> Optional[Dict]:
    """Heuristic metric span finder for common lead-in verbs/phrases."""
    patterns = [
        r"(?:total|sum of|sum|count|average|avg|mean|show|calculate|compute|track|plot|graph|chart|aggregate|analyze|analysis|report|difference in|difference of|contrast|compare|versus|vs|predict|forecast|estimate|project|projected)\s+([a-z][\w\s-]{2,40}?)(?:\s+(?:for|in|during|between|from|over)|\s*[:])",
        r"^\s*([A-Za-z][\w\s-]{2,40}?)\s+(?:between|from)\s+(?:19|20)\d{2}\b",
    ]
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            raw = m.group(1).strip()
            start = lower.find(raw)
            if start != -1:
                end = start + len(raw)
                return {"label": "metric", "start": start, "end": end, "text": text[start:end]}
    return None


def extract_range(lower: str, text: str) -> Optional[Tuple[str, str, Dict, Dict]]:
    m = re.search(r"(?:between|from)\s+(19|20)\d{2}\s*(?:and|to|-)\s*(19|20)\d{2}", lower)
    if not m:
        return None
    years = re.findall(r"(?:19|20)\d{2}", m.group(0))
    if len(years) < 2:
        return None
    y1, y2 = years[0], years[1]
    start_pos = lower.find(y1)
    end_pos = lower.find(y2)
    return (
        y1,
        y2,
        {"label": "start_date", "start": start_pos, "end": start_pos + len(y1), "text": text[start_pos:start_pos+len(y1)]},
        {"label": "end_date", "start": end_pos, "end": end_pos + len(y2), "text": text[end_pos:end_pos+len(y2)]},
    )


def load_intent_bundle(model_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    if model_dir is None:
        return None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        import torch  # type: ignore
    except ImportError:
        return None

    labels_path = model_dir / "labels.json"
    if not labels_path.exists():
        return None
    labels: List[str] = json.loads(labels_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return {"labels": labels, "tokenizer": tokenizer, "model": model, "device": device}


def predict_intents(bundle: Dict[str, Any], text: str, threshold: float = 0.5) -> List[str]:
    import torch  # type: ignore

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze(0)
    labels = bundle["labels"]
    return [labels[i] for i, score in enumerate(probs.tolist()) if score >= threshold]


def load_parser_bundle(model_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    if model_dir is None:
        return None
    try:
        from transformers import AutoTokenizer, T5ForConditionalGeneration  # type: ignore
        import torch  # type: ignore
    except ImportError:
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "device": device}


def predict_parser(bundle: Dict[str, Any], text: str, max_new_tokens: int = 256) -> Dict[str, Any]:
    import torch  # type: ignore

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError:
        parsed = {}
    return {
        "spans": parsed.get("spans", []),
        "frame": parsed.get("frame", {}),
        "intent_labels": parsed.get("intent_labels", []),
        "raw": decoded,
    }


def parse_row_rules(row: Dict) -> Dict:
    text = row["text"]
    lower = text.lower()
    spans: List[Dict] = []
    frame: Dict = {}
    intent = []

    # Add detected years and quarters
    spans.extend(find_year_spans(text))
    quarter_spans = find_quarter_spans(text, lower)
    spans.extend([qs[0] for qs in quarter_spans])

    metric_span = extract_metric_span(text, lower)
    metric_norm = snake(metric_span["text"]) if metric_span else ""
    if metric_span:
        spans.append(metric_span)

    region_span = extract_region(text, lower)
    if region_span:
        spans.append(region_span)

    range_info = extract_range(lower, text)

    # Causal
    if re.search(r"\b(cause|caused|lead to|led to|due to|because of|trigger|triggered)\b", lower):
        # Try "did the X cause the Y" pattern first
        cause_match = re.search(r"did the (.+?) cause the (.+?)\?", lower)
        effect_match = None
        ctext = etext = ""
        if cause_match:
            ctext = cause_match.group(1).strip()
            etext = cause_match.group(2).strip()
        else:
            m = re.search(r"(.+?)\s+(?:cause|caused|lead to|led to|triggered?)\s+(.+?)\??$", lower)
            if m:
                ctext, etext = m.group(1).strip(), m.group(2).strip()
            else:
                # Handle "X due to Y" / "because of"
                m = re.search(r"(.+?)\s+(?:due to|because of)\s+(.+?)\??$", lower)
                if m:
                    etext, ctext = m.group(1).strip(), m.group(2).strip()
        if ctext:
            cstart = lower.find(ctext)
            spans.append({"label": "cause", "start": cstart, "end": cstart + len(ctext), "text": text[cstart:cstart+len(ctext)]})
        if etext:
            estart = lower.find(etext)
            spans.append({"label": "effect", "start": estart, "end": estart + len(etext), "text": text[estart:estart+len(etext)]})
        frame = {"cause": ctext or "", "effect": etext or ""}
        intent = ["causal"]

    # Explanation (why ... after/before)
    elif lower.startswith("why") or "why did" in lower:
        anchor_match = re.search(r"(?:after|before)\s+(.+?)\??$", lower)
        metric_match = re.search(r"why (?:did|was|were)?\s+(.+?)\s+(?:after|before)", lower)
        atext = anchor_match.group(1).strip() if anchor_match else ""
        mtext = metric_match.group(1).strip() if metric_match else ""
        if mtext:
            mstart = lower.find(mtext)
            spans.append({"label": "metric", "start": mstart, "end": mstart + len(mtext), "text": text[mstart:mstart+len(mtext)]})
        if atext:
            astart = lower.find(atext)
            spans.append({"label": "event", "start": astart, "end": astart + len(atext), "text": text[astart:astart+len(atext)]})
        relation = "after" if "after" in lower else "before" if "before" in lower else "after"
        frame = {"metric": mtext or metric_norm, "anchor_event": atext, "relation": relation}
        intent = ["explanation", "sequence"]

    # Overlap
    elif re.search(r"\b(overlap|overlapping|concurrent|simultaneous|conflicting)\b", lower):
        event_text = ""
        m = re.search(r"(overlap(?:ping)?|concurrent|simultaneous|conflicting)\s+([a-z][\w\s-]{2,40})", lower)
        if m:
            event_text = m.group(2).strip()
            estart = lower.find(event_text)
            spans.append({"label": "event", "start": estart, "end": estart + len(event_text), "text": text[estart:estart+len(event_text)]})
        period_val = quarter_spans[0][1] if quarter_spans else (range_info[0] if range_info else None)
        frame = {"event": event_text or "event", "period": period_val or ""}
        intent = ["overlap"]

    # Comparative
    elif re.search(r"\b(vs|versus)\b", lower) or "difference" in lower or "contrast" in lower or "compare" in lower:
        years = [m.group(0) for m in re.finditer(r"\b(?:19|20)\d{2}\b", text)]
        if len(years) >= 2:
            a_year, b_year = years[0], years[1]
            a_pos = lower.find(a_year.lower())
            b_pos = lower.find(b_year.lower())
            spans.append({"label": "a", "start": a_pos, "end": a_pos + len(a_year), "text": a_year})
            spans.append({"label": "b", "start": b_pos, "end": b_pos + len(b_year), "text": b_year})
        if metric_span is None:
            # try to capture metric before years
            m = re.search(r"compare\s+([a-z][\w\s-]{2,40}?)(?:\s+in|\s*:|\s+between)", lower)
            if m:
                raw = m.group(1).strip()
                mstart = lower.find(raw)
                spans.append({"label": "metric", "start": mstart, "end": mstart + len(raw), "text": text[mstart:mstart+len(raw)]})
                metric_norm = snake(raw)
        frame = {"metric": metric_norm, "a": years[0] if years else "", "b": years[1] if len(years) > 1 else ""}
        intent = ["comparative"]

    # Interval (between/from ranges)
    elif range_info:
        y1, y2, sspan, espan = range_info
        spans.extend([sspan, espan])
        frame = {"metric": metric_norm, "start": y1, "end": y2}
        intent = ["interval"]

    # Aggregation (period and maybe region)
    elif quarter_spans or region_span or "total" in lower or "sum" in lower or "aggregate" in lower or "average" in lower or "count" in lower:
        period_val = quarter_spans[0][1] if quarter_spans else ""
        if not period_val:
            # single year as period
            years = [m.group(0) for m in re.finditer(r"\b(?:19|20)\d{2}\b", text)]
            if years:
                period_val = years[0]
        if region_span:
            frame = {"metric": metric_norm, "period": period_val, "region": region_span["text"]}
        else:
            frame = {"metric": metric_norm, "period": period_val}
        intent = ["aggregation"]

    # Prediction
    elif re.search(r"\b(predict|forecast|estimate|project|projected)\b", lower):
        year_match = re.search(r"\b(?:19|20)\d{2}\b", text)
        if year_match:
            ytext = year_match.group(0)
            ystart = lower.find(ytext.lower())
            spans.append({"label": "date", "start": ystart, "end": ystart + len(ytext), "text": ytext})
            frame = {"metric": metric_norm, "date": ytext}
        intent = ["prediction"]

    # Sequence (before/after/followed)
    elif re.search(r"\b(before|after|followed|preceded)\b", lower):
        rel = "after" if "after" in lower else "before" if "before" in lower else "followed"
        anchor_match = re.search(r"(?:after|before|followed)\s+the\s+(.+?)\??$", lower)
        if not anchor_match:
            anchor_match = re.search(r"(?:after|before|followed)\s+(.+?)\??$", lower)
        if not anchor_match:
            anchor_match = re.search(r"(?:preceded)\s+the\s+(.+?)\??$", lower)
        if not anchor_match:
            anchor_match = re.search(r"(?:preceded)\s+(.+?)\??$", lower)
        if anchor_match:
            atext = anchor_match.group(1).strip()
            astart = lower.find(atext)
            spans.append({"label": "anchor", "start": astart, "end": astart + len(atext), "text": text[astart:astart+len(atext)]})
            frame = {"anchor": atext, "relation": rel}
        else:
            frame = {"anchor": "", "relation": rel}
        intent = ["sequence"]

    # Point in time fallback
    else:
        if lower.startswith("when") or "date of" in lower or "when did" in lower:
            event_match = re.search(r"when did (.+?)(?:\?|$)", lower)
            if event_match:
                etext = event_match.group(1).strip()
                estart = lower.find(etext)
                spans.append({"label": "event", "start": estart, "end": estart + len(etext), "text": text[estart:estart+len(etext)]})
                frame = {"event": text[estart:estart+len(etext)]}
            else:
                # Fallback: tag the main noun phrase after "the" if present
                m = re.search(r"(?:date of|when was|when is|when were|when)\s+the\s+(.+?)(?:\?|$)", lower)
                if m:
                    etext = m.group(1).strip()
                    estart = lower.find(etext)
                    spans.append({"label": "event", "start": estart, "end": estart + len(etext), "text": text[estart:estart+len(etext)]})
                    frame = {"event": text[estart:estart+len(etext)]}
            intent = ["point_in_time"]
        elif metric_norm and re.search(r"\b(19|20)\d{2}\b", lower):
            year_match = re.search(r"\b(19|20)\d{2}\b", lower)
            if year_match:
                ytext = year_match.group(0)
                ystart = lower.find(ytext)
                spans.append({"label": "date", "start": ystart, "end": ystart + len(ytext), "text": text[ystart:ystart+len(ytext)]})
                frame = {"metric": metric_norm, "date": ytext}
                intent = ["point_in_time"]

    return {
        "id": row["id"],
        "text": text,
        "spans": spans,
        "frame": frame,
        "intent_labels": intent or row.get("intent_labels", []),
        "source": "rule-parser"
    }


def parse_row(row: Dict, bundles: Dict[str, Any], threshold: float, fallback_on_error: bool) -> Tuple[Dict, Dict]:
    justification = {"id": row.get("id"), "source": "rules", "notes": [], "validation": {}}
    result: Optional[Dict[str, Any]] = None

    parser_bundle = bundles.get("parser")
    intent_bundle = bundles.get("intent")

    if parser_bundle is not None:
        model_pred = predict_parser(parser_bundle, row["text"])
        intents = model_pred.get("intent_labels", []) or []
        if intent_bundle is not None:
            intents = sorted(set(intents + predict_intents(intent_bundle, row["text"], threshold=threshold)))
        spans = model_pred.get("spans", []) or []
        frame = model_pred.get("frame", {}) or {}
        result = {
            "id": row["id"],
            "text": row["text"],
            "spans": spans,
            "frame": frame,
            "intent_labels": intents or row.get("intent_labels", []),
            "source": "model-parser",
            "raw_model": model_pred.get("raw"),
        }
        justification["source"] = "model"
        justification["notes"].append("model_parser_used")

    if result is None:
        result = parse_row_rules(row)
        justification["notes"].append("fallback_rules_no_model")

    validation = validate_prediction(result)

    def needs_fallback(pred: Dict, validation: Dict) -> bool:
        intents = pred.get("intent_labels", []) or []
        if validation.get("errors"):
            return True
        # Overlap predictions without event or period are rarely usable
        if "overlap" in intents and any(w in validation.get("warnings", []) for w in ["overlap_missing_event", "overlap_missing_period"]):
            return True
        # Empty spans/frame signals the model failed to structure output
        if not pred.get("spans") and not pred.get("frame"):
            return True
        return False

    if fallback_on_error and justification.get("source") == "model" and needs_fallback(result, validation):
        model_validation = validation
        rule_result = parse_row_rules(row)
        rule_validation = validate_prediction(rule_result)
        justification["notes"].append("fallback_rules_on_error")
        justification["validation_model"] = model_validation
        justification["validation_rules"] = rule_validation
        justification["validation_final"] = rule_validation
        result = rule_result

        # Preserve the model validation at the top level so callers can see why
        # fallback happened, while keeping rule validation separately.
        justification["validation"] = model_validation
        return result, justification

    justification["validation"] = validation

    return result, justification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to gold jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--use-model", action="store_true", help="Use fine-tuned intent/parser models if available")
    parser.add_argument("--intent-model-dir", type=Path, default=Path("experiments/m2_e3_parse/artifacts/intent"))
    parser.add_argument("--parser-model-dir", type=Path, default=Path("experiments/m2_e3_parse/artifacts/parser"))
    parser.add_argument("--model-threshold", type=float, default=0.25, help="Threshold for multi-label intent predictions")
    parser.add_argument("--fallback-on-error", action="store_true", help="Fallback to rules when model output fails validation")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    run_id = uuid.uuid4().hex
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    bundles = {
        "intent": load_intent_bundle(args.intent_model_dir) if args.use_model else None,
        "parser": load_parser_bundle(args.parser_model_dir) if args.use_model else None,
    }

    preds: List[Dict] = []
    justifications: List[Dict] = []
    for row in rows:
        pred, just = parse_row(row, bundles, threshold=args.model_threshold, fallback_on_error=args.fallback_on_error)
        preds.append(pred)
        justifications.append(just)

    save_jsonl(run_dir / "preds.jsonl", preds)
    save_jsonl(run_dir / "justifications.jsonl", justifications)

    metrics = {
        "run_id": run_id,
        "examples": len(preds),
        "notes": "Model + rules pipeline" if args.use_model else "Rule-based parser; replace with learned model for quality."
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved parsed outputs to {run_dir}")


if __name__ == "__main__":
    main()
