import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.m2_e3.consistency import validate as validate_prediction

# ---------------------------------------------------------------------------
# Canonical intent map used by the Qwen parser (uppercase → raw label).
# Keeps the parse_row_rules output and Qwen output in the same label space.
# ---------------------------------------------------------------------------
_QWEN_TO_RAW: Dict[str, str] = {
    "POINT":    "point_in_time",
    "INTERVAL": "interval",
    "PREDICT":  "prediction",
    "AGG":      "aggregation",
    "SEQUENCE": "sequence",
    "CAUSAL":   "causal",
    "COMPARE":  "comparative",
    "OVERLAP":  "overlap",
}

_QWEN_SYSTEM_PROMPT = (
    "You are a temporal knowledge-graph query parser. "
    "Given a natural-language question, output a JSON object with exactly two keys:\n"
    "  - \"intent\": one of [POINT, INTERVAL, PREDICT, AGG, SEQUENCE, CAUSAL, COMPARE, OVERLAP]\n"
    "  - \"frame\": a dict of slot-value pairs extracted from the question\n"
    "Output ONLY valid JSON. No markdown, no extra text."
)


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
        r"^\s*(?!(?:total|sum|average|avg|count|aggregate|explain|predict|forecast|estimate|project)\b)([a-z][\w\s-]{2,40}?)\s+(?:total|sum|average|avg|count)\s+(?:for|in|during|at)\b",
        r"(?:increase|decrease|rise|drop|fall|dip|surge|spike|change)\s+in\s+([a-z][\w\s-]{2,40}?)(?:\s+\b(?:for|in|during|between|from|over|after|before|due to|caused by|because of lock contention|because of)\b|\s*[:])",
        r"(?:total|sum of|sum|count of|count|average|avg|mean|show|calculate|compute|track|plot|graph|chart|aggregate|analyze|analysis|report|difference in|difference of|contrast|compare|versus|vs|predict|forecast|estimate|project|projected|explain|explain the)\s+([a-z][\w\s-]{2,40}?)(?:\s+\b(?:for|in|during|between|from|over|after|before|due to|caused by|because of lock contention|because of)\b|\s*[:])",
        r"^\s*([A-Za-z][\w\s-]{2,40}?)\s+(?:between|from|in|for|during|at)\s+(?:19|20)\d{2}\b",
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
    """Load a T5 seq2seq parser bundle (legacy).  For Qwen use load_qwen_parser_bundle."""
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
    return {"tokenizer": tokenizer, "model": model, "device": device, "kind": "t5"}


def load_qwen_parser_bundle(adapter_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load the Qwen2.5-0.5B-Instruct model with a LoRA adapter fine-tuned for frame parsing.

    The adapter directory must contain:
    - adapter_config.json   (written by peft.save_pretrained)
    - adapter_model.safetensors (or adapter_model.bin)
    - tokenizer files

    Falls back gracefully to None if transformers/peft are not installed
    or the adapter directory does not look valid.
    """
    if adapter_dir is None:
        return None
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        return None
    try:
        import torch  # type: ignore
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        from peft import PeftModel  # type: ignore
    except ImportError:
        return None

    try:
        # Read base model name from adapter config
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        base_model_name = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
        # pyrefly: ignore [missing-attribute]
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model = model.merge_and_unload()  # merge weights for faster inference
        model.to(device)
        model.eval()
        return {"tokenizer": tokenizer, "model": model, "device": device, "kind": "qwen"}
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(f"Could not load Qwen parser from {adapter_dir}: {exc}")
        return None


def predict_parser(bundle: Dict[str, Any], text: str, max_new_tokens: int = 256) -> Dict[str, Any]:
    """Dispatch to the correct predict function based on bundle kind."""
    kind = bundle.get("kind", "t5")
    if kind == "qwen":
        return predict_qwen_parser(bundle, text, max_new_tokens=max_new_tokens)
    return _predict_t5_parser(bundle, text, max_new_tokens=max_new_tokens)


def _predict_t5_parser(bundle: Dict[str, Any], text: str, max_new_tokens: int = 256) -> Dict[str, Any]:
    """T5 seq2seq parser (legacy)."""
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


def predict_qwen_parser(bundle: Dict[str, Any], text: str, max_new_tokens: int = 128) -> Dict[str, Any]:
    """Qwen2.5 LoRA parser.  Returns the same dict shape as _predict_t5_parser."""
    import torch  # type: ignore

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]

    messages = [
        {"role": "system", "content": _QWEN_SYSTEM_PROMPT},
        {"role": "user",   "content": text.strip()},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][enc["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # Parse the JSON output
    try:
        obj = json.loads(decoded)
    except json.JSONDecodeError:
        # Try to extract first JSON-looking substring
        m = re.search(r"\{.*\}", decoded, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            obj = {}

    # Map canonical uppercase intent back to raw label used by the rest of the pipeline
    canonical_intent = obj.get("intent", "")
    raw_intent = _QWEN_TO_RAW.get(canonical_intent, canonical_intent.lower())

    frame = obj.get("frame") or {}

    return {
        "spans": [],  # Qwen parser outputs frame-only; spans derived from frame keys
        "frame": frame,
        "intent_labels": [raw_intent] if raw_intent else [],
        "raw": decoded,
        "canonical_intent": canonical_intent,
    }


def parse_row_rules(row: Dict) -> Dict:
    text = row["text"]
    lower = text.lower()
    lower = re.sub(r"^(?:versus report|comparison report|compare|contrast|difference)\s*:\s*", "", lower)
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
    # Causal (only if it doesn't contain aggregation keywords before the causal keyword)
    causal_trigger = re.search(r"\b(cause|caused|lead to|led to|due to|because of|trigger|triggered|improve|improved|reduce|reduced|prevent|prevented|fix|fixed)\b", lower)
    is_agg_before_causal = False
    if causal_trigger:
        pre_causal = lower[:causal_trigger.start()]
        is_agg_before_causal = any(re.search(r"\b" + x + r"\b", pre_causal) for x in ["total", "sum", "average", "avg", "count"])

    # Explanation (why ... after/before, or what caused ... after/before)
    if (lower.startswith("why") or "why did" in lower or "what caused" in lower or "what triggers" in lower or "what triggered" in lower or lower.startswith("explain") or lower.startswith("explaining")) and ("after" in lower or "before" in lower):
        anchor_match = re.search(r"(?:after|before)\s+(.+?)\??$", lower)
        metric_match = re.search(r"(?:why|what caused|what triggers|what triggered|explain|explaining)(?:\s+did|\s+was|\s+were|\s+the)?\s+(.+?)\s+(?:after|before)", lower)
        atext = anchor_match.group(1).strip() if anchor_match else ""
        mtext = metric_match.group(1).strip() if metric_match else ""
        # Strip leading determiners from anchor ("the patch" -> "patch")
        atext = re.sub(r"^(the|a|an)\s+", "", atext).strip()
        # Strip trailing change-verb/noun from metric ("latency rise" -> "latency", "errors drop" -> "errors")
        mtext = re.sub(r"\s+(?:rise|rise|drop|dip|surge|spike|fall|increase|decrease|decline|jump|grow|shrink|degrade|improve|worsen|to\s+rise|to\s+drop|to\s+increase|to\s+decrease|to\s+surge|to\s+spike|to\s+fall)\s*$", "", mtext).strip()
        if not mtext and metric_norm:
            mtext = metric_norm
        if mtext:
            mstart = lower.find(mtext)
            if mstart != -1:
                spans.append({"label": "metric", "start": mstart, "end": mstart + len(mtext), "text": text[mstart:mstart+len(mtext)]})
        if atext:
            astart = lower.find(atext)
            if astart != -1:
                spans.append({"label": "event", "start": astart, "end": astart + len(atext), "text": text[astart:astart+len(atext)]})
        relation = "after" if "after" in lower else "before" if "before" in lower else "after"
        frame = {"metric": mtext or metric_norm, "anchor_event": atext, "relation": relation}
        intent = ["explanation", "sequence"]

    elif causal_trigger and not is_agg_before_causal:
        # Try "did the X cause the Y" pattern first
        cause_match = re.search(r"did the (.+?) (?:cause|improve|improved|reduce|reduced|prevent|prevented|trigger|triggered|fix|fixed) (?:the\s+)?(.+?)\?", lower)
        ctext = etext = ""
        if cause_match:
            ctext = cause_match.group(1).strip()
            etext = cause_match.group(2).strip()
        else:
            m = re.search(r"(.+?)\s+(?:cause|caused|lead to|led to|triggered?|improve|improved|reduce|reduced|prevent|prevented|fix|fixed)\s+(?:the\s+)?(.+?)\??$", lower)
            if m:
                ctext, etext = m.group(1).strip(), m.group(2).strip()
            else:
                # Handle "X due to Y" / "because of"
                m = re.search(r"(.+?)\s+(?:due to|because of)\s+(?:the\s+)?(.+?)\??$", lower)
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

    # Overlap
    elif re.search(r"\b(overlap|overlapping|concurrent|simultaneous|conflicting)\b", lower):
        event_text = ""
        m = re.search(r"(overlap(?:ping)?|concurrent|simultaneous|conflicting)\s+([a-z][\w\s-]{2,40})", lower)
        if m:
            event_text = m.group(2).strip()
            # Strip trailing period-reference ("failures in Q4 2023" -> "failures")
            event_text = re.sub(r"\s+(?:in|during|for)\s+(?:q[1-4]\s+)?(?:19|20)\d{2}.*$", "", event_text).strip()
            estart = lower.find(event_text)
            if estart != -1:
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

    # Interval (between/from ranges or during a period)
    elif range_info or "during" in lower:
        if range_info:
            y1, y2, sspan, espan = range_info
            spans.extend([sspan, espan])
            frame = {"metric": metric_norm, "start": y1, "end": y2}
        else:
            period_match = re.search(r"during\s+(?:the\s+)?(.+?)(?:\?|$)", lower)
            ptext = period_match.group(1).strip() if period_match else ""
            ptext = re.sub(r"\b(?:19|20)\d{2}\b", "", ptext).strip()
            frame = {"metric": metric_norm, "period": ptext}
        intent = ["interval"]

    # Prediction
    elif re.search(r"\b(predict|forecast|estimate|project|projected|will|expect|expected)\b", lower):
        period_val = quarter_spans[0][1] if quarter_spans else ""
        year_match = re.search(r"\b(?:19|20)\d{2}\b", text)
        
        # Heuristic metric fallback
        if not metric_norm:
            m_text = lower
            m_text = re.sub(r"^(?:predict|forecast|estimate|project|projected|forecasts|what\s+will|will|expected|expect)\s+", "", m_text)
            m_text = re.sub(r"\b(?:q[1-4]\s+)?(?:19|20)\d{2}\b", "", m_text)
            m_text = re.sub(r"\b(?:19|20)\d{2}\s+q[1-4]\b", "", m_text)
            m_text = re.sub(r"^(?:total|sum of|sum|count of|count|average|avg|mean|by|in|for|during|of|at|the|a|an)\s+", "", m_text)
            m_text = re.sub(r"\s+(?:by|in|for|during|of|at|the|a|an)$", "", m_text)
            m_text = re.sub(r"\s+be$", "", m_text)
            m_text = m_text.strip()
            if m_text:
                metric_norm = snake(m_text)
                
        if period_val:
            frame = {"metric": metric_norm, "period": period_val}
        elif year_match:
            ytext = year_match.group(0)
            ystart = lower.find(ytext.lower())
            spans.append({"label": "date", "start": ystart, "end": ystart + len(ytext), "text": ytext})
            frame = {"metric": metric_norm, "date": ytext}
        else:
            frame = {"metric": metric_norm}
        intent = ["prediction"]

    # Aggregation (period and maybe region)
    elif quarter_spans or region_span or re.search(r"^\s*(?:the\s+)?(?:total|sum|average|avg|count|aggregate)\b", lower):
        period_val = quarter_spans[0][1] if quarter_spans else ""
        if not period_val:
            # single year as period
            years = [m.group(0) for m in re.finditer(r"\b(?:19|20)\d{2}\b", text)]
            if years:
                period_val = years[0]
                
        # Heuristic metric fallback
        if not metric_norm:
            m_text = lower
            m_text = re.sub(r"^(?:total|sum of|sum|count of|count|average|avg|mean|show|calculate|compute)\s+", "", m_text)
            m_text = re.sub(r"\b(?:q[1-4]\s+)?(?:19|20)\d{2}\b", "", m_text)
            m_text = re.sub(r"\b(?:19|20)\d{2}\s+q[1-4]\b", "", m_text)
            m_text = re.sub(r"^(?:by|in|for|during|of|at|the|a|an)\s+", "", m_text)
            m_text = re.sub(r"\s+(?:by|in|for|during|of|at|the|a|an)$", "", m_text)
            m_text = m_text.strip()
            if m_text:
                metric_norm = snake(m_text)
                
        if region_span:
            frame = {"metric": metric_norm, "period": period_val, "region": region_span["text"]}
        else:
            frame = {"metric": metric_norm, "period": period_val}
        intent = ["aggregation"]

    # Sequence (before/after/followed)
    elif re.search(r"\b(before|after|followed|preceded?)\b", lower):
        rel = "after" if "after" in lower else "before" if ("before" in lower or "precede" in lower) else "followed"
        anchor_match = re.search(r"(?:after|before|followed|preceded?)\s+the\s+(.+?)\??$", lower)
        if not anchor_match:
            anchor_match = re.search(r"(?:after|before|followed|preceded?)\s+(.+?)\??$", lower)
        if anchor_match:
            atext = anchor_match.group(1).strip()
            astart = lower.find(atext)
            spans.append({"label": "anchor", "start": astart, "end": astart + len(atext), "text": text[astart:astart+len(atext)]})
            frame = {"metric": metric_norm, "anchor": atext, "relation": rel}
        else:
            frame = {"metric": metric_norm, "anchor": "", "relation": rel}
        intent = ["sequence"]

    # Point in time fallback
    else:
        if lower.startswith("when") or "date of" in lower or "when did" in lower or "launch date" in lower or "release date" in lower or "foundation date" in lower:
            event_match = re.search(r"when did (.+?)(?:\?|$)", lower)
            if event_match:
                etext = event_match.group(1).strip()
                # Strip trailing verb if query is "when did X occur/happen/start"
                etext = re.sub(r"\s+(?:occur|happen|start|end|begin|finish|take place)\s*$", "", etext).strip()
                estart = lower.find(etext)
                if estart != -1:
                    spans.append({"label": "event", "start": estart, "end": estart + len(etext), "text": text[estart:estart+len(etext)]})
                    frame = {"event": text[estart:estart+len(etext)]}
            else:
                # "date of / launch date of / when was the ..."
                m = re.search(r"(?:launch date of|release date of|foundation date of|date of|when was|when is|when were|when)\s+the\s+(.+?)(?:\?|$)", lower)
                if not m:
                    m = re.search(r"(?:launch date of|release date of|foundation date of|date of|when was|when is)\s+(.+?)(?:\?|$)", lower)
                if m:
                    etext = m.group(1).strip()
                    # Strip trailing verb phrases ("inaugurated", "signed", "built", "founded", "launched", "opened", "invented", "completed", "declared", "announced", "established")
                    etext = re.sub(r"\s+(?:inaugurated|signed|built|founded|launched|opened|invented|completed|declared|announced|established)\s*$", "", etext).strip()
                    estart = lower.find(etext)
                    if estart != -1:
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
    parser = argparse.ArgumentParser(
        description="Run the M2 parser pipeline (rules / Qwen model / hybrid).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to gold/test jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/m2_e3_parse/runs"))
    parser.add_argument(
        "--mode",
        choices=["rules", "qwen", "qwen+fallback", "t5", "t5+fallback"],
        default="rules",
        help=(
            "rules          — rule-based parser only (no model)\n"
            "qwen           — Qwen LoRA model only (no rule fallback)\n"
            "qwen+fallback  — Qwen model with rule fallback on validation errors  [recommended]\n"
            "t5             — legacy T5 model only\n"
            "t5+fallback    — legacy T5 model with rule fallback"
        ),
    )
    # Qwen adapter
    parser.add_argument(
        "--qwen-adapter-dir",
        type=Path,
        default=Path("experiments/m2_e3_parse/artifacts/qwen_parser_lora"),
        help="Directory containing the Qwen LoRA adapter (adapter_config.json must exist)",
    )
    # Legacy T5 / intent models
    parser.add_argument("--intent-model-dir", type=Path,
                        default=Path("experiments/m2_e3_parse/artifacts/intent"))
    parser.add_argument("--parser-model-dir", type=Path,
                        default=Path("experiments/m2_e3_parse/artifacts/parser"))
    parser.add_argument("--model-threshold", type=float, default=0.25)
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    run_id = uuid.uuid4().hex
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    use_qwen = args.mode.startswith("qwen")
    use_t5 = args.mode.startswith("t5")
    use_fallback = args.mode.endswith("+fallback")

    print(f"Mode: {args.mode}  |  rows: {len(rows)}  |  run_id: {run_id}")

    bundles: Dict[str, Any] = {"intent": None, "parser": None}
    if use_qwen:
        print(f"Loading Qwen LoRA parser from {args.qwen_adapter_dir} ...")
        qwen_bundle = load_qwen_parser_bundle(args.qwen_adapter_dir)
        if qwen_bundle is None:
            print("WARNING: Qwen bundle could not be loaded — falling back to rules.")
        bundles["parser"] = qwen_bundle
    elif use_t5:
        print(f"Loading T5 parser from {args.parser_model_dir} ...")
        bundles["intent"] = load_intent_bundle(args.intent_model_dir)
        bundles["parser"] = load_parser_bundle(args.parser_model_dir)

    preds: List[Dict] = []
    justifications: List[Dict] = []
    for row in rows:
        pred, just = parse_row(
            row, bundles,
            threshold=args.model_threshold,
            fallback_on_error=use_fallback,
        )
        preds.append(pred)
        justifications.append(just)

    save_jsonl(run_dir / "preds.jsonl", preds)
    save_jsonl(run_dir / "justifications.jsonl", justifications)

    fallback_count = sum(1 for j in justifications if "fallback_rules_on_error" in j.get("notes", []))
    metrics = {
        "run_id": run_id,
        "mode": args.mode,
        "examples": len(preds),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(preds), 4) if preds else 0.0,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved {len(preds)} predictions to {run_dir}")
    print(f"Fallback rate: {metrics['fallback_rate']:.1%}  ({fallback_count}/{len(preds)})")


if __name__ == "__main__":
    main()
