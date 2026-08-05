import re
from typing import Dict, List

YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
QUARTER_RE = re.compile(r"^(?:19|20)\d{2}-Q[1-4]$")


def check_period(period: str) -> List[str]:
    issues = []
    if not period:
        issues.append("missing_period")
        return issues
    if YEAR_RE.match(period):
        return issues
    if QUARTER_RE.match(period):
        return issues
    issues.append("invalid_period_format")
    return issues


def validate(pred: Dict) -> Dict:
    errors: List[str] = []
    warnings: List[str] = []
    frame = pred.get("frame", {}) or {}
    intents = pred.get("intent_labels", []) or []

    if "interval" in intents:
        if not frame.get("start") or not frame.get("end"):
            errors.append("interval_missing_start_or_end")
        else:
            try:
                if int(frame["start"]) > int(frame["end"]):
                    warnings.append("interval_start_gt_end")
            except Exception:
                warnings.append("interval_non_numeric_bounds")
    if "aggregation" in intents:
        if not frame.get("metric"):
            errors.append("aggregation_missing_metric")
        if not frame.get("period"):
            warnings.append("aggregation_missing_period")
        else:
            warnings.extend(check_period(frame.get("period", "")))
    if "comparative" in intents:
        if not frame.get("a") or not frame.get("b"):
            errors.append("comparative_missing_a_or_b")
        if not frame.get("metric"):
            warnings.append("comparative_missing_metric")
    if "causal" in intents:
        if not frame.get("cause") or not frame.get("effect"):
            errors.append("causal_missing_cause_or_effect")
    if "sequence" in intents:
        if not frame.get("anchor") and not frame.get("anchor_event"):
            warnings.append("sequence_missing_anchor")
        if not frame.get("relation"):
            warnings.append("sequence_missing_relation")
    if "overlap" in intents:
        if not frame.get("event"):
            warnings.append("overlap_missing_event")
        if not frame.get("period"):
            warnings.append("overlap_missing_period")
        else:
            warnings.extend(check_period(frame.get("period", "")))
    if "point_in_time" in intents:
        if not frame.get("time") and not frame.get("date"):
            warnings.append("point_missing_time_or_date")
    if "prediction" in intents:
        if not frame.get("date"):
            warnings.append("prediction_missing_date")

    return {"errors": errors, "warnings": warnings}
