"""M3-E3: Comprehension & Utility Metrics.

This module provides *study tooling* (schemas + aggregations) for Milestone 3
comprehension/utility/cognitive-load evaluation.

Unlike M3-E2 (proxy metrics computed automatically), M3-E3 metrics are primarily
human-study outcomes. The code here focuses on:
- exporting tasks in a consistent format
- ingesting response files
- aggregating responses into the headline metrics from the experimental plan

It intentionally avoids prescribing a specific survey platform.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

Bucket = Literal["point", "interval", "sequence", "causal", "overlap", "other"]


def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def bucket_from_time_scope(time_scope: Any) -> Bucket:
    t = str(time_scope or "").strip().lower()
    if t in {"point", "timestamp", "date"}:
        return "point"
    if t in {"interval", "range", "duration"}:
        return "interval"
    if t in {"sequence", "ordering", "order"}:
        return "sequence"
    if t in {"causal", "causality"}:
        return "causal"
    if t in {"overlap"}:
        return "overlap"
    return "other"


class ExplanationItem(BaseModel):
    """An explanation instance to be shown to a participant."""

    explanation_id: str
    domain: str = "unknown"
    bucket: Bucket = "other"

    query: str = ""
    explanation_text: str = ""
    gold_context: Optional[List[dict]] = None


QuestionType = Literal["mcq", "fill_blank", "timeline", "inference"]


class ComprehensionQuestion(BaseModel):
    """One question tied to a single explanation."""

    question_id: str
    question_type: QuestionType
    prompt: str

    # For MCQ.
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None

    # For rubric-graded questions (timeline/inference) or as guidance.
    rubric: Optional[str] = None


class ComprehensionTask(BaseModel):
    """A comprehension item = explanation + a small set of questions."""

    schema_version: str = "m3-e3a.v1"
    created_at: str = Field(default_factory=_utc_iso_z)

    item: ExplanationItem
    questions: List[ComprehensionQuestion] = Field(default_factory=list)


class ComprehensionResponse(BaseModel):
    """One participant response for one question."""

    participant_id: str
    explanation_id: str
    question_id: str
    question_type: QuestionType

    # Raw answer is stored for audit/regrade.
    answer: Optional[str] = None

    # If provided, this is the authoritative score in [0,1].
    score: Optional[float] = None

    # Optional timing.
    response_time_sec: Optional[float] = None

    # Optional grouping columns (helpful when response file doesn't include tasks).
    domain: Optional[str] = None
    bucket: Optional[Bucket] = None


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if f != f:  # NaN
        return None
    return f


def score_responses_against_tasks(
    responses: Sequence[ComprehensionResponse],
    tasks: Sequence[ComprehensionTask],
) -> List[ComprehensionResponse]:
    """Fill missing `score` fields for MCQ/fill_blank using task answer keys.

    Timeline/inference questions are assumed to be rubric-scored and left as-is.
    """

    task_map: Dict[Tuple[str, str], ComprehensionQuestion] = {}
    meta_map: Dict[str, Tuple[str, Bucket]] = {}
    for t in tasks:
        meta_map[t.item.explanation_id] = (t.item.domain, t.item.bucket)
        for q in t.questions:
            task_map[(t.item.explanation_id, q.question_id)] = q

    scored: List[ComprehensionResponse] = []
    for r in responses:
        rr = r.model_copy(deep=True)

        # Backfill bucket/domain when missing.
        if rr.explanation_id in meta_map:
            dom, buck = meta_map[rr.explanation_id]
            rr.domain = rr.domain or dom
            rr.bucket = rr.bucket or buck

        if rr.score is None and rr.question_type in {"mcq", "fill_blank"}:
            q = task_map.get((rr.explanation_id, rr.question_id))
            if q and q.correct_answer is not None:
                given = (rr.answer or "").strip()
                gold = str(q.correct_answer).strip()
                rr.score = 1.0 if given and (given == gold) else 0.0

        # Clamp to [0,1] if present.
        if rr.score is not None:
            rr.score = max(0.0, min(1.0, float(rr.score)))

        scored.append(rr)
    return scored


def aggregate_comprehension(
    responses: Sequence[ComprehensionResponse],
) -> Dict[str, Any]:
    """Aggregate M3-E3a metrics.

    Returns:
      - overall_accuracy
      - accuracy_by_bucket
      - accuracy_by_domain
      - mean_response_time_sec (if available)
      - time_to_comprehension_sec (mean over responses; proxy)
    """

    scored = [r for r in responses if r.score is not None]

    overall_accuracy = _mean([r.score for r in scored])

    by_bucket: Dict[str, List[float]] = defaultdict(list)
    by_domain: Dict[str, List[float]] = defaultdict(list)
    for r in scored:
        if r.bucket is not None:
            by_bucket[str(r.bucket)].append(float(r.score))
        if r.domain is not None:
            by_domain[str(r.domain)].append(float(r.score))

    accuracy_by_bucket = {k: _mean(v) for k, v in sorted(by_bucket.items())}
    accuracy_by_domain = {k: _mean(v) for k, v in sorted(by_domain.items())}

    times = [_safe_float(r.response_time_sec) for r in responses]
    times = [t for t in times if t is not None and t >= 0]
    mean_time = _mean(times)

    # Proxy: we don't know when the participant "understood"; this is just mean per-question time.
    return {
        "n_responses": len(responses),
        "n_scored": len(scored),
        "overall_accuracy": overall_accuracy,
        "accuracy_by_bucket": accuracy_by_bucket,
        "accuracy_by_domain": accuracy_by_domain,
        "mean_response_time_sec": mean_time,
    }


Condition = Literal["with_explanation", "without_explanation", "control"]


class UtilityTask(BaseModel):
    task_id: str
    domain: str
    prompt: str
    explanation_text: Optional[str] = None


class UtilityResponse(BaseModel):
    participant_id: str
    task_id: str
    domain: str
    condition: Condition

    # Outcomes
    success: Optional[bool] = None
    confidence_1_5: Optional[float] = None
    time_sec: Optional[float] = None
    expert_agreement: Optional[float] = None  # 0..1 (or null)


def aggregate_utility(responses: Sequence[UtilityResponse]) -> Dict[str, Any]:
    """Aggregate M3-E3b metrics."""

    def _mean_bool(values: Iterable[Optional[bool]]) -> Optional[float]:
        vals = [1.0 if v else 0.0 for v in values if v is not None]
        return _mean(vals)

    by_condition: Dict[str, List[UtilityResponse]] = defaultdict(list)
    for r in responses:
        by_condition[r.condition].append(r)

    with_s = by_condition.get("with_explanation", [])
    without_s = by_condition.get("without_explanation", [])
    control_s = by_condition.get("control", [])

    success_with = _mean_bool([r.success for r in with_s])
    success_without = _mean_bool([r.success for r in without_s])
    success_control = _mean_bool([r.success for r in control_s])

    # Relative improvement vs without_explanation.
    success_improvement = None
    if success_with is not None and success_without not in (None, 0.0):
        success_improvement = (success_with - success_without) / success_without

    conf_with = _mean(
        [_safe_float(r.confidence_1_5) for r in with_s if _safe_float(r.confidence_1_5) is not None]
    )
    conf_without = _mean(
        [
            _safe_float(r.confidence_1_5)
            for r in without_s
            if _safe_float(r.confidence_1_5) is not None
        ]
    )
    conf_delta = None
    if conf_with is not None and conf_without is not None:
        conf_delta = conf_with - conf_without

    time_with = _mean(
        [_safe_float(r.time_sec) for r in with_s if _safe_float(r.time_sec) is not None]
    )
    time_without = _mean(
        [_safe_float(r.time_sec) for r in without_s if _safe_float(r.time_sec) is not None]
    )
    time_reduction = None
    if time_with is not None and time_without not in (None, 0.0):
        time_reduction = (time_without - time_with) / time_without

    expert_agreement = _mean(
        [
            _safe_float(r.expert_agreement)
            for r in responses
            if _safe_float(r.expert_agreement) is not None
        ]
    )

    return {
        "n_responses": len(responses),
        "success_rate": {
            "with_explanation": success_with,
            "without_explanation": success_without,
            "control": success_control,
        },
        "success_improvement_vs_without": success_improvement,
        "confidence_mean_1_5": {"with_explanation": conf_with, "without_explanation": conf_without},
        "confidence_delta_vs_without": conf_delta,
        "time_mean_sec": {"with_explanation": time_with, "without_explanation": time_without},
        "time_reduction_vs_without": time_reduction,
        "expert_agreement_mean": expert_agreement,
    }


class CognitiveLoadResponse(BaseModel):
    participant_id: str
    condition: str

    # NASA TLX (0..100) subscales (unweighted average by default).
    tlx_mental: Optional[float] = None
    tlx_physical: Optional[float] = None
    tlx_temporal: Optional[float] = None
    tlx_performance: Optional[float] = None
    tlx_effort: Optional[float] = None
    tlx_frustration: Optional[float] = None

    mental_effort_0_10: Optional[float] = None
    retention_score_0_1: Optional[float] = None
    attention_on_key_ratio_0_1: Optional[float] = None


def _tlx_mean(r: CognitiveLoadResponse) -> Optional[float]:
    parts = [
        _safe_float(r.tlx_mental),
        _safe_float(r.tlx_physical),
        _safe_float(r.tlx_temporal),
        _safe_float(r.tlx_performance),
        _safe_float(r.tlx_effort),
        _safe_float(r.tlx_frustration),
    ]
    parts = [p for p in parts if p is not None]
    return _mean(parts)


def aggregate_cognitive_load(responses: Sequence[CognitiveLoadResponse]) -> Dict[str, Any]:
    """Aggregate M3-E3c metrics."""

    by_condition: Dict[str, List[CognitiveLoadResponse]] = defaultdict(list)
    for r in responses:
        by_condition[str(r.condition)].append(r)

    tlx_by_condition = {
        k: _mean([v for v in (_tlx_mean(r) for r in rs) if v is not None])
        for k, rs in sorted(by_condition.items())
    }
    mental_effort_by_condition = {
        k: _mean([v for v in (_safe_float(r.mental_effort_0_10) for r in rs) if v is not None])
        for k, rs in sorted(by_condition.items())
    }
    retention_by_condition = {
        k: _mean([v for v in (_safe_float(r.retention_score_0_1) for r in rs) if v is not None])
        for k, rs in sorted(by_condition.items())
    }
    attention_by_condition = {
        k: _mean(
            [v for v in (_safe_float(r.attention_on_key_ratio_0_1) for r in rs) if v is not None]
        )
        for k, rs in sorted(by_condition.items())
    }

    return {
        "n_responses": len(responses),
        "tlx_mean_by_condition": tlx_by_condition,
        "mental_effort_mean_by_condition": mental_effort_by_condition,
        "retention_mean_by_condition": retention_by_condition,
        "attention_on_key_mean_by_condition": attention_by_condition,
    }
