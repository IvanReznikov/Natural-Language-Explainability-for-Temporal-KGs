"""M3-E4: Efficiency & Consistency Metrics.

This module provides schemas and aggregation utilities for M3-E4a..d.
It intentionally focuses on data interchange + summarization rather than
binding to a specific generation stack.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field

from .m3_e3 import bucket_from_time_scope


Bucket = Literal["point", "interval", "sequence", "causal", "overlap", "other"]


def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
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


def _percentile(values: Iterable[float], pct: float) -> Optional[float]:
    vals = sorted([v for v in values if v is not None])
    if not vals:
        return None
    if pct <= 0:
        return vals[0]
    if pct >= 100:
        return vals[-1]
    idx = int(round((pct / 100.0) * (len(vals) - 1)))
    return vals[min(max(idx, 0), len(vals) - 1)]


# -------------------------
# M3-E4a Efficiency
# -------------------------


class EfficiencyScenario(BaseModel):
    schema_version: str = "m3-e4a.scenario.v1"
    created_at: str = Field(default_factory=_utc_iso_z)

    scenario_id: str
    domain: str = "unknown"
    bucket: Bucket = "other"
    complexity_level: int = 3  # 1..5
    time_scope: str = ""

    query: str = ""
    explanation_text: str = ""
    gold_facts: Optional[List[dict]] = None


class EfficiencyRun(BaseModel):
    schema_version: str = "m3-e4a.run.v1"
    created_at: str = Field(default_factory=_utc_iso_z)

    scenario_id: str
    method: str
    complexity_level: int

    latency_ms: Optional[float] = None
    tokens_out: Optional[float] = None
    cost: Optional[float] = None
    quality_proxy: Optional[float] = None


def aggregate_efficiency(runs: Iterable[EfficiencyRun]) -> Dict[str, Any]:
    by_method: Dict[str, List[EfficiencyRun]] = defaultdict(list)
    by_method_level: Dict[str, Dict[int, List[EfficiencyRun]]] = defaultdict(lambda: defaultdict(list))

    for r in runs:
        by_method[str(r.method)].append(r)
        by_method_level[str(r.method)][int(r.complexity_level)].append(r)

    def _summarize(items: List[EfficiencyRun]) -> Dict[str, Any]:
        latencies = [_safe_float(r.latency_ms) for r in items if _safe_float(r.latency_ms) is not None]
        mean_latency = _mean(latencies)
        throughput = (1000.0 / mean_latency) if mean_latency else None
        return {
            "n": len(items),
            "latency_ms": {
                "mean": mean_latency,
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
            },
            "throughput_per_sec": throughput,
            "tokens_out_mean": _mean([_safe_float(r.tokens_out) for r in items]),
            "cost_mean": _mean([_safe_float(r.cost) for r in items]),
            "quality_proxy_mean": _mean([_safe_float(r.quality_proxy) for r in items]),
        }

    summary: Dict[str, Any] = {"by_method": {}, "by_method_level": {}}
    for m, items in sorted(by_method.items()):
        summary["by_method"][m] = _summarize(items)
    for m, level_map in sorted(by_method_level.items()):
        summary["by_method_level"][m] = {str(k): _summarize(v) for k, v in sorted(level_map.items())}

    return summary


# -------------------------
# M3-E4b Consistency under revisions
# -------------------------


RevisionType = Literal["date_correction", "add_causal", "contradiction", "removal"]


class ConsistencyFact(BaseModel):
    schema_version: str = "m3-e4b.fact.v1"
    created_at: str = Field(default_factory=_utc_iso_z)

    fact_id: str
    domain: str = "unknown"
    query: str = ""
    base_explanation: str = ""
    gold_facts: Optional[List[dict]] = None


class ConsistencyRevision(BaseModel):
    schema_version: str = "m3-e4b.revision.v1"
    created_at: str = Field(default_factory=_utc_iso_z)

    revision_id: str
    fact_id: str
    revision_type: RevisionType
    delta: Dict[str, Any] = Field(default_factory=dict)


class ConsistencyResult(BaseModel):
    schema_version: str = "m3-e4b.result.v1"
    created_at: str = Field(default_factory=_utc_iso_z)

    revision_id: str
    method: str

    update_accuracy: Optional[float] = None
    contradiction_detected: Optional[bool] = None
    coherence_rating_1_5: Optional[float] = None
    resolution_time_sec: Optional[float] = None


def aggregate_consistency(results: Iterable[ConsistencyResult]) -> Dict[str, Any]:
    by_method: Dict[str, List[ConsistencyResult]] = defaultdict(list)
    for r in results:
        by_method[str(r.method)].append(r)

    def _summarize(items: List[ConsistencyResult]) -> Dict[str, Any]:
        update_acc = _mean([_safe_float(r.update_accuracy) for r in items])
        contradiction_rate = _mean([1.0 if r.contradiction_detected else 0.0 for r in items if r.contradiction_detected is not None])
        coherence = _mean([_safe_float(r.coherence_rating_1_5) for r in items])
        resolution = _mean([_safe_float(r.resolution_time_sec) for r in items])
        return {
            "n": len(items),
            "update_accuracy_mean": update_acc,
            "contradiction_detection_rate": contradiction_rate,
            "coherence_rating_mean_1_5": coherence,
            "resolution_time_mean_sec": resolution,
        }

    summary: Dict[str, Any] = {"by_method": {}}
    for m, items in sorted(by_method.items()):
        summary["by_method"][m] = _summarize(items)

    return summary


# -------------------------
# M3-E4c Cross-explanation coherence
# -------------------------


class CoherenceScenario(BaseModel):
    schema_version: str = "m3-e4c.scenario.v1"
    created_at: str = Field(default_factory=_utc_iso_z)

    scenario_id: str
    domain: str = "unknown"
    query: str = ""
    gold_facts: Optional[List[dict]] = None


class CoherenceExplanation(BaseModel):
    schema_version: str = "m3-e4c.explanation.v1"
    created_at: str = Field(default_factory=_utc_iso_z)

    scenario_id: str
    style: str
    text: str


class CoherenceScore(BaseModel):
    schema_version: str = "m3-e4c.score.v1"
    created_at: str = Field(default_factory=_utc_iso_z)

    scenario_id: str
    style: Optional[str] = None

    semantic_consistency: Optional[float] = None
    narrative_consistency: Optional[float] = None
    logical_consistency: Optional[float] = None


def aggregate_coherence(scores: Iterable[CoherenceScore]) -> Dict[str, Any]:
    by_style: Dict[str, List[CoherenceScore]] = defaultdict(list)
    overall: List[CoherenceScore] = []
    all_scores: List[CoherenceScore] = []
    for s in scores:
        all_scores.append(s)
        if s.style:
            by_style[str(s.style)].append(s)
        else:
            overall.append(s)

    if not overall:
        overall = list(all_scores)

    def _summarize(items: List[CoherenceScore]) -> Dict[str, Any]:
        return {
            "n": len(items),
            "semantic_consistency_mean": _mean([_safe_float(r.semantic_consistency) for r in items]),
            "narrative_consistency_mean": _mean([_safe_float(r.narrative_consistency) for r in items]),
            "logical_consistency_mean": _mean([_safe_float(r.logical_consistency) for r in items]),
        }

    summary = {"overall": _summarize(overall), "by_style": {}}
    for st, items in sorted(by_style.items()):
        summary["by_style"][st] = _summarize(items)

    return summary


# -------------------------
# M3-E4d Granularity robustness
# -------------------------


Granularity = Literal["seconds", "minutes", "hours", "days", "months", "years", "decades"]


class GranularityScenario(BaseModel):
    schema_version: str = "m3-e4d.scenario.v1"
    created_at: str = Field(default_factory=_utc_iso_z)

    scenario_id: str
    domain: str = "unknown"
    bucket: Bucket = "other"
    query: str = ""
    gold_facts: Optional[List[dict]] = None


class GranularityVariant(BaseModel):
    schema_version: str = "m3-e4d.variant.v1"
    created_at: str = Field(default_factory=_utc_iso_z)

    scenario_id: str
    granularity: Granularity
    text: str

    quality_score: Optional[float] = None
    length_chars: Optional[int] = None


def aggregate_granularity(variants: Iterable[GranularityVariant]) -> Dict[str, Any]:
    by_granularity: Dict[str, List[GranularityVariant]] = defaultdict(list)
    for v in variants:
        by_granularity[str(v.granularity)].append(v)

    summary: Dict[str, Any] = {"by_granularity": {}}
    for g, items in sorted(by_granularity.items()):
        summary["by_granularity"][g] = {
            "n": len(items),
            "quality_mean": _mean([_safe_float(r.quality_score) for r in items]),
            "length_chars_mean": _mean([float(r.length_chars) for r in items if r.length_chars is not None]),
        }
    return summary


__all__ = [
    "EfficiencyScenario",
    "EfficiencyRun",
    "ConsistencyFact",
    "ConsistencyRevision",
    "ConsistencyResult",
    "CoherenceScenario",
    "CoherenceExplanation",
    "CoherenceScore",
    "GranularityScenario",
    "GranularityVariant",
    "aggregate_efficiency",
    "aggregate_consistency",
    "aggregate_coherence",
    "aggregate_granularity",
    "bucket_from_time_scope",
]
