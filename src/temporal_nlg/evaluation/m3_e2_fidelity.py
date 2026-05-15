"""M3-E2: Explanation Fidelity Metrics.

This module implements lightweight, automatic proxy metrics for explanation fidelity
across temporal relationship types (point-in-time, interval, sequence, causality).

Notes:
- Some M3-E2 metrics require human/expert validation (e.g., ambiguity resolution,
  causal link correctness). Those are represented as `None` in the returned metrics.
- The scorer is designed to work with the project JSONL format used in
  `data/jsonls/temporal_graph.jsonl`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import re

from temporal_nlg.temporal_expr import TemporalNormalizer, TemporalTagger


_YEAR_RE = re.compile(r"\b(1\d{3}|2\d{3})\b")


_POINT_SCOPES = {"point", "timestamp", "date"}
_INTERVAL_SCOPES = {"interval", "range", "duration"}
_SEQUENCE_SCOPES = {"sequence", "ordering", "order"}
_CAUSAL_SCOPES = {"causal", "causality"}
_OVERLAP_SCOPES = {"overlap"}

_CAUSAL_MARKERS = ("because", "due to", "as a result", "resulted in", "led to", "caused")
_SEQUENCE_MARKERS = ("then", "next", "after", "before", "followed by", "subsequently")
_INTERVAL_MARKERS = ("from", "to", "between", "until", "during")
_OVERLAP_MARKERS = ("while", "simultaneously", "at the same time", "overlap")

_COMPARISON_MARKERS = (
    "compared",
    "whereas",
    "while",
    "in contrast",
    "longer",
    "shorter",
    "earlier",
    "later",
    "before",
    "after",
)

_HEDGE_MARKERS = ("might", "may", "possibly", "likely", "uncertain", "suggest", "appears")
_CERTAINTY_MARKERS = ("definitely", "certainly", "clearly", "undoubtedly")


@dataclass(frozen=True)
class TemporalMention:
    kind: str  # 'date' | 'range' | 'duration' | 'year'
    value: str


def _safe_lower(text: Any) -> str:
    return str(text).lower() if text is not None else ""


def _extract_years(text: str) -> List[int]:
    return [int(m.group(1)) for m in _YEAR_RE.finditer(text)]


def _parse_iso_date(value: str) -> Optional[date]:
    value = value.strip()
    try:
        # Accept YYYY-MM-DD
        return datetime.fromisoformat(value).date()
    except Exception:
        return None


def _parse_iso_range(value: str) -> Optional[Tuple[date, date]]:
    value = value.strip()
    if "/" not in value:
        return None
    parts = [p.strip() for p in value.split("/", 1)]
    if len(parts) != 2:
        return None
    start = _parse_iso_date(parts[0])
    end = _parse_iso_date(parts[1])
    if not start or not end:
        return None
    return start, end


def _duration_to_days(iso_duration: str) -> Optional[float]:
    """Approximate ISO-like duration (e.g., P2Y3M10D, PT4H) into days."""

    if not iso_duration or not iso_duration.startswith("P"):
        return None

    # Very small, permissive parser for the subset produced by TemporalNormalizer.
    # We intentionally approximate: Y=365, M=30, W=7, D=1; time components in days.
    text = iso_duration[1:]
    date_part, _, time_part = text.partition("T")

    def _num(s: str) -> float:
        try:
            return float(s)
        except Exception:
            return 0.0

    total_days = 0.0

    for m in re.finditer(r"(\d+(?:\.\d+)?)(Y|M|W|D)", date_part):
        val = _num(m.group(1))
        unit = m.group(2)
        if unit == "Y":
            total_days += val * 365.0
        elif unit == "M":
            total_days += val * 30.0
        elif unit == "W":
            total_days += val * 7.0
        elif unit == "D":
            total_days += val

    seconds = 0.0
    for m in re.finditer(r"(\d+(?:\.\d+)?)(H|M|S)", time_part):
        val = _num(m.group(1))
        unit = m.group(2)
        if unit == "H":
            seconds += val * 3600.0
        elif unit == "M":
            seconds += val * 60.0
        elif unit == "S":
            seconds += val

    total_days += seconds / 86400.0
    return total_days if total_days > 0 else None


def _timestamp_tier_score(diff_days: int) -> float:
    if diff_days <= 1:
        return 1.0
    if diff_days <= 7:
        return 0.8
    return 0.0


def _best_date_match_score(pred_dates: Sequence[date], gold_dates: Sequence[date], pred_years: Sequence[int]) -> float:
    if not gold_dates:
        return 1.0

    best = 0.0
    if pred_dates:
        for pd in pred_dates:
            for gd in gold_dates:
                diff = abs((pd - gd).days)
                best = max(best, _timestamp_tier_score(diff))
        return best

    # If no parseable dates were found, fall back to year-only matching.
    if pred_years:
        gold_years = {d.year for d in gold_dates}
        if any(y in gold_years for y in pred_years):
            return 0.5

    return 0.0


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _unique_preserve(seq: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _extract_entities_from_facts(gold_facts: Sequence[dict]) -> List[str]:
    entities: List[str] = []
    for f in gold_facts:
        for k in ("subject", "object", "value"):
            v = f.get(k)
            if not v:
                continue
            s = str(v).strip()
            if not s:
                continue
            # Skip pure numbers.
            if re.fullmatch(r"[+-]?(\d+(?:\.\d+)?)", s):
                continue
            entities.append(s)
    # Keep only a manageable, deduped list.
    return _unique_preserve([e.lower() for e in entities])[:25]


def _extract_fact_atoms(gold_facts: Sequence[dict]) -> List[str]:
    """Extract a simple set of 'fact atoms' used for context relevance.

    We keep this intentionally heuristic/robust, because the dataset isn't
    guaranteed to follow a strict schema.
    """

    atoms: List[str] = []
    for f in gold_facts:
        subj = str(f.get("subject") or "").strip().lower()
        obj = str(f.get("object") or "").strip().lower()
        val = str(f.get("value") or "").strip().lower()
        rel = str(f.get("relation") or "").strip().lower()

        for s in (subj, obj, val):
            if s and not re.fullmatch(r"[+-]?(\d+(?:\.\d+)?)", s):
                atoms.append(s)

        if rel and rel not in {"", "none", "n/a"}:
            atoms.append(rel)

        start = str(f.get("start") or "").strip()
        end = str(f.get("end") or "").strip()
        for d in (start, end):
            parsed = _parse_iso_date(d) if d else None
            if parsed is not None:
                atoms.append(str(parsed.year))

    return _unique_preserve([a for a in atoms if a])[:50]


def _fact_recall(text: str, atoms: Sequence[str]) -> float:
    """Proxy for Context Relevance: fraction of gold 'atoms' present in text."""
    if not atoms:
        return 1.0
    text_l = text.lower()
    hit = sum(1 for a in atoms if a and a in text_l)
    return hit / len(atoms)


def _entity_coverage(text: str, entities: Sequence[str]) -> float:
    if not entities:
        return 1.0
    text_l = text.lower()
    hit = sum(1 for e in entities if e and e in text_l)
    return hit / len(entities)


def _unnecessary_detail_score(pred_text: str, gold_years: Sequence[int]) -> float:
    pred_years = set(_extract_years(pred_text))
    if not pred_years:
        return 1.0
    gold_set = set(gold_years)
    extra = pred_years - gold_set
    ratio = len(extra) / len(pred_years) if pred_years else 0.0
    return max(0.0, 1.0 - ratio)


def _unnecessary_detail_ratio(pred_text: str, gold_years: Sequence[int]) -> float:
    """Raw unnecessary-detail ratio based on extra years (0..1)."""
    pred_years = set(_extract_years(pred_text))
    if not pred_years:
        return 0.0
    gold_set = set(gold_years)
    extra = pred_years - gold_set
    return len(extra) / len(pred_years)


def _comparison_clarity(text: str) -> float:
    text_l = text.lower()
    return float(any(m in text_l for m in _COMPARISON_MARKERS))


def _marker_presence(text: str, markers: Sequence[str]) -> float:
    text_l = text.lower()
    return float(any(m in text_l for m in markers))


def _confidence_signal(text: str) -> Optional[float]:
    """Proxy for confidence calibration: -1 hedged, +1 overconfident, 0 neutral."""
    text_l = text.lower()
    has_hedge = any(m in text_l for m in _HEDGE_MARKERS)
    has_certainty = any(m in text_l for m in _CERTAINTY_MARKERS)
    if has_hedge and not has_certainty:
        return -1.0
    if has_certainty and not has_hedge:
        return 1.0
    if has_hedge and has_certainty:
        return 0.0
    return 0.0


def _scope_bucket(time_scope: str) -> str:
    t = _safe_lower(time_scope).strip()
    if t in _POINT_SCOPES:
        return "point"
    if t in _INTERVAL_SCOPES:
        return "interval"
    if t in _SEQUENCE_SCOPES:
        return "sequence"
    if t in _CAUSAL_SCOPES:
        return "causal"
    if t in _OVERLAP_SCOPES:
        return "overlap"
    return "other"


class M3E2FidelityEvaluator:
    """Compute automatic proxy metrics for M3-E2."""

    def __init__(self) -> None:
        self.tagger = TemporalTagger()
        self.normalizer = TemporalNormalizer()

    def extract_temporals(self, text: str) -> Tuple[List[TemporalMention], List[date], List[Tuple[date, date]], List[float], List[int]]:
        mentions: List[TemporalMention] = []
        dates: List[date] = []
        ranges: List[Tuple[date, date]] = []
        durations_days: List[float] = []

        for expr in self.tagger.tag(text):
            normalized = self.normalizer.normalize(expr)
            if not normalized.normalized:
                continue
            value = normalized.normalized

            if value.startswith("P"):
                mentions.append(TemporalMention(kind="duration", value=value))
                d = _duration_to_days(value)
                if d is not None:
                    durations_days.append(d)
                continue

            rng = _parse_iso_range(value)
            if rng is not None:
                mentions.append(TemporalMention(kind="range", value=value))
                ranges.append(rng)
                continue

            dte = _parse_iso_date(value)
            if dte is not None:
                mentions.append(TemporalMention(kind="date", value=value))
                dates.append(dte)
                continue

        years = _extract_years(text)
        for y in years:
            mentions.append(TemporalMention(kind="year", value=str(y)))

        return mentions, dates, ranges, durations_days, years

    def evaluate_example(self, record: Dict[str, Any], prediction_text: str) -> Dict[str, Any]:
        time_scope = record.get("time_scope", "")
        bucket = _scope_bucket(time_scope)

        gold_facts: List[dict] = list(record.get("gold_facts") or [])
        entities = _extract_entities_from_facts(gold_facts)
        fact_atoms = _extract_fact_atoms(gold_facts)

        gold_dates: List[date] = []
        gold_ranges: List[Tuple[date, date]] = []
        for f in gold_facts:
            s = f.get("start")
            e = f.get("end")
            if s and e and str(s).strip() and str(e).strip():
                sd = _parse_iso_date(str(s))
                ed = _parse_iso_date(str(e))
                if sd and ed:
                    if sd == ed:
                        gold_dates.append(sd)
                    else:
                        gold_ranges.append((sd, ed))
            elif s and str(s).strip():
                sd = _parse_iso_date(str(s))
                if sd:
                    gold_dates.append(sd)

        gold_years: List[int] = []
        for d in gold_dates:
            gold_years.append(d.year)
        for sd, ed in gold_ranges:
            gold_years.extend([sd.year, ed.year])

        mentions, pred_dates, pred_ranges, pred_durations, pred_years = self.extract_temporals(prediction_text)

        # Shared proxy metrics.
        # M3-E2a: Context Relevance proxy
        context_relevance = _fact_recall(prediction_text, fact_atoms)
        # M3-E2a: Unnecessary Detail Ratio (raw) + inversely-scored variant
        unnecessary_detail_ratio = _unnecessary_detail_ratio(prediction_text, gold_years)
        unnecessary_detail = max(0.0, 1.0 - unnecessary_detail_ratio)

        # Generic proxies
        entity_coverage = _entity_coverage(prediction_text, entities)
        confidence_signal = _confidence_signal(prediction_text)

        metrics: Dict[str, Any] = {
            "id": record.get("id"),
            "domain": record.get("domain"),
            "time_scope": time_scope,
            "bucket": bucket,
            "context_relevance": context_relevance,
            "entity_coverage": entity_coverage,
            "unnecessary_detail_score": unnecessary_detail,
            "unnecessary_detail_ratio": unnecessary_detail_ratio,
            "ambiguity_resolution": None,
            "confidence_signal": confidence_signal,
            "num_temporal_mentions": len(mentions),
        }

        # Point-in-time
        if bucket == "point":
            metrics.update(
                {
                    "timestamp_accuracy": _best_date_match_score(pred_dates, gold_dates, pred_years),
                    "context_relevance_proxy": context_relevance,
                    "unnecessary_detail_proxy": unnecessary_detail,
                }
            )
            return metrics

        # Interval
        if bucket == "interval":
            # Choose the best matching interval among gold and predicted ranges.
            boundary_scores: List[float] = []
            duration_scores: List[float] = []

            candidate_pred_ranges: List[Tuple[date, date]] = list(pred_ranges)
            # Common case: the model writes "from <date> to <date>" which the tagger
            # captures as two DATE expressions (not a single RANGE). Derive candidate
            # ranges from all observed dates.
            if len(pred_dates) >= 2:
                for i in range(len(pred_dates)):
                    for j in range(i + 1, len(pred_dates)):
                        a, b = pred_dates[i], pred_dates[j]
                        if a <= b:
                            candidate_pred_ranges.append((a, b))
                        else:
                            candidate_pred_ranges.append((b, a))

            for gs, ge in gold_ranges:
                best_for_gold = 0.0
                best_duration = None
                gold_days = abs((ge - gs).days)

                for ps, pe in candidate_pred_ranges:
                    start_score = _timestamp_tier_score(abs((ps - gs).days))
                    end_score = _timestamp_tier_score(abs((pe - ge).days))
                    best_for_gold = max(best_for_gold, 0.5 * (start_score + end_score))

                    pred_days = abs((pe - ps).days)
                    best_duration = pred_days

                if best_for_gold > 0:
                    boundary_scores.append(best_for_gold)

                if gold_days > 0:
                    # Duration correctness: prefer explicit predicted durations if present.
                    candidate_durations: List[float] = []
                    if best_duration is not None:
                        candidate_durations.append(best_duration)
                    candidate_durations.extend(pred_durations)

                    if candidate_durations:
                        # Pick closest duration.
                        diff = min(abs(cd - gold_days) for cd in candidate_durations)
                        # Tiered like timestamp: <=1 day exact, <=7 days close.
                        duration_scores.append(_timestamp_tier_score(int(round(diff))))

            boundary_accuracy = _mean(boundary_scores)
            duration_correctness = _mean(duration_scores)

            # Proxy: overlap representation (we cannot fully validate overlaps automatically)
            overlap_representation = _marker_presence(prediction_text, _OVERLAP_MARKERS)

            metrics.update(
                {
                    "boundary_accuracy": boundary_accuracy if boundary_accuracy is not None else 0.0,
                    "duration_correctness": duration_correctness if duration_correctness is not None else 0.0,
                    "overlap_representation": overlap_representation,
                    "interval_comparison_clarity": _comparison_clarity(prediction_text),
                    "interval_marker_presence": _marker_presence(prediction_text, _INTERVAL_MARKERS),
                }
            )
            return metrics

        # Sequence
        if bucket == "sequence":
            # Build a gold sequence from dated facts (subject order by start date).
            dated: List[Tuple[date, str]] = []
            for f in gold_facts:
                subj = f.get("subject")
                s = f.get("start")
                if subj and s:
                    sd = _parse_iso_date(str(s))
                    if sd:
                        dated.append((sd, str(subj).strip().lower()))

            dated.sort(key=lambda x: x[0])
            gold_items = _unique_preserve([s for _, s in dated])[:6]

            pred_l = prediction_text.lower()
            present = [g for g in gold_items if g and g in pred_l]
            step_completeness = (len(present) / len(gold_items)) if gold_items else None

            ordering_accuracy = None
            if len(gold_items) >= 2:
                # Pairwise order based on first mention indices.
                idx_map = {g: pred_l.find(g) for g in gold_items if pred_l.find(g) != -1}
                total_pairs = 0
                correct_pairs = 0
                for i in range(len(gold_items)):
                    for j in range(i + 1, len(gold_items)):
                        a = gold_items[i]
                        b = gold_items[j]
                        if a not in idx_map or b not in idx_map:
                            continue
                        total_pairs += 1
                        if idx_map[a] < idx_map[b]:
                            correct_pairs += 1
                ordering_accuracy = (correct_pairs / total_pairs) if total_pairs else 0.0

            metrics.update(
                {
                    "ordering_accuracy": ordering_accuracy,
                    "step_completeness": step_completeness,
                    "causal_coherence": float(any(m in pred_l for m in _CAUSAL_MARKERS)),
                    "narrative_consistency": None,
                    "sequence_marker_presence": float(any(m in pred_l for m in _SEQUENCE_MARKERS)),
                }
            )
            return metrics

        # Causality
        if bucket == "causal":
            pred_l = prediction_text.lower()
            metrics.update(
                {
                    "causal_link_accuracy": None,
                    "temporal_constraint_correctness": _best_date_match_score(pred_dates, gold_dates, pred_years),
                    "alternative_cause_awareness": float("alternative" in pred_l or "other" in pred_l or "also" in pred_l),
                    "confidence_calibration": None,
                    "causal_marker_presence": float(any(m in pred_l for m in _CAUSAL_MARKERS)),
                }
            )
            return metrics

        # Overlap (optional bucket)
        if bucket == "overlap":
            pred_l = prediction_text.lower()
            metrics.update(
                {
                    "overlap_representation": float(any(m in pred_l for m in _OVERLAP_MARKERS)),
                    "boundary_accuracy": _best_date_match_score(pred_dates, gold_dates, pred_years),
                }
            )
            return metrics

        # Default: return shared metrics only.
        return metrics


def aggregate_by_bucket(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate numeric metrics by `bucket` (mean over items)."""

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(str(r.get("bucket") or "other"), []).append(r)

    out: Dict[str, Dict[str, float]] = {}
    for bucket, items in buckets.items():
        numeric_keys = set()
        for it in items:
            for k, v in it.items():
                if isinstance(v, (int, float)) and k not in {"num_temporal_mentions"}:
                    numeric_keys.add(k)

        summary: Dict[str, float] = {"count": float(len(items))}
        for k in sorted(numeric_keys):
            vals = [float(it[k]) for it in items if isinstance(it.get(k), (int, float))]
            if vals:
                summary[k] = sum(vals) / len(vals)
        out[bucket] = summary

    return out
