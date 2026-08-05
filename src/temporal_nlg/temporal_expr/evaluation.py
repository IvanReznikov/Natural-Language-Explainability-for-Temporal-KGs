"""Lightweight evaluation utilities for M2-E1."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Sequence

from .datasets import TemporalDatasetExample
from .schemas import (
    DocumentContext,
    NormalizedTemporal,
    Span,
    TemporalExpression,
    TemporalExpressionType,
)
from .tagger import TemporalTagger
from .normalizer import TemporalNormalizer


def span_f1(
    predicted: Sequence[TemporalExpression], gold: Sequence[TemporalExpression]
) -> Dict[str, float]:
    """Compute span-level precision/recall/F1 on exact-match spans."""

    pred_spans = {(expr.span.start, expr.span.end) for expr in predicted}
    gold_spans = {(expr.span.start, expr.span.end) for expr in gold}

    true_pos = len(pred_spans & gold_spans)
    precision = true_pos / len(pred_spans) if pred_spans else 0.0
    recall = true_pos / len(gold_spans) if gold_spans else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def normalization_accuracy(
    predicted: Iterable[NormalizedTemporal], gold: Iterable[str]
) -> Dict[str, float]:
    """Exact-match accuracy for normalized values."""

    predicted_list: List[NormalizedTemporal] = list(predicted)
    gold_list: List[str] = list(gold)
    matched = 0
    total = min(len(predicted_list), len(gold_list))
    for idx in range(total):
        if predicted_list[idx].normalized == gold_list[idx]:
            matched += 1
    accuracy = matched / total if total else 0.0
    return {"accuracy": accuracy, "total": total}


def evaluate_dataset(
    examples: Sequence[TemporalDatasetExample],
    tagger: TemporalTagger,
    normalizer: TemporalNormalizer,
) -> Dict[str, float]:
    """Run tagging + normalization over a dataset and report simple metrics."""

    f1_sum = 0.0
    count = 0
    norm_matched = 0
    norm_total = 0

    for ex in examples:
        predicted = tagger.tag(ex.text)
        gold_exprs = _gold_spans_to_exprs(ex.text, ex.gold_spans)
        span_metrics = span_f1(predicted, gold_exprs)
        f1_sum += span_metrics["f1"]
        count += 1

        ctx = None
        if ex.reference_time:
            ctx = DocumentContext(reference_time=datetime.fromisoformat(ex.reference_time))
        normalized = [normalizer.normalize(expr, context=ctx) for expr in predicted]
        gold_norm = ex.gold_normalized
        total = min(len(normalized), len(gold_norm))
        norm_total += total
        for idx in range(total):
            if normalized[idx].normalized == gold_norm[idx]:
                norm_matched += 1

    tagging_f1 = f1_sum / count if count else 0.0
    norm_accuracy = norm_matched / norm_total if norm_total else 0.0

    return {"tagging_f1": tagging_f1, "normalization_accuracy": norm_accuracy}


def _gold_spans_to_exprs(text: str, spans: Sequence[dict]) -> List[TemporalExpression]:
    results: List[TemporalExpression] = []
    for span_dict in spans:
        start = span_dict["start"]
        end = span_dict["end"]
        expr_type = TemporalExpressionType(span_dict.get("type", "DATE"))
        segment = text[start:end]
        results.append(
            TemporalExpression(
                text=segment,
                span=Span(start=start, end=end, text=segment),
                expr_type=expr_type,
            )
        )
    return results
