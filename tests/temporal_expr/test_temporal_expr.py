"""Smoke tests for temporal expression tagging and normalization (M2-E1)."""

from datetime import datetime

from temporal_nlg.temporal_expr import (
    ContextAwareResolver,
    DocumentContext,
    TemporalNormalizer,
    TemporalTagger,
)
from temporal_nlg.temporal_expr.evaluation import normalization_accuracy, span_f1


def test_tagger_and_normalizer_end_to_end():
    text = (
        "We met on January 15, 2024 at 14:30 and again last Tuesday. "
        "Every Monday we sync around noon for 3 days. Next month we review on 2025-12-20 and we run twice a week."
    )
    reference_time = datetime(2025, 12, 10)

    tagger = TemporalTagger()
    entities = tagger.tag(text)
    assert len(entities) == 9

    context = DocumentContext(reference_time=reference_time)
    normalizer = TemporalNormalizer(default_reference=reference_time)
    resolver = ContextAwareResolver(normalizer)
    normalized = resolver.resolve_all(entities, context=context)

    values = [n.normalized for n in normalized]
    expected = [
        "2024-01-15",
        "14:30",
        "2025-12-09",
        "every-monday",
        "12:00",
        "P3D",
        "2026-01-10",
        "2025-12-20",
        "twice-a-week",
    ]
    assert values == expected

    metrics = normalization_accuracy(normalized, expected)
    assert metrics["accuracy"] == 1.0


def test_span_f1_exact_match():
    text = "By tomorrow we finalize the draft."
    reference_time = datetime(2025, 12, 10)
    tagger = TemporalTagger()
    pred = tagger.tag(text)
    gold = pred.copy()

    metrics = span_f1(pred, gold)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_composite_duration_parsing():
    text = "Procedure took 1 hour 30 minutes and cooldown lasted 2h15m."
    tagger = TemporalTagger()
    entities = tagger.tag(text)
    assert len(entities) == 2

    normalizer = TemporalNormalizer()
    normalized = [normalizer.normalize(e).normalized for e in entities]
    assert normalized == ["PT1H30M", "PT2H15M"]


def test_relative_with_time_and_range():
    reference_time = datetime(2025, 12, 10, tzinfo=datetime.now().astimezone().tzinfo)
    tagger = TemporalTagger()

    text = "We meet tomorrow at 3pm. The event runs January 5-7, 2025."
    entities = tagger.tag(text)
    assert any("tomorrow at 3pm".lower() == e.text.lower() for e in entities)
    assert any("January 5-7, 2025".lower() == e.text.lower() for e in entities)

    normalizer = TemporalNormalizer(default_reference=reference_time)
    values = [normalizer.normalize(e, context=DocumentContext(reference_time=reference_time)).normalized for e in entities]
    assert "2025-12-11T15:00" in values
    assert "2025-01-05/2025-01-07" in values
