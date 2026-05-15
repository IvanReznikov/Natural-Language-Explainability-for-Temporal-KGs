"""Tests for dataset loading and corpus-level evaluation helpers."""

import json
from datetime import datetime

from temporal_nlg.temporal_expr import (
    TemporalDatasetExample,
    TemporalNormalizer,
    TemporalTagger,
    evaluate_dataset,
    load_jsonl_temporal_dataset,
)


def test_load_jsonl_dataset(tmp_path):
    dataset_path = tmp_path / "data.jsonl"
    records = [
        {
            "text": "Event on 2025-12-20",
            "gold_spans": [{"start": 9, "end": 19, "type": "DATE"}],
            "gold_normalized": ["2025-12-20"],
        }
    ]
    with dataset_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    loaded = load_jsonl_temporal_dataset(dataset_path)
    assert len(loaded) == 1
    assert isinstance(loaded[0], TemporalDatasetExample)
    assert loaded[0].text == records[0]["text"]


def test_evaluate_dataset_end_to_end(tmp_path):
    # Two examples: absolute date and relative expression with reference time.
    dataset_path = tmp_path / "data.jsonl"
    records = [
        {
            "text": "Event on 2025-12-20",
            "gold_spans": [{"start": 9, "end": 19, "type": "DATE"}],
            "gold_normalized": ["2025-12-20"],
        },
        {
            "text": "Meet tomorrow",
            "gold_spans": [{"start": 5, "end": 13, "type": "DATE"}],
            "gold_normalized": ["2025-12-11"],
            "reference_time": "2025-12-10T00:00:00",
        },
        {
            "text": "Treatment lasts 5 days",
            "gold_spans": [{"start": 16, "end": 22, "type": "DURATION"}],
            "gold_normalized": ["P5D"],
        },
    ]
    with dataset_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    dataset = load_jsonl_temporal_dataset(dataset_path)
    tagger = TemporalTagger()
    normalizer = TemporalNormalizer(default_reference=datetime(2025, 12, 10))

    metrics = evaluate_dataset(dataset, tagger=tagger, normalizer=normalizer)
    assert metrics["tagging_f1"] == 1.0
    assert metrics["normalization_accuracy"] == 1.0
