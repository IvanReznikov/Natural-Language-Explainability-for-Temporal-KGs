import numpy as np

from experiments.m2_e2_intent import run_intent_classifier as intent


def test_threshold_predictions_enforces_at_least_one_label():
    proba = np.array([[0.1, 0.2, 0.05], [0.01, 0.02, 0.5]])
    preds = intent.threshold_predictions(proba, threshold=0.9)
    assert preds.shape == proba.shape
    assert all(row.sum() == 1 for row in preds)


def test_normalize_intent_and_load_dataset_filters_unknowns(tmp_path):
    data = [
        {"query": "When did it happen?", "intents": ["Point in Time"]},
        {"query": "Compare years", "intents": ["Comparative", "Unknown"]},
        {"query": "", "intents": ["prediction"]},
    ]
    data_path = tmp_path / "intent.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(intent.json.dumps(row) + "\n")

    examples = intent.load_dataset(data_path, allowed_labels=intent.INTENT_TAXONOMY)
    labels = [lbl for ex in examples for lbl in ex.intents]

    assert len(examples) == 2  # skips empty query
    assert "point_in_time" in labels
    assert "comparative" in labels
    assert "unknown" not in labels
