#!/usr/bin/env python3
"""Train and evaluate the M2-E2 intent classifier on a small slice, then print sample predictions."""

import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.m2_e2_intent import run_intent_classifier as intent_mod


def describe_examples(examples: List[intent_mod.IntentExample], preds, labels):
    for ex, pred_row in zip(examples, preds):
        chosen = [lbl for idx, lbl in enumerate(labels) if pred_row[idx] == 1]
        print({"query": ex.query[:90], "predicted": chosen, "gold": ex.intents})


def main():
    root = Path(__file__).resolve().parents[2]
    dataset = root / "experiments" / "m2_e2_intent" / "data" / "annotated_queries.jsonl"

    examples = intent_mod.load_dataset(dataset, intent_mod.INTENT_TAXONOMY)
    train, val, test = intent_mod.split_dataset(examples, test_size=0.15, val_size=0.1, seed=42)

    artifacts = intent_mod.train_classifier(
        train_examples=train,
        val_examples=val,
        labels=intent_mod.INTENT_TAXONOMY,
        threshold=0.45,
        per_label_thresholds=False,
        threshold_grid=[0.45],
        calibration="isotonic",
        calibration_cv=3,
        class_weight_mode="none",
        pos_class_weight=1.0,
        word_ngram_range=(1, 2),
        max_features_word=4000,
        include_char=True,
        char_ngram_range=(3, 5),
        max_features_char=8000,
        seed=42,
    )

    y_true = artifacts["binarizer"].transform([ex.intents for ex in test])
    _, y_pred = intent_mod.predict(artifacts, test, artifacts["label_thresholds"])
    metrics = intent_mod.evaluate(y_true, y_pred, intent_mod.INTENT_TAXONOMY)

    print(
        {
            "macro_f1": round(metrics["macro_f1"], 3),
            "micro_f1": round(metrics["micro_f1"], 3),
            "subset_accuracy": round(metrics["subset_accuracy"], 3),
        }
    )
    print("Sample predictions (first 5):")
    describe_examples(test[:5], y_pred[:5], intent_mod.INTENT_TAXONOMY)


if __name__ == "__main__":
    main()
