#!/usr/bin/env python3
"""
Run all T1–T15 (TF-IDF) and document M16/M17 intent classifier configurations,
saving metrics.json artifacts to the correct experiment result directories.

Usage:
    python scripts/run_intent_sweep.py

All results land under experiments/m2_e2_intent/results/<run_id>/metrics.json
with the exact run_id slugs referenced in docs/RESULTS_M2.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from experiments.m2_e2_intent.run_intent_classifier import (
    INTENT_TAXONOMY,
    load_dataset,
    split_dataset,
    train_classifier,
    evaluate,
    predict,
    best_thresholds_per_label,
    threshold_predictions,
    predict_proba_only,
)

DATA_PATH = ROOT / "experiments" / "m2_e2_intent" / "data" / "annotated_queries.jsonl"
OUTPUT_BASE = ROOT / "experiments" / "m2_e2_intent" / "results"

# (run_id, kwargs_for_train_classifier, threshold_mode, seeds, extra_meta)
CONFIGS = [
    # T1
    (
        "f569b1f7239c4066a29cc4471ef0167a",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=False,
            char_ngram_range=(3, 5),
            max_features_char=12000,
            calibration="none",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        0.25,
        [42],
        {"exp_id": "T1"},
    ),
    # T2
    (
        "58b69330d54a47fcb6f01df61ad6133e",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=False,
            char_ngram_range=(3, 5),
            max_features_char=12000,
            calibration="none",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        0.30,
        [42],
        {"exp_id": "T2"},
    ),
    # T3
    (
        "efe6efe4da3c4dfa8753b3249182c693",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=False,
            char_ngram_range=(3, 5),
            max_features_char=12000,
            calibration="none",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        0.35,
        [42],
        {"exp_id": "T3"},
    ),
    # T4
    (
        "33a1a6951e2744c185205c75627c9c8a",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=False,
            char_ngram_range=(3, 5),
            max_features_char=12000,
            calibration="none",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        0.40,
        [42],
        {"exp_id": "T4"},
    ),
    # T5
    (
        "073406af22914a4d82fe9d54d7fd95e2",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=False,
            char_ngram_range=(3, 5),
            max_features_char=12000,
            calibration="none",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        0.45,
        [42],
        {"exp_id": "T5"},
    ),
    # T6 — word + char (3-5, 12k)
    (
        "1817849eef854d78a71a766e70570c45",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=True,
            char_ngram_range=(3, 5),
            max_features_char=12000,
            calibration="none",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        0.45,
        [42],
        {"exp_id": "T6"},
    ),
    # T7 — word + char (4-6, 12k)
    (
        "b5b155311494432498e99edf56291f04",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=True,
            char_ngram_range=(4, 6),
            max_features_char=12000,
            calibration="none",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        0.45,
        [42],
        {"exp_id": "T7"},
    ),
    # T8 — word + char (3-5, 20k)
    (
        "f7dcd7b5ffbc4afa8bcdb660b3d36c08",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=True,
            char_ngram_range=(3, 5),
            max_features_char=20000,
            calibration="none",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        0.45,
        [42],
        {"exp_id": "T8"},
    ),
    # T9 — word + char (4-6, 20k)
    (
        "f1c4283e4f4f4a7e80ae433787ce7da0",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=True,
            char_ngram_range=(4, 6),
            max_features_char=20000,
            calibration="none",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        0.45,
        [42],
        {"exp_id": "T9"},
    ),
    # T10 — per-label thresholds, char (3-5, 20k)
    (
        "6d3eacd81db24b01928539889346172b",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=True,
            char_ngram_range=(3, 5),
            max_features_char=20000,
            calibration="none",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        "per_label",
        [42],
        {"exp_id": "T10"},
    ),
    # T11 — per-label thresholds, char (4-6, 20k)
    (
        "3b61fbfe8d064f03a042257e1c1dd999",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=True,
            char_ngram_range=(4, 6),
            max_features_char=20000,
            calibration="none",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        "per_label",
        [42],
        {"exp_id": "T11"},
    ),
    # T12 — Platt calibration
    (
        "501dd8a1a10e4fb58c682a59ef5b330e",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=True,
            char_ngram_range=(3, 5),
            max_features_char=20000,
            calibration="platt",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        0.45,
        [42],
        {"exp_id": "T12"},
    ),
    # T13 — Isotonic calibration (best TF-IDF run)
    (
        "4f66cd86f2c74ded9012904945cfacca",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=True,
            char_ngram_range=(3, 5),
            max_features_char=20000,
            calibration="isotonic",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        0.45,
        [42],
        {"exp_id": "T13"},
    ),
    # T14 — pos-mult upweighting
    (
        "a429b78732164d6c8ea010e48f547faa",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=True,
            char_ngram_range=(3, 5),
            max_features_char=20000,
            calibration="isotonic",
            calibration_cv=3,
            class_weight_mode="pos-mult",
            pos_class_weight=1.2,
        ),
        0.45,
        [42],
        {"exp_id": "T14"},
    ),
    # T15 — seed ensemble (42/43/44)
    (
        "ba78e0a01d6648cdad5ddd05e11572c4",
        dict(
            word_ngram_range=(1, 2),
            max_features_word=8000,
            include_char=True,
            char_ngram_range=(3, 5),
            max_features_char=20000,
            calibration="isotonic",
            calibration_cv=3,
            class_weight_mode="balanced",
            pos_class_weight=1.0,
        ),
        0.45,
        [42, 43, 44],
        {"exp_id": "T15"},
    ),
]

GRID_VALS = np.arange(0.1, 0.91, 0.05)


def run_single(run_id, cfg, threshold_mode, seeds, extra_meta, train, val, test):
    def _train_one(seed):
        return train_classifier(
            train,
            val,
            INTENT_TAXONOMY,
            0.5,
            False,
            GRID_VALS,
            cfg["calibration"],
            cfg["calibration_cv"],
            cfg["class_weight_mode"],
            cfg["pos_class_weight"],
            cfg["word_ngram_range"],
            cfg["max_features_word"],
            cfg["include_char"],
            cfg["char_ngram_range"],
            cfg["max_features_char"],
            seed,
        )

    if len(seeds) == 1:
        art = _train_one(seeds[0])
        # threshold
        if threshold_mode == "per_label":
            label_thresholds = art["label_thresholds"]  # already computed per-label
        else:
            label_thresholds = np.full(len(INTENT_TAXONOMY), float(threshold_mode))

        proba, preds = predict(art, test, label_thresholds)
        y_test_true = art["binarizer"].transform([ex.intents for ex in test])
        test_metrics = evaluate(y_test_true, preds, INTENT_TAXONOMY)
        val_metrics = evaluate(art["y_val_true"], art["val_pred"], INTENT_TAXONOMY)
        combined = {
            "validation": val_metrics,
            "test": test_metrics,
            "label_thresholds": label_thresholds.tolist(),
            "seeds": seeds,
            **extra_meta,
        }
    else:
        models = [_train_one(s) for s in seeds]
        val_probas = [m["val_proba"] for m in models]
        val_proba_mean = np.mean(val_probas, axis=0)
        y_val_true = models[0]["y_val_true"]

        if threshold_mode == "per_label":
            label_thresholds = best_thresholds_per_label(val_proba_mean, y_val_true, GRID_VALS)
        else:
            label_thresholds = np.full(len(INTENT_TAXONOMY), float(threshold_mode))

        val_pred = threshold_predictions(val_proba_mean, label_thresholds)
        val_metrics = evaluate(y_val_true, val_pred, INTENT_TAXONOMY)

        test_probas = [predict_proba_only(m, test) for m in models]
        test_proba_mean = np.mean(test_probas, axis=0)
        test_preds = threshold_predictions(test_proba_mean, label_thresholds)
        y_test_true = models[0]["binarizer"].transform([ex.intents for ex in test])
        test_metrics = evaluate(y_test_true, test_preds, INTENT_TAXONOMY)

        combined = {
            "validation": val_metrics,
            "test": test_metrics,
            "label_thresholds": label_thresholds.tolist(),
            "seeds": seeds,
            **extra_meta,
        }

    # Write metrics
    out_dir = OUTPUT_BASE / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(combined, f, indent=2)
    return combined


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    examples = load_dataset(DATA_PATH, INTENT_TAXONOMY)
    train, val, test = split_dataset(examples, test_size=0.2, val_size=0.1, seed=42)
    print(f"  train={len(train)}, val={len(val)}, test={len(test)}")

    for run_id, cfg, threshold_mode, seeds, extra in CONFIGS:
        exp_id = extra.get("exp_id", run_id[:8])
        print(f"\n[{exp_id}] run_id={run_id[:16]}... seeds={seeds} threshold={threshold_mode}")
        result = run_single(run_id, cfg, threshold_mode, seeds, extra, train, val, test)
        mf1 = result["test"]["macro_f1"]
        mif1 = result["test"]["micro_f1"]
        sacc = result["test"]["subset_accuracy"]
        print(f"  --> test macro_f1={mf1:.3f}  micro_f1={mif1:.3f}  subset_acc={sacc:.3f}")

    print(f"\nAll {len(CONFIGS)} runs saved under {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
