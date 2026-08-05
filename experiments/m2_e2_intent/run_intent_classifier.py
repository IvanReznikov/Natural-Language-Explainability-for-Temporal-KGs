#!/usr/bin/env python3
"""
M2-E2: Query Intent Classification and Multi-Intent Detection
- Implements the taxonomy and metrics from ideas/temporal-experiments.md (M2-E2a/b/c)
- Lightweight baseline: TF-IDF + One-vs-Rest Logistic Regression
- Handles single and multi-intent labels with configurable probability threshold
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union
from uuid import uuid4

import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

INTENT_TAXONOMY = [
    "point_in_time",
    "interval",
    "sequence",
    "causal",
    "comparative",
    "aggregation",
    "prediction",
    "explanation",
]


@dataclass
class IntentExample:
    query: str
    intents: List[str]


def normalize_intent(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def load_dataset(path: Path, allowed_labels: Sequence[str]) -> List[IntentExample]:
    examples: List[IntentExample] = []
    allowed = set(allowed_labels)

    with path.open() as f:
        for line in f:
            record = json.loads(line)
            query = record.get("query", "").strip()
            intents = [normalize_intent(lbl) for lbl in record.get("intents", [])]
            intents = [lbl for lbl in intents if lbl in allowed]
            if not query or not intents:
                continue
            examples.append(IntentExample(query=query, intents=intents))

    if not examples:
        raise ValueError(f"No usable examples found in {path}")
    return examples


def split_dataset(
    examples: List[IntentExample],
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[List[IntentExample], List[IntentExample], List[IntentExample]]:
    train, test = train_test_split(examples, test_size=test_size, random_state=seed)
    train, val = train_test_split(train, test_size=val_size, random_state=seed)
    return train, val, test


def vectorize(
    train_texts: List[str],
    other_texts: List[str],
    word_ngram_range: Tuple[int, int],
    max_features_word: int,
    include_char: bool,
    char_ngram_range: Tuple[int, int],
    max_features_char: int,
) -> Tuple[Dict[str, TfidfVectorizer], np.ndarray, np.ndarray]:
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=word_ngram_range,
        min_df=1,
        max_features=max_features_word,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train = word_vectorizer.fit_transform(train_texts)
    X_other = word_vectorizer.transform(other_texts)

    vectorizers: Dict[str, TfidfVectorizer] = {"word": word_vectorizer}

    if include_char:
        char_vectorizer = TfidfVectorizer(
            analyzer="char",
            lowercase=True,
            ngram_range=char_ngram_range,
            min_df=2,
            max_features=max_features_char,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        Xc_train = char_vectorizer.fit_transform(train_texts)
        Xc_other = char_vectorizer.transform(other_texts)
        vectorizers["char"] = char_vectorizer
        X_train = hstack([X_train, Xc_train])
        X_other = hstack([X_other, Xc_other])

    return vectorizers, X_train, X_other


def train_classifier(
    train_examples: List[IntentExample],
    val_examples: List[IntentExample],
    labels: Sequence[str],
    threshold: float,
    per_label_thresholds: bool,
    threshold_grid: Sequence[float],
    calibration: str,
    calibration_cv: int,
    class_weight_mode: str,
    pos_class_weight: float,
    word_ngram_range: Tuple[int, int],
    max_features_word: int,
    include_char: bool,
    char_ngram_range: Tuple[int, int],
    max_features_char: int,
    seed: int,
) -> Dict:
    mlb = MultiLabelBinarizer(classes=list(labels))
    y_train = mlb.fit_transform([ex.intents for ex in train_examples])
    y_val = mlb.transform([ex.intents for ex in val_examples])

    vectorizers, X_train, X_val = vectorize(
        [ex.query for ex in train_examples],
        [ex.query for ex in val_examples],
        word_ngram_range,
        max_features_word,
        include_char,
        char_ngram_range,
        max_features_char,
    )

    if class_weight_mode == "balanced":
        class_weight: Union[None, str, Dict[int, float]] = "balanced"
    elif class_weight_mode == "pos-mult":
        class_weight = {0: 1.0, 1: pos_class_weight}
    else:
        class_weight = None

    base_lr = LogisticRegression(
        max_iter=1000, solver="lbfgs", class_weight=class_weight, random_state=seed
    )
    if calibration != "none":
        method = "sigmoid" if calibration == "platt" else "isotonic"
        estimator = CalibratedClassifierCV(estimator=base_lr, method=method, cv=calibration_cv)
    else:
        estimator = base_lr

    clf = OneVsRestClassifier(estimator)
    clf.fit(X_train, y_train)

    val_proba = clf.predict_proba(X_val)
    if per_label_thresholds:
        label_thresholds = best_thresholds_per_label(val_proba, y_val, threshold_grid)
    else:
        label_thresholds = np.full(len(labels), threshold, dtype=float)

    y_val_pred = threshold_predictions(val_proba, label_thresholds)

    return {
        "vectorizers": vectorizers,
        "binarizer": mlb,
        "classifier": clf,
        "val_pred": y_val_pred,
        "y_val_true": y_val,
        "label_thresholds": label_thresholds,
        "val_proba": val_proba,
    }


def threshold_predictions(proba: np.ndarray, threshold: Union[float, np.ndarray]) -> np.ndarray:
    if isinstance(threshold, float):
        th = np.full(proba.shape[1], threshold, dtype=float)
    else:
        th = np.asarray(threshold, dtype=float)
        if th.shape[0] != proba.shape[1]:
            raise ValueError("Per-label threshold length must match number of labels")

    preds = (proba >= th).astype(int)
    for i, row in enumerate(preds):
        if row.sum() == 0:
            # Avoid empty predictions by choosing the argmax intent
            preds[i, int(np.argmax(proba[i]))] = 1
    return preds


def best_thresholds_per_label(
    proba: np.ndarray, y_true: np.ndarray, grid: Sequence[float]
) -> np.ndarray:
    best = np.full(proba.shape[1], 0.5, dtype=float)
    for j in range(proba.shape[1]):
        best_f1 = -1.0
        for t in grid:
            pred_j = (proba[:, j] >= t).astype(int)
            f1 = f1_score(y_true[:, j], pred_j, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best[j] = t
    return best


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> Dict:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    subset_acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(labels),
        zero_division=0,
        output_dict=True,
    )

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "subset_accuracy": subset_acc,
        "per_label": {label: report[label] for label in labels},
    }


def predict(
    artifacts: Dict,
    examples: List[IntentExample],
    threshold: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizers: Dict[str, TfidfVectorizer] = artifacts["vectorizers"]
    clf: OneVsRestClassifier = artifacts["classifier"]
    mlb: MultiLabelBinarizer = artifacts["binarizer"]

    word_vectorizer = vectorizers["word"]
    X = word_vectorizer.transform([ex.query for ex in examples])

    char_vectorizer = vectorizers.get("char")
    if char_vectorizer is not None:
        X_char = char_vectorizer.transform([ex.query for ex in examples])
        X = hstack([X, X_char])

    proba = clf.predict_proba(X)
    preds = threshold_predictions(proba, threshold)
    return proba, preds


def predict_proba_only(artifacts: Dict, examples: List[IntentExample]) -> np.ndarray:
    """Return probabilities for a batch using stored vectorizers and classifier."""
    vectorizers: Dict[str, TfidfVectorizer] = artifacts["vectorizers"]
    clf: OneVsRestClassifier = artifacts["classifier"]

    X = vectorizers["word"].transform([ex.query for ex in examples])
    char_vectorizer = vectorizers.get("char")
    if char_vectorizer is not None:
        X_char = char_vectorizer.transform([ex.query for ex in examples])
        X = hstack([X, X_char])

    return clf.predict_proba(X)


def save_outputs(
    output_dir: Path,
    metrics: Dict,
    examples: List[IntentExample],
    proba: np.ndarray,
    preds: np.ndarray,
    labels: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    preds_path = run_dir / "predictions.jsonl"
    with preds_path.open("w") as f:
        for ex, prob_row, pred_row in zip(examples, proba, preds):
            intents = [label for label, active in zip(labels, pred_row) if active]
            record = {
                "query": ex.query,
                "gold": ex.intents,
                "predicted": intents,
                "probabilities": {label: float(score) for label, score in zip(labels, prob_row)},
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {preds_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="M2-E2 intent classification baseline")
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to JSONL with queries + intents"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write results")
    parser.add_argument(
        "--threshold", type=float, default=0.35, help="Probability threshold for multi-intent"
    )
    parser.add_argument(
        "--per-label-thresholds",
        action="store_true",
        help="Optimize per-label thresholds on validation set",
    )
    parser.add_argument(
        "--threshold-grid-start",
        type=float,
        default=0.1,
        help="Grid start for per-label threshold sweep",
    )
    parser.add_argument(
        "--threshold-grid-stop",
        type=float,
        default=0.9,
        help="Grid stop (inclusive) for threshold sweep",
    )
    parser.add_argument(
        "--threshold-grid-step", type=float, default=0.05, help="Grid step for threshold sweep"
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "platt", "isotonic"],
        default="none",
        help="Probability calibration method (Platt=sigmoid)",
    )
    parser.add_argument("--calibration-cv", type=int, default=3, help="CV folds for calibration")
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none", "pos-mult"],
        default="balanced",
        help="Class weight strategy: sklearn balanced, none, or positive-class multiplier",
    )
    parser.add_argument(
        "--pos-class-weight",
        type=float,
        default=1.0,
        help="Positive-class weight when class-weight=pos-mult (1.0 disables)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated random seeds for ensembling (avg probabilities)",
    )
    parser.add_argument(
        "--word-ngram-max", type=int, default=2, help="Max word n-gram size (min is fixed at 1)"
    )
    parser.add_argument(
        "--word-max-features", type=int, default=8000, help="Max word TF-IDF features"
    )
    parser.add_argument(
        "--use-char-ngrams", action="store_true", help="Include character TF-IDF features"
    )
    parser.add_argument("--char-ngram-min", type=int, default=3, help="Min char n-gram size")
    parser.add_argument("--char-ngram-max", type=int, default=5, help="Max char n-gram size")
    parser.add_argument(
        "--char-max-features", type=int, default=12000, help="Max char TF-IDF features"
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument(
        "--val-size", type=float, default=0.1, help="Validation split fraction from train"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    examples = load_dataset(args.dataset, INTENT_TAXONOMY)
    train, val, test = split_dataset(
        examples, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )

    grid_vals = np.arange(
        args.threshold_grid_start, args.threshold_grid_stop + 1e-9, args.threshold_grid_step
    )
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if len(seeds) == 1:
        artifacts = train_classifier(
            train,
            val,
            INTENT_TAXONOMY,
            args.threshold,
            args.per_label_thresholds,
            grid_vals,
            args.calibration,
            args.calibration_cv,
            args.class_weight,
            args.pos_class_weight,
            (1, args.word_ngram_max),
            args.word_max_features,
            args.use_char_ngrams,
            (args.char_ngram_min, args.char_ngram_max),
            args.char_max_features,
            seeds[0],
        )

        proba, preds = predict(artifacts, test, artifacts["label_thresholds"])
        metrics = evaluate(artifacts["y_val_true"], artifacts["val_pred"], INTENT_TAXONOMY)
        test_metrics = evaluate(
            artifacts["binarizer"].transform([ex.intents for ex in test]),
            preds,
            INTENT_TAXONOMY,
        )

        combined_metrics = {
            "validation": metrics,
            "test": test_metrics,
            "label_thresholds": artifacts["label_thresholds"].tolist(),
            "seeds": seeds,
        }
        proba_to_save, preds_to_save = proba, preds
        artifacts_to_use = artifacts
    else:
        models = []
        val_probas = []
        for seed in seeds:
            art = train_classifier(
                train,
                val,
                INTENT_TAXONOMY,
                args.threshold,
                args.per_label_thresholds,
                grid_vals,
                args.calibration,
                args.calibration_cv,
                args.class_weight,
                args.pos_class_weight,
                (1, args.word_ngram_max),
                args.word_max_features,
                args.use_char_ngrams,
                (args.char_ngram_min, args.char_ngram_max),
                args.char_max_features,
                seed,
            )
            models.append(art)
            val_probas.append(art["val_proba"])

        val_proba_mean = np.mean(val_probas, axis=0)
        y_val_true = models[0]["y_val_true"]

        if args.per_label_thresholds:
            label_thresholds = best_thresholds_per_label(val_proba_mean, y_val_true, grid_vals)
        else:
            label_thresholds = np.full(len(INTENT_TAXONOMY), args.threshold, dtype=float)

        val_pred = threshold_predictions(val_proba_mean, label_thresholds)
        metrics = evaluate(y_val_true, val_pred, INTENT_TAXONOMY)

        test_probas = []
        for art in models:
            test_probas.append(predict_proba_only(art, test))
        test_proba_mean = np.mean(test_probas, axis=0)
        test_preds = threshold_predictions(test_proba_mean, label_thresholds)
        test_metrics = evaluate(
            models[0]["binarizer"].transform([ex.intents for ex in test]),
            test_preds,
            INTENT_TAXONOMY,
        )

        combined_metrics = {
            "validation": metrics,
            "test": test_metrics,
            "label_thresholds": label_thresholds.tolist(),
            "seeds": seeds,
        }
        proba_to_save, preds_to_save = test_proba_mean, test_preds
        artifacts_to_use = models[0]

    save_outputs(
        args.output_dir, combined_metrics, test, proba_to_save, preds_to_save, INTENT_TAXONOMY
    )


if __name__ == "__main__":
    main()
