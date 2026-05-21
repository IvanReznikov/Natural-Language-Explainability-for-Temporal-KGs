#!/usr/bin/env python3
"""Classify sample texts with the default M2-E4 taxonomy model (LinearSVC word+char)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.m2_e4_taxonomy.predict_taxonomy import extract_text, load_model


def main():
    root = Path(__file__).resolve().parents[2]
    model_path = root / "output" / "m2_e4_taxonomy" / "e4a_taxonomy" / "taxonomy_model.joblib"
    if not model_path.exists():
        print(f"SKIPPED: model artifact not found at {model_path}")
        print("Run experiments/m2_e4_taxonomy/train_taxonomy_default.py first.")
        return
    model = load_model(model_path)

    samples = [
        {"text": "Revenue grew 12% year over year with margins holding steady."},
        {"text": "No data reported for this quarter."},
        {"text": "Plot the daily visitors over the last two weeks."},
    ]

    texts = [extract_text(item) for item in samples]
    preds = model.predict(texts)

    for item, label in zip(samples, preds):
        print({"text": item["text"], "predicted_label": label})


if __name__ == "__main__":
    main()
