import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_labels(path: Path) -> List[str]:
    return json.loads(path.read_text())


def predict(text: str, model_dir: Path, threshold: float = 0.5) -> List[Dict]:
    labels = load_labels(model_dir / "labels.json")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze(0)
    preds = []
    for idx, score in enumerate(probs.tolist()):
        if score >= threshold:
            preds.append({"label": labels[idx], "score": score})
    # sort by score desc
    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", type=str, default=None, help="Single text to classify")
    parser.add_argument(
        "--file", type=Path, default=None, help="Optional jsonl with {'text': ...} rows"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.text and not args.file:
        raise SystemExit("Provide --text or --file")

    if args.text:
        preds = predict(args.text, args.model_dir, threshold=args.threshold)
        print(json.dumps(preds, indent=2))
        return

    rows = []
    with args.file.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    for row in rows:
        preds = predict(row["text"], args.model_dir, threshold=args.threshold)
        out = {"id": row.get("id"), "text": row.get("text"), "predicted_intents": preds}
        print(json.dumps(out))


if __name__ == "__main__":
    main()
