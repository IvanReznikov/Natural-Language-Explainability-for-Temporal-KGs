import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


def safe_parse(json_str: str) -> Dict:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def predict(text: str, model_dir: Path, max_new_tokens: int = 256) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
    parsed = safe_parse(decoded)
    spans = parsed.get("spans", [])
    frame = parsed.get("frame", {})
    intent_labels = parsed.get("intent_labels", [])
    return {"spans": spans, "frame": frame, "intent_labels": intent_labels, "raw": decoded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", type=str, default=None, help="Single text to parse")
    parser.add_argument(
        "--file", type=Path, default=None, help="Optional jsonl with {'text': ...} rows"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if not args.text and not args.file:
        raise SystemExit("Provide --text or --file")

    if args.text:
        result = predict(args.text, args.model_dir, max_new_tokens=args.max_new_tokens)
        print(json.dumps(result, indent=2))
        return

    with args.file.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            parsed = predict(row["text"], args.model_dir, max_new_tokens=args.max_new_tokens)
            out = {"id": row.get("id"), "text": row.get("text"), **parsed}
            print(json.dumps(out))


if __name__ == "__main__":
    main()
