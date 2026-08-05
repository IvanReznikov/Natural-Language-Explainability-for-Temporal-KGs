import argparse
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "output" / "m2_e4_taxonomy" / "e4a_taxonomy" / "taxonomy_model.joblib"
DEFAULT_OUT = Path("-")


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def read_items(args):
    if args.text:
        yield {"text": args.text}
    elif args.input:
        p = Path(args.input)
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if args.jsonl:
                    yield json.loads(line)
                else:
                    yield {"text": line}
    else:
        for line in sys.stdin:
            line = line.strip()
            if line:
                yield {"text": line}


def extract_text(item):
    # Accept either raw text or records with a text-like field
    if "text" in item:
        return str(item["text"])
    # Fallback: join all values
    return " ".join(str(v) for v in item.values())


def main():
    ap = argparse.ArgumentParser(description="Predict taxonomy label with default model")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to joblib model")
    ap.add_argument(
        "--input", type=str, help="Input file (text per line or JSONL if --jsonl)", default=None
    )
    ap.add_argument("--text", type=str, help="Single text to classify", default=None)
    ap.add_argument("--jsonl", action="store_true", help="Treat input file as JSONL")
    ap.add_argument(
        "--output", type=Path, default=DEFAULT_OUT, help="Output JSONL path or '-' for stdout"
    )
    args = ap.parse_args()

    model = load_model(args.model)

    outputs = []
    for item in read_items(args):
        text = extract_text(item)
        label = model.predict([text])[0]
        outputs.append({"text": text, "label": label})

    if args.output == DEFAULT_OUT:
        for obj in outputs:
            sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    else:
        with args.output.open("w", encoding="utf-8") as f:
            for obj in outputs:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
