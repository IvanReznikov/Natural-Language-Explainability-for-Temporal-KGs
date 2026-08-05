import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def split(
    rows: List[Dict], train_ratio: float, val_ratio: float
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]
    return train, val, test


def collect_labels(rows: List[Dict]) -> List[str]:
    labels = set()
    for row in rows:
        for lbl in row.get("intent_labels", []):
            labels.add(lbl)
    return sorted(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to full jsonl dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/m2_e3_parse/data/splits"),
        help="Where to write splits and labels.json",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = load_jsonl(args.data)
    train, val, test = split(rows, args.train_ratio, args.val_ratio)

    labels = collect_labels(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output_dir / "train.jsonl", train)
    save_jsonl(args.output_dir / "val.jsonl", val)
    save_jsonl(args.output_dir / "test.jsonl", test)

    with (args.output_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    stats = {
        "total": len(rows),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "labels": labels,
    }
    with (args.output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
