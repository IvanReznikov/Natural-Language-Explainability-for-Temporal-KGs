import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 13
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(rows, key_fn, train_ratio=0.8, val_ratio=0.1):
    rng = random.Random(SEED)
    buckets = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)

    train, val, test = [], [], []
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
        n = len(bucket_rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        # Ensure no empty splits when possible
        if n_test == 0 and n_val > 0:
            n_val -= 1
            n_test += 1
        train.extend(bucket_rows[:n_train])
        val.extend(bucket_rows[n_train : n_train + n_val])
        test.extend(bucket_rows[n_train + n_val :])
    return train, val, test


def simple_split(rows, train_ratio=0.8, val_ratio=0.1):
    rng = random.Random(SEED)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return rows[:n_train], rows[n_train : n_train + n_val], rows[n_train + n_val :]


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "result_taxonomy",
            "source": DATA_DIR / "taxonomy.jsonl",
            "split_fn": lambda rows: stratified_split(rows, key_fn=lambda r: r.get("label", "")),
        },
        {
            "name": "result_summaries",
            "source": DATA_DIR / "summaries.jsonl",
            "split_fn": simple_split,
        },
        {
            "name": "narrative_consistency",
            "source": DATA_DIR / "narrative_consistency.jsonl",
            "split_fn": lambda rows: stratified_split(rows, key_fn=lambda r: r.get("label", "")),
        },
    ]

    for cfg in configs:
        rows = load_jsonl(cfg["source"])
        train, val, test = cfg["split_fn"](rows)
        write_jsonl(SPLIT_DIR / f"{cfg['name']}_train.jsonl", train)
        write_jsonl(SPLIT_DIR / f"{cfg['name']}_val.jsonl", val)
        write_jsonl(SPLIT_DIR / f"{cfg['name']}_test.jsonl", test)
        print(f"{cfg['name']}: train={len(train)}, val={len(val)}, test={len(test)}, total={len(rows)}")


if __name__ == "__main__":
    main()
