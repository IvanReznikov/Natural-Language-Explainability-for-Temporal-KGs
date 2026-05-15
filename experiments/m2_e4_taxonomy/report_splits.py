import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
SPLIT_DIR = ROOT / "data" / "splits"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def report_taxonomy():
    train = load_jsonl(SPLIT_DIR / "result_taxonomy_train.jsonl")
    val = load_jsonl(SPLIT_DIR / "result_taxonomy_val.jsonl")
    test = load_jsonl(SPLIT_DIR / "result_taxonomy_test.jsonl")
    counts = Counter(row.get("label", "") for row in train + val + test)
    return {
        "name": "result_taxonomy",
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "labels": dict(sorted(counts.items())),
    }


def report_summaries():
    train = load_jsonl(SPLIT_DIR / "result_summaries_train.jsonl")
    val = load_jsonl(SPLIT_DIR / "result_summaries_val.jsonl")
    test = load_jsonl(SPLIT_DIR / "result_summaries_test.jsonl")
    return {
        "name": "result_summaries",
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
    }


def report_consistency():
    train = load_jsonl(SPLIT_DIR / "narrative_consistency_train.jsonl")
    val = load_jsonl(SPLIT_DIR / "narrative_consistency_val.jsonl")
    test = load_jsonl(SPLIT_DIR / "narrative_consistency_test.jsonl")
    counts = Counter(row.get("label", "") for row in train + val + test)
    return {
        "name": "narrative_consistency",
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "labels": dict(sorted(counts.items())),
    }


def main():
    reports = [report_taxonomy(), report_summaries(), report_consistency()]
    for rep in reports:
        print(f"\n{rep['name']} splits: {rep['splits']}")
        labels = rep.get("labels")
        if labels:
            print("labels:")
            for lbl, cnt in labels.items():
                print(f"  {lbl}: {cnt}")


if __name__ == "__main__":
    main()
