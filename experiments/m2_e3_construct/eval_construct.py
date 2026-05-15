import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True, help="Gold file with canonical_query")
    parser.add_argument("--pred", type=Path, required=True, help="Constructed outputs")
    args = parser.parse_args()

    gold_rows = {row["id"]: row for row in load_jsonl(args.gold)}
    pred_rows = {row["id"]: row for row in load_jsonl(args.pred)}

    total = 0
    matches = 0

    for qid, gold in gold_rows.items():
        if qid not in pred_rows:
            continue
        total += 1
        if gold.get("canonical_query") == pred_rows[qid].get("canonical_query"):
            matches += 1

    acc = matches / total if total else 0.0
    metrics = {"examples": total, "template_accuracy": acc}
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
