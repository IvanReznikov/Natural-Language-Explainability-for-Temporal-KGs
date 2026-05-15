import json
from argparse import ArgumentParser
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiments" / "m2_e4" / "data" / "splits"
OUTPUT_DIR = ROOT / "output" / "m2_e4_taxonomy" / "e4a_taxonomy"
TASK_PREFIX = "result_taxonomy"
LABEL_KEYS = ["label", "result_type", "type"]
TEXT_KEYS = ["text", "result", "narrative", "summary", "content"]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def rec_to_xy(rec):
    label = None
    for lk in LABEL_KEYS:
        if lk in rec:
            label = rec[lk]
            break
    texts = []
    for tk in TEXT_KEYS:
        if tk in rec and rec[tk]:
            texts.append(str(rec[tk]))
    if not texts:
        texts = [str(v) for k, v in rec.items() if k not in LABEL_KEYS]
    if label is None:
        raise ValueError(f"No label key found in record {rec}")
    return " \n ".join(texts), label


def load_split(name: str):
    path = DATA_DIR / f"{TASK_PREFIX}_{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return [rec_to_xy(rec) for rec in load_jsonl(path)]


def build_pipeline():
    return Pipeline([
        (
            "features",
            FeatureUnion(
                [
                    (
                        "word",
                        TfidfVectorizer(
                            ngram_range=(1, 2), max_features=20000, min_df=2, lowercase=True
                        ),
                    ),
                    (
                        "char",
                        TfidfVectorizer(
                            analyzer="char", ngram_range=(3, 5), max_features=40000, min_df=2
                        ),
                    ),
                ]
            ),
        ),
        ("clf", LinearSVC()),
    ])


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "acc": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
    }


def main():
    parser = ArgumentParser(description="Train default taxonomy classifier (E6)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    X_train, y_train = zip(*train)
    X_val, y_val = zip(*val)
    X_test, y_test = zip(*test)

    model = build_pipeline()
    model.fit(X_train, y_train)

    metrics = {
        "val": evaluate(model, X_val, y_val),
        "test": evaluate(model, X_test, y_test),
        "train": evaluate(model, X_train, y_train),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "taxonomy_model.joblib"
    metrics_path = out_dir / "taxonomy_metrics.json"

    joblib.dump(model, model_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Metrics: val acc={metrics['val']['acc']:.3f} f1={metrics['val']['macro_f1']:.3f}; test acc={metrics['test']['acc']:.3f} f1={metrics['test']['macro_f1']:.3f}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
