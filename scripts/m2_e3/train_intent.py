import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def prepare_dataset(split_path: Path, labels: List[str]) -> Dataset:
    data = list(load_jsonl(split_path))
    label_to_id = {l: i for i, l in enumerate(labels)}

    def map_labels(example):
        vec = [0.0] * len(labels)
        for l in example.get("intent_labels", []):
            if l in label_to_id:
                vec[label_to_id[l]] = 1.0
        example["label_vec"] = vec
        return example

    return Dataset.from_list([map_labels(ex) for ex in data])


def compute_metrics_builder(threshold: float = 0.5):
    def compute(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        labels = labels.astype(int)
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    return compute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=Path("experiments/m2_e3_parse/data/splits"))
    parser.add_argument("--model-name", type=str, default="microsoft/MiniLM-L6-v2")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/m2_e3_parse/artifacts/intent"))
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    args = parser.parse_args()

    labels_path = args.splits_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.json at {labels_path}. Run data_prep.py first.")
    labels: List[str] = json.loads(labels_path.read_text())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = prepare_dataset(args.splits_dir / "train.jsonl", labels)
    val_ds = prepare_dataset(args.splits_dir / "val.jsonl", labels)

    def tokenize(batch):
        enc = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=args.max_length)
        enc["labels"] = batch["label_vec"]
        return enc

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.logging_steps,
        save_strategy="no",
        save_steps=args.logging_steps,
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_builder(),
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with (args.output_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


if __name__ == "__main__":
    main()
