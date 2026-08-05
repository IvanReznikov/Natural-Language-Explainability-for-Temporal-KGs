import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    TrainerCallback,
)


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_target(row: Dict) -> str:
    # Normalize spans ordering for stability
    spans = sorted(
        row.get("spans", []), key=lambda s: (s.get("start", 0), s.get("end", 0), s.get("label", ""))
    )
    payload = {
        "spans": spans,
        "frame": row.get("frame", {}),
        "intent_labels": row.get("intent_labels", []),
    }
    return json.dumps(payload, ensure_ascii=True)


def prepare_dataset(split_path: Path) -> Dataset:
    rows = load_jsonl(split_path)
    data = [{"input_text": r["text"], "target_text": make_target(r)} for r in rows]
    return Dataset.from_list(data)


class LossLoggerCallback(TrainerCallback):
    """Log losses to stdout every logging step.

    HF logging can be swallowed when running via subprocess with capture_output; this
    callback prints key metrics explicitly so they appear in captured stdout.
    """

    def __init__(self, log_every: int) -> None:
        self.log_every = log_every

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs is None:
            return
        # Only emit on the scheduled logging steps
        if state.global_step % self.log_every != 0:
            return
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        ppl = logs.get("perplexity")
        msg_parts = [f"step={state.global_step}"]
        if loss is not None:
            msg_parts.append(f"loss={loss:.4f}")
        if ppl is not None:
            msg_parts.append(f"ppl={ppl:.4f}")
        if lr is not None:
            msg_parts.append(f"lr={lr:.6f}")
        if msg_parts:
            print("[train_parser_t5] " + " | ".join(msg_parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir", type=Path, default=Path("experiments/m2_e3_parse/data/splits")
    )
    parser.add_argument("--model-name", type=str, default="google/flan-t5-small")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/m2_e3_parse/artifacts/parser")
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = prepare_dataset(args.splits_dir / "train.jsonl")
    val_ds = prepare_dataset(args.splits_dir / "val.jsonl")

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                max_length=args.max_target_length,
                padding="max_length",
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["input_text", "target_text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["input_text", "target_text"])

    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.logging_steps,
        save_strategy="no",
        save_steps=args.logging_steps,
        predict_with_generate=True,
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=[],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LossLoggerCallback(log_every=args.logging_steps)],
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
