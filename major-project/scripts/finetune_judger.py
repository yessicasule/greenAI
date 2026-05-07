"""
Fine-tune Cascade Judger
========================

Fine-tunes FrugalGPT's DistilBERT generation judger on a local dataset
so it can assess output quality for the green-weight cascade.

Usage:
    python scripts/finetune_judger.py \
        --dataset green_weight/data/eval_prompts.jsonl \
        --output models/judger_finetuned \
        --num-epochs 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from torch.utils.data import Dataset
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

_DIFFICULTY_TO_LABEL = {"easy": 0, "medium": 1, "hard": 2}


class JudgerDataset(Dataset):
    """Pre-tokenized dataset mapping prompt+answer pairs to difficulty labels."""

    def __init__(self, data: List[Dict], tokenizer):
        labels = [
            _DIFFICULTY_TO_LABEL.get(row.get("difficulty_label", "medium"), 1)
            for row in data
        ]
        texts = [
            row["prompt"] + " [SEP] " + row.get("reference_answer", "")
            for row in data
        ]
        encodings = tokenizer(
            texts, truncation=True, max_length=256, padding="max_length"
        )
        self.input_ids = [torch.tensor(x) for x in encodings["input_ids"]]
        self.attention_mask = [torch.tensor(x) for x in encodings["attention_mask"]]
        self.labels = [torch.tensor(l) for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finetune_judger(dataset_path: Path, output_dir: Path, num_epochs: int, device: str):
    if not _DEPS_OK:
        print("[FAIL] Missing dependencies: torch, transformers")
        print("Install with: pip install torch transformers")
        sys.exit(1)

    rows = load_jsonl(dataset_path)
    print(f"Loaded {len(rows)} examples from {dataset_path}")

    split = int(len(rows) * 0.8)
    train_rows, val_rows = rows[:split], rows[split:]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        no_cuda=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=JudgerDataset(train_rows, tokenizer),
        eval_dataset=JudgerDataset(val_rows, tokenizer),
    )

    print(f"Fine-tuning judger for {num_epochs} epochs...")
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[OK] Saved fine-tuned judger to {output_dir}")
    print("Update config.yaml: cascade.judger.base_model =", str(output_dir))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune cascade judger")
    parser.add_argument("--dataset", type=str,
                        default="green_weight/data/eval_prompts.jsonl",
                        help="Path to eval_prompts.jsonl")
    parser.add_argument("--output", type=str, default="models/judger_finetuned",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--num-epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[FAIL] Dataset not found: {dataset_path}")
        print("Run first: python scripts/prepare_eval_dataset.py")
        sys.exit(1)

    finetune_judger(
        dataset_path=dataset_path,
        output_dir=Path(args.output),
        num_epochs=args.num_epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
