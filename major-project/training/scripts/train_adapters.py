"""
Session 3: retrain the 3 QAT LoRA adapters on the cluster.

Script port of adapter-training.ipynb (the notebook that produced the
current adapters) -- same data sampling, LoRA ranks, and training args.
Only paths, token handling, and a smoke-test switch differ.

Env vars:
  HF_TOKEN      required
  CSV_PATH      combined_cleaned.csv (default: major-project/dataset/processed/)
  OUTPUT_DIR    where adapters are written. Default is a NEW folder, never
                major-project/adapters/, so Session 4 runs keep using the
                adapters they started with.
  SMOKE_LIMIT   if set, N samples per class and N training steps -- trial run
"""

import gc
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

MODEL_NAME = "meta-llama/Llama-3.2-1B"
PROJECT = Path(__file__).resolve().parents[2]
CSV_PATH = os.environ.get("CSV_PATH", str(PROJECT / "dataset/processed/combined_cleaned.csv"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(PROJECT / "adapters_retrained"))
SMOKE = int(os.environ["SMOKE_LIMIT"]) if os.environ.get("SMOKE_LIMIT") else None

MAX_SAMPLES_PER_CLASS = SMOKE or 2000
MAX_LENGTH = 256

# (adapter name, complexity label, LoRA r, alpha) -- matches adapter_config.json of the current adapters
ADAPTERS = [
    ("adapter_simple", "simple", 16, 32),
    ("adapter_medium", "medium", 8, 16),
    ("adapter_complex", "complex", 4, 8),
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ["HF_TOKEN"])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = MAX_LENGTH


def format_prompt(row):
    return f"""### Instruction:
{row['question']}

### Response:
{row['response']}"""


def load_datasets():
    df = pd.read_csv(CSV_PATH)
    df["text"] = df.apply(format_prompt, axis=1)
    out = {}
    for _, label, _, _ in ADAPTERS:
        part = df[df["complexity"] == label]
        part = part.sample(min(MAX_SAMPLES_PER_CLASS, len(part)), random_state=42)
        print(f"{label}: {len(part)} samples")
        out[label] = Dataset.from_pandas(part[["text"]])
    return out


def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        token=os.environ["HF_TOKEN"],
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    return model


def train_adapter(dataset, name, r, alpha):
    print(f"\n{'=' * 60}\nTRAINING {name} (r={r}, alpha={alpha})\n{'=' * 60}")
    model = load_model()
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = model.half()
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)

    save_path = f"{OUTPUT_DIR}/{name}"
    args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=1,
        max_steps=SMOKE or -1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1 if SMOKE else 20,
        save_strategy="no" if SMOKE else "epoch",
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=False,
        max_grad_norm=0.0,
        seed=42,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[SAVED] {save_path}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CSV: {CSV_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    if SMOKE:
        print(f"SMOKE TEST: {SMOKE} samples/class, {SMOKE} steps -- not a real run")
    datasets = load_datasets()
    for name, label, r, alpha in ADAPTERS:
        train_adapter(datasets[label], name, r, alpha)
    print("\n[SUCCESS] ALL 3 ADAPTERS TRAINED")
