"""
Kaggle QAT Training Notebook - Objective 2: Multi-Gear Engine
===============================================================

This script runs on Kaggle to train LoRA adapters for each bit-width gear.
It uses QLoRA for efficient training on T4 GPUs.

Run on Kaggle with GPU T4 x2 accelerator.
"""

# =============================================================================
# CELL 1: Setup and Imports
# =============================================================================

import os
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer

# HuggingFace token (from user)
HF_TOKEN = "hf_your_token_here"

# Model selection - Llama-3.2-1B is small enough for Kaggle T4
MODEL_NAME = "meta-llama/Llama-3.2-1B"
# Alternative: "TinyLlama/TinyLlama-1.1B"

# Paths
DATA_PATH = "/kaggle/input/your-dataset/combined_cleaned.csv"  # Upload dataset to Kaggle
OUTPUT_PATH = "/kaggle/working/adapters"

print("[OK] Imports complete")
print(f"[OK] Using model: {MODEL_NAME}")
print(f"[OK] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# CELL 2: Load and Prepare Dataset
# =============================================================================

def format_prompt(row):
    """Format a row into instruction-response template."""
    return f"""### Instruction:
{row['question']}

### Response:
{row['response']}"""


def load_and_tokenize_dataset(csv_path: str, tokenizer, max_length: int = 512):
    """
    Load CSV and create tokenized dataset splits by complexity.

    Returns separate datasets for simple, medium, complex.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Format prompts
    df['text'] = df.apply(format_prompt, axis=1)

    # Split by complexity
    simple_df = df[df['complexity'] == 'simple'].copy()
    medium_df = df[df['complexity'] == 'medium'].copy()
    complex_df = df[df['complexity'] == 'complex'].copy()

    print(f"[OK] Simple: {len(simple_df)} samples")
    print(f"[OK] Medium: {len(medium_df)} samples")
    print(f"[OK] Complex: {len(complex_df)} samples")

    # Balance datasets (take min count from each)
    min_count = min(len(simple_df), len(medium_df), len(complex_df))
    simple_df = simple_df.sample(min_count, random_state=42)
    medium_df = medium_df.sample(min_count, random_state=42)
    complex_df = complex_df.sample(min_count, random_state=42)

    print(f"\n[OK] Balanced to {min_count} samples per complexity")

    # Convert to HuggingFace Datasets
    simple_ds = Dataset.from_pandas(simple_df[['text']])
    medium_ds = Dataset.from_pandas(medium_df[['text']])
    complex_ds = Dataset.from_pandas(complex_df[['text']])

    return simple_ds, medium_ds, complex_ds


# Test loading (when running on Kaggle)
# simple_ds, medium_ds, complex_ds = load_and_tokenize_dataset(DATA_PATH, tokenizer)


# =============================================================================
# CELL 3: Model Loading with Quantization
# =============================================================================

def load_base_model(model_name: str, load_in_4bit: bool = True):
    """
    Load base model with optional 4-bit quantization.

    This is the 'fake quantization' trick - model loads in low precision
    but gradients still flow for LoRA training.
    """
    print(f"Loading {model_name}...")

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare for training
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    print("[OK] Model loaded")
    print(f"[OK] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer


# =============================================================================
# CELL 4: LoRA Configuration
# =============================================================================

def get_lora_config(r: int = 16, lora_alpha: int = 32, dropout: float = 0.05):
    """
    Create LoRA configuration.

    Args:
        r: LoRA rank (lower = fewer params, higher = more expressive)
        lora_alpha: Scaling factor
        dropout: Regularization
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


# =============================================================================
# CELL 5: Training Function
# =============================================================================

def train_adapter(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    adapter_name: str,
    num_epochs: int = 1,
    batch_size: int = 4,
):
    """
    Train a LoRA adapter for a specific complexity/bit-width.

    Args:
        model: Base model (already has LoRA attached)
        tokenizer: Tokenizer
        dataset: Training dataset
        output_dir: Where to save adapter
        adapter_name: Name of adapter (e.g., "adapter_4bit")
        num_epochs: Training epochs
        batch_size: Batch size
    """
    print(f"\n{'='*60}")
    print(f"Training {adapter_name}")
    print(f"{'='*60}")
    print(f"Samples: {len(dataset)}")
    print(f"Epochs: {num_epochs}")

    # Training arguments optimized for Kaggle T4
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{adapter_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",  # Disable wandb for Kaggle
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=512,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save adapter
    save_path = f"{output_dir}/{adapter_name}"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"[OK] Saved adapter to {save_path}")

    return trainer.model


# =============================================================================
# CELL 6: Main Training Pipeline
# =============================================================================

def train_all_gears(
    csv_path: str,
    output_dir: str = "/kaggle/working/adapters",
    model_name: str = MODEL_NAME,
):
    """
    Train all three gear adapters.

    This is the main entry point for Objective 2.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load base model in 4-bit (fake quantization for QAT)
    print("\n" + "="*60)
    print("LOADING BASE MODEL (4-bit for QAT)")
    print("="*60)
    model, tokenizer = load_base_model(model_name, load_in_4bit=True)

    # Load datasets
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    simple_ds, medium_ds, complex_ds = load_and_tokenize_dataset(
        csv_path, tokenizer
    )

    # Train Gear 1: 4-bit (Simple prompts)
    # Use higher LoRA rank for lower bit-width to compensate
    print("\n" + "="*60)
    print("GEAR 1: 4-BIT ADAPTER (Simple Prompts)")
    print("="*60)
    lora_config_4bit = get_lora_config(r=32, lora_alpha=64)  # Higher rank for 4-bit
    model_4bit = get_peft_model(model, lora_config_4bit)
    train_adapter(
        model_4bit, tokenizer, simple_ds,
        output_dir, "adapter_4bit",
        num_epochs=2,  # More epochs for 4-bit
    )

    # Train Gear 2: 8-bit (Medium prompts)
    print("\n" + "="*60)
    print("GEAR 2: 8-BIT ADAPTER (Medium Prompts)")
    print("="*60)
    # Reload base model to clear 4-bit adapter
    model, tokenizer = load_base_model(model_name, load_in_4bit=True)
    lora_config_8bit = get_lora_config(r=16, lora_alpha=32)
    model_8bit = get_peft_model(model, lora_config_8bit)
    train_adapter(
        model_8bit, tokenizer, medium_ds,
        output_dir, "adapter_8bit",
        num_epochs=1,
    )

    # Train Gear 3: 16-bit (Complex prompts)
    print("\n" + "="*60)
    print("GEAR 3: 16-BIT ADAPTER (Complex Prompts)")
    print("="*60)
    # For 16-bit, load without 4-bit quantization
    model, tokenizer = load_base_model(model_name, load_in_4bit=False)
    lora_config_16bit = get_lora_config(r=8, lora_alpha=16)  # Lower rank, full precision
    model_16bit = get_peft_model(model, lora_config_16bit)
    train_adapter(
        model_16bit, tokenizer, complex_ds,
        output_dir, "adapter_16bit",
        num_epochs=1,
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Adapters saved to: {output_dir}")
    print("\nOutput structure:")
    print("  adapters/")
    print("    adapter_4bit/  - For simple prompts (greetings, facts)")
    print("    adapter_8bit/  - For medium prompts (summarization)")
    print("    adapter_16bit/ - For complex prompts (code, reasoning)")


# =============================================================================
# CELL 7: Validation Function
# =============================================================================

def validate_adapters(
    model_name: str = MODEL_NAME,
    adapter_dir: str = "/kaggle/working/adapters",
):
    """
    Validate each adapter with sample prompts.

    Logs outputs for qualitative analysis.
    """
    test_prompts = {
        "simple": [
            "Hi, how are you?",
            "What is the capital of France?",
            "Tell me a fun fact.",
        ],
        "medium": [
            "Summarize the concept of machine learning.",
            "Explain photosynthesis in simple terms.",
            "What are the benefits of exercise?",
        ],
        "complex": [
            "Write a Python function to implement quicksort.",
            "Explain the time complexity of Dijkstra's algorithm.",
            "Derive the quadratic formula from ax² + bx + c = 0.",
        ],
    }

    results = []

    for gear in ["4bit", "8bit", "16bit"]:
        print(f"\n{'='*60}")
        print(f"Testing {gear} adapter")
        print(f"{'='*60}")

        # Load model with adapter
        if gear == "16bit":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                token=HF_TOKEN,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto",
                token=HF_TOKEN,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token

        # Load adapter
        adapter_path = f"{adapter_dir}/adapter_{gear}"
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        # Test on all prompt types
        for prompt_type, prompts in test_prompts.items():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                results.append({
                    "gear": gear,
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                    "response": response[len(prompt):].strip(),  # Remove prompt
                })

                print(f"\n[{gear}] [{prompt_type}] {prompt[:40]}...")
                print(f"Response: {response[len(prompt):].strip()[:100]}...")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{adapter_dir}/validation_results.csv", index=False)
    print(f"\n[OK] Validation results saved to {adapter_dir}/validation_results.csv")

    return results_df


# =============================================================================
# CELL 8: Run Training (Main Entry Point)
# =============================================================================

if __name__ == "__main__":
    # This runs the full training pipeline
    # Upload your combined_cleaned.csv to Kaggle first

    # Uncomment to run:
    # train_all_gears(
    #     csv_path="/kaggle/input/green-weight-data/combined_cleaned.csv",
    #     output_dir="/kaggle/working/adapters",
    # )

    # Uncomment to validate:
    # validate_adapters()

    print("Ready to run!")
    print("1. Upload combined_cleaned.csv to Kaggle dataset")
    print("2. Run train_all_gears() to train adapters")
    print("3. Run validate_adapters() to test outputs")
