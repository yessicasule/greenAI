"""
Kaggle Accuracy Evaluation — Session 2 of the research plan
===========================================================

Evaluates Llama-3.2-1B at each precision tier (4-bit / 8-bit / fp16),
with and without the QAT LoRA adapters, using lm-eval-harness.

Answers RQ4 (do QAT adapters help low-bit quality?) and provides the
per-tier accuracy column of the paper's results table.

How to run on Kaggle:
  1. New notebook -> Accelerator: GPU T4 x1.
  2. Kaggle secret HF_TOKEN set; Llama-3.2 license accepted on HF.
  3. (For adapter runs) upload the repo's `adapters/` folder as a Kaggle
     dataset named e.g. "greenweight-adapters" and set ADAPTER_ROOT below.
  4. Paste this file into one cell and run. Budget: ~4-6 h.

Outputs (/kaggle/working):
  accuracy_per_tier.json   - full lm-eval results per condition
  accuracy_summary.csv     - one row per (tier, adapter) x task
"""

import csv
import gc
import json
import os
from pathlib import Path

if Path("/kaggle").exists():
    os.system("pip -q install lm-eval bitsandbytes accelerate peft")

import warnings

# transformers resets warning filters to "always" in places, defeating
# Python's once-per-site dedup for the cast warning fixed in load_tier.
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.2-1B"
# tinyBenchmarks tasks give benchmark-faithful scores in ~100 examples each.
# Fallback tasks (with LIMIT applied) if tiny* are unavailable in this lm-eval.
PREFERRED_TASKS = ["tinyMMLU", "tinyGSM8k", "tinyHellaswag"]
FALLBACK_TASKS = ["arc_easy", "gsm8k", "hellaswag"]
FALLBACK_LIMIT = 200          # examples per fallback task
ADAPTER_ROOT = "/kaggle/input/greenweight-adapters"   # or None to skip adapters


def _find_kaggle_path(basename):
    """Locate a Kaggle input file/dir by basename regardless of mount
    layout. Kaggle has mounted dataset inputs under two different layouts
    observed in this project: classic /kaggle/input/<slug>/... and, on
    some accounts/notebooks, /kaggle/input/datasets/<username>/<slug>/...
    — confirmed 2026-08-13. Glob instead of hardcoding one."""
    import glob
    for pattern in (
        f"/kaggle/input/{basename}",
        f"/kaggle/input/*/{basename}",
        f"/kaggle/input/datasets/*/{basename}",
        f"/kaggle/input/datasets/*/*/{basename}",
    ):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


_SCRIPT_DIR = Path(__file__).resolve().parent

_found_adapters = _find_kaggle_path("greenweight-adapters")
if not _found_adapters and not Path(ADAPTER_ROOT).exists():
    # Off Kaggle the adapters live at major-project/adapters/. Without this
    # the Kaggle-only default misses and main() falls back to base-model-only
    # evals -- which silently drops the with/without-QAT comparison that is
    # the entire point of Session 2.
    for _cand in (_SCRIPT_DIR.parent.parent / "adapters", Path("adapters")):
        if (_cand / "adapter_simple").exists():
            _found_adapters = str(_cand.resolve())
            break
if _found_adapters:
    ADAPTER_ROOT = _found_adapters
# tier -> adapter trained for the matching complexity band
ADAPTER_FOR_TIER = {
    "4bit": "adapter_simple",
    "8bit": "adapter_medium",
    "16bit": "adapter_complex",
}
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def load_tier(tier):
    # torch_dtype on EVERY tier, not just 16bit. Without it the modules
    # bitsandbytes leaves unquantized keep the model's declared bfloat16,
    # so every Linear8bitLt matmul casts to fp16 and warns about it. In
    # the routing experiment that produced 1,103,631 warnings and a 93 MB
    # log, and the job was bottlenecked writing warning text rather than
    # computing. lm-eval runs far more generations than that, so this
    # would be worse here.
    kwargs = {"device_map": {"": 0}, "torch_dtype": torch.float16}
    if tier == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif tier == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    return AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)


def resolve_tasks():
    """Use tiny* tasks when available, else fallback tasks with a limit."""
    from lm_eval.tasks import TaskManager
    available = set(TaskManager().all_tasks)
    if all(t in available for t in PREFERRED_TASKS):
        return PREFERRED_TASKS, None
    print(f"tiny* tasks not available; using {FALLBACK_TASKS} with limit={FALLBACK_LIMIT}")
    return FALLBACK_TASKS, FALLBACK_LIMIT


def evaluate(model, tokenizer, tasks, limit):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")
    try:
        results = lm_eval.simple_evaluate(
            model=lm, tasks=tasks, limit=limit, random_seed=42,
            numpy_random_seed=42, torch_random_seed=42,
        )
        return results["results"]
    finally:
        del lm
        gc.collect()
        torch.cuda.empty_cache()


def main():
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tasks, limit = resolve_tasks()

    adapter_root = Path(ADAPTER_ROOT) if ADAPTER_ROOT else None
    if adapter_root and not adapter_root.exists():
        print(f"[WARN] {adapter_root} not found — running base-model-only evals.")
        adapter_root = None

    all_results = {}
    rows = []

    for tier in ["16bit", "8bit", "4bit"]:
        variants = [("base", None)]
        if adapter_root:
            apath = adapter_root / ADAPTER_FOR_TIER[tier]
            if apath.exists():
                variants.append(("qat_adapter", apath))
            else:
                print(f"[WARN] missing adapter {apath}, skipping adapter run for {tier}")

        for variant, apath in variants:
            name = f"{tier}_{variant}"
            print(f"\n===== Evaluating {name} | tasks={tasks} limit={limit} =====")
            model = load_tier(tier)
            if apath is not None:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(apath))
            model.eval()

            try:
                res = evaluate(model, tokenizer, tasks, limit)
                all_results[name] = res
                for task, metrics in res.items():
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            rows.append({
                                "condition": name, "tier": tier, "variant": variant,
                                "task": task, "metric": metric, "value": value,
                            })
                print(json.dumps(res, indent=2, default=str))
            finally:
                if hasattr(model, 'to'):
                    model.to('cpu')
                del model
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

    (OUT_DIR / "accuracy_per_tier.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    with open(OUT_DIR / "accuracy_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["condition", "tier", "variant", "task", "metric", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved: accuracy_per_tier.json, accuracy_summary.csv — download both.")


if __name__ == "__main__":
    main()
