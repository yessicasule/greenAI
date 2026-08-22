"""
Kaggle Energy Benchmark — Session 1 of the research plan
========================================================

Measures REAL energy per generated token for Llama-3.2-1B at three
precision tiers (4-bit NF4, 8-bit LLM.int8, fp16) on a Kaggle T4 GPU.

This produces the ground-truth energy table (J/token per tier) that the
whole paper rests on. Everything else (routing savings, Pareto curves)
is derived from these numbers plus routing decisions.

How to run on Kaggle:
  1. New notebook -> Settings -> Accelerator: GPU T4 x1 (NOT P100 —
     Pascal lacks the NVML energy counter).
  2. Add your HF token as a Kaggle secret named HF_TOKEN and accept the
     Llama-3.2 license on huggingface.co first.
  3. Upload backend/src/green_weight/data/eval_prompts.jsonl as a dataset
     (optional; falls back to a built-in prompt list).
  4. Paste this whole file into one cell and run.

Outputs (in /kaggle/working):
  energy_per_inference.csv  - one row per (run, tier, prompt)
  energy_summary.csv        - mean/std/95% CI of J/token per tier
  hardware_info.json        - GPU, driver, idle power baseline
"""

import csv
import gc
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- setup
if Path("/kaggle").exists():
    os.system("pip -q install pynvml bitsandbytes accelerate")

import pynvml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.2-1B"
MAX_NEW_TOKENS = 128          # fixed for all tiers — controls the comparison
N_WARMUP = 5
N_RUNS = 3                    # repeat everything 3x for confidence intervals
EVAL_FILE_CANDIDATES = [
    "/kaggle/input/eval-prompts/eval_prompts.jsonl",
    "eval_prompts.jsonl",
]
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def _find_kaggle_path(basename):
    """Locate a Kaggle input file by basename regardless of mount layout.
    Kaggle has mounted dataset inputs under two different layouts observed
    in this project: the classic /kaggle/input/<slug>/... and, on some
    accounts/notebooks, /kaggle/input/datasets/<username>/<slug>/... —
    confirmed 2026-08-13 when the classic-path guess silently fell through
    to the built-in prompt list. Glob for it instead of hardcoding one."""
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

FALLBACK_PROMPTS = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who wrote Romeo and Juliet?",
    "Explain supply and demand.",
    "Describe the water cycle.",
    "What are the main causes of climate change?",
    "Explain the theory of natural selection.",
    "What is the CAP theorem in distributed systems?",
    "Prove that the square root of 2 is irrational.",
    "Explain quantum entanglement and its implications for quantum computing.",
    "Write a recursive algorithm for the Tower of Hanoi and derive its time complexity.",
    "Discuss the philosophical implications of Godel's incompleteness theorems.",
]


def load_prompts():
    found = _find_kaggle_path("eval_prompts.jsonl")
    candidates = ([found] if found else []) + EVAL_FILE_CANDIDATES
    for path in candidates:
        if path and Path(path).exists():
            seen, prompts = set(), []
            with open(path) as f:
                for line in f:
                    p = json.loads(line)["prompt"]
                    if p not in seen:          # the file contains duplicates
                        seen.add(p)
                        prompts.append(p)
            print(f"Loaded {len(prompts)} unique prompts from {path}")
            return prompts
    print(f"Using {len(FALLBACK_PROMPTS)} built-in prompts")
    return FALLBACK_PROMPTS


# ------------------------------------------------------- energy metering
class GpuEnergyMeter:
    """Prefers the NVML total-energy counter (mJ, Volta+/T4 OK);
    falls back to integrating power samples at 20 Hz."""

    def __init__(self, gpu_index=0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.has_counter = True
        except pynvml.NVMLError:
            self.has_counter = False
        print(f"NVML energy counter available: {self.has_counter}")
        self._sampling = False

    def _read_counter_j(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0

    def _sample_loop(self):
        prev = time.perf_counter()
        while self._sampling:
            time.sleep(0.05)
            now = time.perf_counter()
            watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            self._integrated_j += watts * (now - prev)
            prev = now

    def start(self):
        if self.has_counter:
            self._start_j = self._read_counter_j()
        else:
            self._integrated_j = 0.0
            self._sampling = True
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()

    def stop(self):
        if self.has_counter:
            return self._read_counter_j() - self._start_j
        self._sampling = False
        self._thread.join()
        return self._integrated_j

    def idle_baseline(self, seconds=20):
        """Average idle power (W) with no work running."""
        samples = []
        for _ in range(int(seconds / 0.5)):
            samples.append(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
            time.sleep(0.5)
        return sum(samples) / len(samples)

    def info(self):
        # nvmlDeviceGetName / nvmlSystemGetDriverVersion return bytes on
        # some pynvml versions (see check_nvml_energy_support.py, which
        # explicitly handles this) — decode defensively so json.dumps()
        # below can't crash on a bytes value before any measurement runs.
        gpu = pynvml.nvmlDeviceGetName(self.handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        return {
            "gpu": gpu.decode() if isinstance(gpu, bytes) else gpu,
            "driver": driver.decode() if isinstance(driver, bytes) else driver,
            "energy_counter": self.has_counter,
        }


# ---------------------------------------------------------- tier loading
def load_tier(tier):
    """Load Llama-3.2-1B at the given precision. One tier in memory at a time."""
    kwargs = {"device_map": {"": 0}}
    if tier == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16 hardware
        )
    elif tier == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif tier == "16bit":
        kwargs["torch_dtype"] = torch.float16
    else:
        raise ValueError(tier)
    return AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)


def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# --------------------------------------------------------------- measure
def measure_tier(tier, tokenizer, prompts, meter, run_id, writer):
    print(f"\n=== run {run_id} | tier {tier} ===")
    model = load_tier(tier)
    model.eval()

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,                      # greedy => reproducible
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        return out.shape[1] - inputs["input_ids"].shape[1]

    for p in prompts[:N_WARMUP]:
        generate(p)

    rows = []
    for i, prompt in enumerate(prompts):
        meter.start()
        t0 = time.perf_counter()
        n_tokens = generate(prompt)
        latency = time.perf_counter() - t0
        joules = meter.stop()
        row = {
            "run": run_id,
            "tier": tier,
            "prompt_id": i,
            "prompt_chars": len(prompt),
            "tokens_out": n_tokens,
            "energy_j": round(joules, 3),
            "j_per_token": round(joules / max(n_tokens, 1), 4),
            "latency_s": round(latency, 3),
        }
        rows.append(row)
        writer.writerow(row)
    unload(model)
    return rows


def summarize(all_rows):
    by_tier = {}
    for r in all_rows:
        by_tier.setdefault(r["tier"], []).append(r["j_per_token"])
    summary = []
    for tier, vals in by_tier.items():
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(n - 1, 1))
        ci95 = 1.96 * std / math.sqrt(n)
        summary.append({
            "tier": tier, "n": n,
            "mean_j_per_token": round(mean, 4),
            "std": round(std, 4),
            "ci95": round(ci95, 4),
        })
    return summary


def main():
    torch.manual_seed(42)
    meter = GpuEnergyMeter()

    hw = meter.info()
    print("Measuring 20s idle power baseline...")
    hw["idle_power_w"] = round(meter.idle_baseline(20), 2)
    hw["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    hw["model"] = MODEL_ID
    hw["max_new_tokens"] = MAX_NEW_TOKENS
    (OUT_DIR / "hardware_info.json").write_text(json.dumps(hw, indent=2))
    print(json.dumps(hw, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = load_prompts()

    fieldnames = ["run", "tier", "prompt_id", "prompt_chars",
                  "tokens_out", "energy_j", "j_per_token", "latency_s"]
    all_rows = []
    with open(OUT_DIR / "energy_per_inference.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_id in range(1, N_RUNS + 1):
            for tier in ["4bit", "8bit", "16bit"]:
                all_rows += measure_tier(tier, tokenizer, prompts, meter, run_id, writer)
                f.flush()

    summary = summarize(all_rows)
    with open(OUT_DIR / "energy_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\n================ J/token per tier ================")
    for s in summary:
        print(f"  {s['tier']:>6}: {s['mean_j_per_token']} ± {s['ci95']} J/token (n={s['n']})")
    print("\nSaved: energy_per_inference.csv, energy_summary.csv, hardware_info.json")
    print("Download all three — they are the paper's Table 1.")


if __name__ == "__main__":
    main()
