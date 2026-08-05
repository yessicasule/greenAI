"""
Kaggle Routing Experiment — Session 4 of the research plan
==========================================================

The paper's main experiment. Produces the accuracy-vs-energy numbers for
all experimental conditions:

  1. static_16bit   2. static_8bit    3. static_4bit
  4. fuzzy_router (ours)              5. random_matched (tier mix matched to #4)
  6. threshold_router (naive 33/66)   7. oracle (cheapest correct tier)
  8. oracle_cascade (4->8->16 escalate-until-correct, cumulative energy)

Design: Phase A measures EVERY prompt on EVERY tier once (energy via the
NVML hardware counter, greedy decoding). Because greedy decoding is
deterministic, any router's per-prompt response and energy equal those of
the tier it selects — so Phase B derives conditions 4-8 from Phase A's
measurements without re-running the GPU. Router compute overhead is
measured separately (it is CPU-only and reported, not ignored).

How to run on Kaggle:
  1. New notebook -> Accelerator: GPU T4 x1. Secret HF_TOKEN set.
  2. Upload eval_prompts.jsonl (from scripts/prepare_eval_dataset.py) as a
     Kaggle dataset; adjust EVAL_FILE below if the path differs.
  3. Optional: upload adapters/ dataset and set ADAPTER_ROOT to use the
     QAT adapters per tier (the "full system" configuration).
  4. Paste this file into one cell and run. Budget: ~4-6 h for 500 prompts.

Outputs (/kaggle/working):
  routing_per_prompt.csv        - per (prompt x tier): energy, tokens, correct
  routing_conditions_summary.csv- per condition: accuracy, J/token, J/request
  routing_run_info.json         - hardware, config, router overhead
"""

import csv
import gc
import json
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

if Path("/kaggle").exists():
    os.system("pip -q install pynvml bitsandbytes accelerate peft")

import pynvml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.2-1B"
MAX_NEW_TOKENS = 128
N_WARMUP = 5
SEED = 42
TIERS = ["4bit", "8bit", "16bit"]
TIER_THRESHOLDS = {"4bit_upper": 33, "8bit_upper": 66}   # matches config.yaml
RANDOM_ROUTER_RESAMPLES = 20
EVAL_FILE_CANDIDATES = [
    "/kaggle/input/eval-prompts/eval_prompts.jsonl",
    "eval_prompts.jsonl",
]
ADAPTER_ROOT = "/kaggle/input/greenweight-adapters"  # set to None to disable
ADAPTER_FOR_TIER = {"4bit": "adapter_simple", "8bit": "adapter_medium",
                    "16bit": "adapter_complex"}
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


# ====================================================================
# Complexity sensor + fuzzy gearbox
# (verbatim logic from green_weight/core/prompt_complexity.py and
#  green_weight/controllers/fuzzy_gearbox.py, embedded for portability)
# ====================================================================
class ComplexityScorer:
    REASONING_KEYWORDS = {
        "explain", "why", "how", "analyze", "compare", "contrast",
        "evaluate", "justify", "prove", "derive", "solve", "step",
        "reasoning", "logic", "therefore", "because", "consequence",
    }
    CODE_PATTERNS = [r"```[\s\S]*?```", r"`[^`]+`", r"def\s+\w+", r"class\s+\w+",
                     r"import\s+\w+", r"for\s+\w+\s+in", r"if\s+.*?:"]
    MATH_PATTERNS = [r"[\+\-\*\/\=\^√∑∏∫]",
                     r"\d+\s*[\+\-\*\/\^]\s*\d+",
                     r"\b(equation|formula|calculate|compute|derivative|integral)\b"]

    def __init__(self):
        self._code = [re.compile(p, re.IGNORECASE) for p in self.CODE_PATTERNS]
        self._math = [re.compile(p, re.IGNORECASE) for p in self.MATH_PATTERNS]

    def calculate_complexity(self, prompt: str) -> float:
        text = prompt.strip()
        words = text.split()
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        wc, sc = len(words), len(sentences)
        awl = sum(len(w) for w in words) / max(wc, 1)
        fk = 0.0 if sc == 0 else (0.39 * wc / sc) + (11.8 * awl * 0.5) - 15.59
        code = sum(len(p.findall(text)) for p in self._code)
        mathn = sum(len(p.findall(text)) for p in self._math)
        tl = text.lower()
        reasoning = sum(1 for k in self.REASONING_KEYWORDS if k in tl)
        score = (min(fk * 2, 30) + min(wc / 10, 25) + min(code * 10, 25)
                 + min(mathn * 5, 15) + min(reasoning * 5, 15)
                 + min(text.count("?") * 5, 10))
        return min(score, 100.0)


class FuzzyGearbox:
    """Triangular memberships at centers 20/50/80, centroid defuzzification."""
    CENTERS = {"simple": 20.0, "medium": 50.0, "complex": 80.0}
    OUTPUTS = {"simple": 4.0, "medium": 8.0, "complex": 16.0}
    WIDTH = 100.0 / 6

    @staticmethod
    def _tri(x, c, w):
        if x < c - w or x > c + w:
            return 0.0
        return (x - (c - w)) / w if x < c else ((c + w) - x) / w

    def decide(self, complexity: float) -> str:
        x = max(0.0, min(complexity, 100.0))
        m = {k: self._tri(x, c, self.WIDTH) for k, c in self.CENTERS.items()}
        total = sum(m.values())
        if total == 0:
            m = {"simple": 0.33, "medium": 0.34, "complex": 0.33}
            total = 1.0
        crisp = sum(m[k] * self.OUTPUTS[k] for k in m) / total
        bits = min([4, 8, 16], key=lambda b: abs(b - crisp))
        return f"{bits}bit"


def threshold_router(complexity: float) -> str:
    if complexity <= TIER_THRESHOLDS["4bit_upper"]:
        return "4bit"
    if complexity <= TIER_THRESHOLDS["8bit_upper"]:
        return "8bit"
    return "16bit"


# ====================================================================
# Correctness proxy: reference-answer matching (documented in the paper
# as a proxy metric; benchmark accuracy comes from Session 2)
# ====================================================================
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def is_correct(response: str, reference: str) -> bool:
    if not reference:
        return False
    ref_n, resp_n = _norm(reference), _norm(response)
    # numeric reference (e.g. GSM8K): final number must appear in response
    ref_nums = _NUM_RE.findall(reference)
    if ref_nums and len(ref_n.split()) <= 3:
        resp_nums = _NUM_RE.findall(response)
        return ref_nums[-1] in resp_nums
    # short factual reference: substring match
    if len(ref_n.split()) <= 6:
        return ref_n in resp_n
    # longer reference: stem-level token-F1 >= 0.5 (5-char stems so that
    # morphological variants like evaporates/evaporation still match)
    def stems(tokens):
        return {t[:5] for t in tokens}
    ref_t, resp_t = stems(ref_n.split()), stems(resp_n.split())
    if not ref_t or not resp_t:
        return False
    overlap = len(ref_t & resp_t)
    p, r = overlap / len(resp_t), overlap / len(ref_t)
    return (2 * p * r / (p + r) if p + r else 0.0) >= 0.5


# ====================================================================
# NVML energy meter (same protocol as Session 1)
# ====================================================================
class GpuEnergyMeter:
    def __init__(self, gpu_index=0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.has_counter = True
        except pynvml.NVMLError:
            self.has_counter = False
        self._sampling = False

    def _counter_j(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0

    def _loop(self):
        prev = time.perf_counter()
        while self._sampling:
            time.sleep(0.05)
            now = time.perf_counter()
            w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            self._integrated += w * (now - prev)
            prev = now

    def start(self):
        if self.has_counter:
            self._start = self._counter_j()
        else:
            self._integrated = 0.0
            self._sampling = True
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()

    def stop(self):
        if self.has_counter:
            return self._counter_j() - self._start
        self._sampling = False
        self._t.join()
        return self._integrated

    def info(self):
        return {"gpu": pynvml.nvmlDeviceGetName(self.handle),
                "driver": pynvml.nvmlSystemGetDriverVersion(),
                "energy_counter": self.has_counter}


# ====================================================================
# Phase A: measure every prompt on every tier
# ====================================================================
def load_tier(tier, adapter_root):
    kwargs = {"device_map": {"": 0}}
    if tier == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
    elif tier == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    used_adapter = False
    if adapter_root:
        apath = Path(adapter_root) / ADAPTER_FOR_TIER[tier]
        if apath.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(apath))
            used_adapter = True
    model.eval()
    return model, used_adapter


def load_prompts():
    for path in EVAL_FILE_CANDIDATES:
        if Path(path).exists():
            rows, seen = [], set()
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r["prompt"] not in seen:
                            seen.add(r["prompt"])
                            rows.append(r)
            print(f"Loaded {len(rows)} unique prompts from {path}")
            return rows
    raise FileNotFoundError("eval_prompts.jsonl not found — upload it as a dataset")


def phase_a(prompts, meter, tokenizer, adapter_root):
    measurements = {}   # (prompt_id, tier) -> dict
    adapters_used = {}
    for tier in TIERS:
        print(f"\n===== Phase A: tier {tier} =====")
        model, used_adapter = load_tier(tier, adapter_root)
        adapters_used[tier] = used_adapter
        print(f"  QAT adapter loaded: {used_adapter}")

        def generate(prompt):
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            n_new = out.shape[1] - inputs["input_ids"].shape[1]
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            return text, n_new

        for p in prompts[:N_WARMUP]:
            generate(p["prompt"])

        for i, p in enumerate(prompts):
            meter.start()
            t0 = time.perf_counter()
            text, n_tok = generate(p["prompt"])
            latency = time.perf_counter() - t0
            joules = meter.stop()
            measurements[(i, tier)] = {
                "prompt_id": i, "tier": tier,
                "difficulty": p.get("difficulty_label", "?"),
                "energy_j": joules, "tokens_out": n_tok,
                "j_per_token": joules / max(n_tok, 1),
                "latency_s": latency,
                "correct": is_correct(text, p.get("reference_answer", "")),
                "response": text,
            }
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(prompts)} prompts done")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return measurements, adapters_used


# ====================================================================
# Phase B: derive all conditions from Phase A measurements
# ====================================================================
def summarize_condition(name, picks, meas, extra=None):
    """picks: list of (prompt_id, [tiers_actually_executed], billed_tier)."""
    energy, tokens, correct, dist = 0.0, 0, 0, {t: 0 for t in TIERS}
    latency = 0.0
    for pid, executed, billed in picks:
        for t in executed:
            m = meas[(pid, t)]
            energy += m["energy_j"]
            tokens += m["tokens_out"]
            latency += m["latency_s"]
        correct += int(meas[(pid, billed)]["correct"])
        dist[billed] += 1
    n = len(picks)
    row = {
        "condition": name, "n_prompts": n,
        "accuracy": round(correct / n, 4),
        "total_energy_j": round(energy, 1),
        "j_per_request": round(energy / n, 2),
        "j_per_token": round(energy / max(tokens, 1), 4),
        "mean_latency_s": round(latency / n, 3),
        "pct_4bit": round(dist["4bit"] / n, 3),
        "pct_8bit": round(dist["8bit"] / n, 3),
        "pct_16bit": round(dist["16bit"] / n, 3),
    }
    if extra:
        row.update(extra)
    return row


def phase_b(prompts, meas):
    scorer, gearbox = ComplexityScorer(), FuzzyGearbox()
    rng = random.Random(SEED)

    # router overhead (CPU): time the scorer over the whole set
    t0 = time.perf_counter()
    complexity = [scorer.calculate_complexity(p["prompt"]) for p in prompts]
    router_overhead_s = time.perf_counter() - t0

    n = len(prompts)
    ids = list(range(n))
    rows = []

    # 1-3 static
    for tier in TIERS:
        rows.append(summarize_condition(
            f"static_{tier}", [(i, [tier], tier) for i in ids], meas))

    # 4 fuzzy
    fuzzy_tiers = [gearbox.decide(c) for c in complexity]
    rows.append(summarize_condition(
        "fuzzy_router", [(i, [fuzzy_tiers[i]], fuzzy_tiers[i]) for i in ids], meas))

    # 5 random router with tier distribution matched to fuzzy (mean of resamples)
    resample_rows = []
    for _ in range(RANDOM_ROUTER_RESAMPLES):
        shuffled = fuzzy_tiers[:]
        rng.shuffle(shuffled)
        resample_rows.append(summarize_condition(
            "random_matched", [(i, [shuffled[i]], shuffled[i]) for i in ids], meas))
    avg = {k: round(sum(r[k] for r in resample_rows) / len(resample_rows), 4)
           for k in resample_rows[0] if isinstance(resample_rows[0][k], (int, float))}
    avg["condition"] = "random_matched"
    avg["n_resamples"] = RANDOM_ROUTER_RESAMPLES
    rows.append(avg)

    # 6 threshold
    th = [threshold_router(c) for c in complexity]
    rows.append(summarize_condition(
        "threshold_router", [(i, [th[i]], th[i]) for i in ids], meas))

    # 7 oracle: cheapest tier that is correct (16bit if none)
    oracle_picks = []
    for i in ids:
        pick = next((t for t in TIERS if meas[(i, t)]["correct"]), "16bit")
        oracle_picks.append((i, [pick], pick))
    rows.append(summarize_condition("oracle", oracle_picks, meas))

    # 8 oracle cascade: escalate 4->8->16 until correct; energy is cumulative
    cascade_picks = []
    for i in ids:
        executed = []
        for t in TIERS:
            executed.append(t)
            if meas[(i, t)]["correct"]:
                break
        cascade_picks.append((i, executed, executed[-1]))
    rows.append(summarize_condition("oracle_cascade", cascade_picks, meas))

    return rows, complexity, fuzzy_tiers, router_overhead_s


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    meter = GpuEnergyMeter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = load_prompts()

    adapter_root = ADAPTER_ROOT if ADAPTER_ROOT and Path(ADAPTER_ROOT).exists() else None
    meas, adapters_used = phase_a(prompts, meter, tokenizer, adapter_root)
    rows, complexity, fuzzy_tiers, overhead = phase_b(prompts, meas)

    # per-prompt dump (paper's raw data artifact)
    with open(OUT_DIR / "routing_per_prompt.csv", "w", newline="",
              encoding="utf-8") as f:
        fields = ["prompt_id", "tier", "difficulty", "energy_j", "tokens_out",
                  "j_per_token", "latency_s", "correct", "complexity",
                  "fuzzy_tier", "response"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (pid, tier), m in sorted(meas.items()):
            row = dict(m)
            row["complexity"] = round(complexity[pid], 2)
            row["fuzzy_tier"] = fuzzy_tiers[pid]
            row["energy_j"] = round(row["energy_j"], 3)
            row["j_per_token"] = round(row["j_per_token"], 4)
            row["latency_s"] = round(row["latency_s"], 3)
            w.writerow(row)

    fields = sorted({k for r in rows for k in r})
    with open(OUT_DIR / "routing_conditions_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    info = meter.info()
    info.update({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID, "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED,
        "n_prompts": len(prompts), "adapters_used": adapters_used,
        "router_overhead_s_total": round(overhead, 4),
        "router_overhead_ms_per_prompt": round(1000 * overhead / len(prompts), 3),
        "correctness_metric": "reference-match proxy (see is_correct docstring)",
    })
    (OUT_DIR / "routing_run_info.json").write_text(json.dumps(info, indent=2))

    print("\n================= CONDITION SUMMARY =================")
    for r in rows:
        print(f"  {r['condition']:>18}: acc={r['accuracy']:.3f}  "
              f"J/req={r['j_per_request']}  J/tok={r['j_per_token']}")
    base = next(r for r in rows if r["condition"] == "static_16bit")
    ours = next(r for r in rows if r["condition"] == "fuzzy_router")
    saving = 100 * (1 - ours["j_per_request"] / base["j_per_request"])
    print(f"\n  Fuzzy vs static fp16: {saving:.1f}% energy/request saved, "
          f"accuracy {base['accuracy']:.3f} -> {ours['accuracy']:.3f}")
    print("\nSaved: routing_per_prompt.csv, routing_conditions_summary.csv, "
          "routing_run_info.json — download all three.")


if __name__ == "__main__":
    main()
