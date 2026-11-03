"""
Extra thesis analyses (CPU only, no new measurements).

Everything here is computed from existing per-prompt logs, so no GPU run is
implied by these numbers. They use the reference-match proxy, not lm-eval.

Outputs in paper/analysis/:
  thesis_s5_by_source.csv        Stack B proxy accuracy per source dataset, paired vs float16
  thesis_energy_per_correct.csv  J per correct answer for each policy
  thesis_threshold_sweep.csv     replay of the fuzzy score at other tier cuts (NEW.md Phase 6)
  thesis_feature_signal.csv      feature vs correctness correlation and cross-validated AUC
  thesis_gsm8k_metrics.csv       strict vs flexible tinyGSM8k scores per configuration

Run from major-project/: python training/scripts/thesis_extra_analysis.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "analysis"
PROMPTS = ROOT / "backend" / "src" / "green_weight" / "data" / "eval_prompts.jsonl"
RNG = np.random.default_rng(42)
FEATS = ["flesch_kincaid", "token_length", "entropy", "syntax_depth", "has_code_or_math"]
N_BOOT = 2000
RUN_DIR = "results/session5_run2_j2094"  # accuracy is identical across runs (greedy decoding)


def correct(series):
    return series.astype(str).str.lower().eq("true").astype(float).to_numpy()


def stack_b(config, batch_size=1):
    path = ROOT / RUN_DIR / f"{config}_bs{batch_size}" / "per_prompt.csv"
    return pd.read_csv(path).set_index("prompt_id").sort_index()


prompts = [json.loads(line) for line in open(PROMPTS, encoding="utf-8")]
source = {i: p["source"].split("/")[-1] for i, p in enumerate(prompts)}

# ---- 1. Stack B accuracy by source dataset, paired against the same model's float16
rows = []
for cfg, base in [("8b_fp8", "8b_fp16"), ("8b_awq4", "8b_fp16"), ("1b_fp8", "1b_fp16")]:
    q, b = stack_b(cfg), stack_b(base)
    assert (q.index == b.index).all()
    src = np.array([source[i] for i in q.index])
    cq, cb = correct(q["correct"]), correct(b["correct"])
    for name in ["ALL"] + sorted(set(src)):
        m = np.ones(len(q), bool) if name == "ALL" else (src == name)
        diff = cq[m] - cb[m]
        boots = RNG.choice(diff, size=(N_BOOT, len(diff)), replace=True).mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"config": cfg, "baseline": base, "source": name, "n": int(m.sum()),
                     "acc_config": cq[m].mean(), "acc_base": cb[m].mean(), "diff": diff.mean(),
                     "ci_lo": lo, "ci_hi": hi, "only_base_correct": int((diff < 0).sum()),
                     "only_config_correct": int((diff > 0).sum())})
pd.DataFrame(rows).round(4).to_csv(OUT / "thesis_s5_by_source.csv", index=False)

# ---- 2. J per correct answer, from the logged policy tables
rows = []
for run in (5, 6, 7):
    table = pd.read_csv(OUT / f"routing_run{run}_conditions_ci.csv")
    for _, r in table.iterrows():
        rows.append({"stack": "A", "run": f"run{run}", "pool": "1B tiers", "policy": r["condition"],
                     "j_per_request": r["j_per_request"], "acc": r["accuracy"],
                     "j_per_correct": r["j_per_request"] / r["accuracy"]})
for name, pool in [("session5_routing_8b_awq4+8b_fp8+8b_fp16_bs1.csv", "8B precisions"),
                   ("session5_routing_1b_fp16+8b_awq4_bs1.csv", "1B vs 8B AWQ")]:
    table = pd.read_csv(OUT / name)
    for _, r in table.iterrows():
        rows.append({"stack": "B", "run": r["run"], "pool": pool, "policy": r["policy"],
                     "j_per_request": r["energy_j_per_req"], "acc": r["proxy_acc"],
                     "j_per_correct": r["energy_j_per_req"] / r["proxy_acc"]})
pd.DataFrame(rows).round(3).to_csv(OUT / "thesis_energy_per_correct.csv", index=False)

# ---- 3. Threshold sensitivity: same fuzzy score, other tier cuts
a = pd.read_csv(ROOT / "routing_run5_per_prompt.csv")
a["ok"] = correct(a["correct"])
score = a.drop_duplicates("prompt_id").set_index("prompt_id")["complexity"].sort_index()
TIERS = ["4bit", "8bit", "16bit"]
E_a = a.pivot(index="prompt_id", columns="tier", values="energy_j").sort_index()[TIERS].to_numpy()
C_a = a.pivot(index="prompt_id", columns="tier", values="ok").sort_index()[TIERS].to_numpy()
B_CFG = ["8b_awq4", "8b_fp8", "8b_fp16"]
E_b = np.column_stack([stack_b(c)["energy_j"].to_numpy() for c in B_CFG])
C_b = np.column_stack([correct(stack_b(c)["correct"]) for c in B_CFG])
s = score.to_numpy()
idx = np.arange(len(s))
rows = []
for lo, hi in [(25, 75), (30, 70), (33, 66), (40, 60), (45, 55)]:
    choice = np.where(s < lo, 0, np.where(s > hi, 2, 1))
    for stack, E, C in [("A run 5", E_a, C_a), ("B run 2, 8B pool", E_b, C_b)]:
        rand = np.array([C[idx, RNG.permutation(choice)].mean() for _ in range(1000)])
        rows.append({"stack": stack, "low_cut": lo, "high_cut": hi,
                     "pct_low": (choice == 0).mean(), "pct_mid": (choice == 1).mean(),
                     "pct_high": (choice == 2).mean(), "j_per_request": E[idx, choice].mean(),
                     "acc": C[idx, choice].mean(), "random_matched_acc": rand.mean(),
                     "random_acc_p2_5": np.percentile(rand, 2.5),
                     "random_acc_p97_5": np.percentile(rand, 97.5)})
pd.DataFrame(rows).round(4).to_csv(OUT / "thesis_threshold_sweep.csv", index=False)

# ---- 4. Do the five features predict correctness or quantization sensitivity?
F = pd.read_csv(OUT / "prompt_features.csv")
key = "prompt_id" if "prompt_id" in F.columns else F.columns[0]
X = F.set_index(key).sort_index()[FEATS].to_numpy()
targets = {
    "A: float16 right, 4-bit wrong": np.clip(C_a[:, 2] - C_a[:, 0], 0, 1),
    "A: float16 correct": C_a[:, 2],
    "B 8B: float16 right, AWQ wrong": np.clip(C_b[:, 2] - C_b[:, 0], 0, 1),
    "B 8B: float16 correct": C_b[:, 2],
    "B: 1B float16 correct": correct(stack_b("1b_fp16")["correct"]),
}
rows = []
z = (X - X.mean(0)) / X.std(0)
for name, y in targets.items():
    row = {"target": name, "positives": int(y.sum())}
    for i, f in enumerate(FEATS):
        rho, p = spearmanr(X[:, i], y)
        row[f"rho_{f}"], row[f"p_{f}"] = rho, p
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    pred = cross_val_predict(LogisticRegression(max_iter=1000), z, y.astype(int), cv=cv,
                             method="predict_proba")[:, 1]
    row["cv_auc"] = roc_auc_score(y, pred)
    rows.append(row)
pd.DataFrame(rows).round(4).to_csv(OUT / "thesis_feature_signal.csv", index=False)

# ---- 5. tinyGSM8k strict vs flexible scoring
rows = []
for path in glob.glob(str(ROOT / "results/session5_acc_j20*/*/*/results_*.json")):
    g = json.load(open(path))["results"]["tinyGSM8k"]
    rows.append({"stack": "B", "config": Path(path).parents[1].name,
                 "strict": g["exact_match,strict-match"],
                 "flexible": g["exact_match,flexible-extract"]})
s2 = pd.read_csv(ROOT / "results/session2_j1784/accuracy_summary.csv")
for cond, g in s2[s2.task == "tinyGSM8k"].groupby("condition"):
    v = g.set_index("metric")["value"]
    rows.append({"stack": "A", "config": cond, "strict": float(v["exact_match,strict-match"]),
                 "flexible": float(v["exact_match,flexible-extract"])})
pd.DataFrame(rows).round(4).sort_values(["stack", "config"]).to_csv(
    OUT / "thesis_gsm8k_metrics.csv", index=False)
print("done")
