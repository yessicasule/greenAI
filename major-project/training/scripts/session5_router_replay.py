"""
Offline routing replay on Session 5 measurements (no GPU).

Session 5 runs every prompt on every configuration, so each prompt's energy and
correctness are known for every option. A router is then just a choice of one
option per prompt, and can be evaluated by lookup.

Policies compared over a pool of options (ordered cheapest-first by measured J/request):
  static:<cfg>    always one option
  fuzzy           the project's Mamdani controller (4bit -> cheapest, 8bit -> middle,
                  16bit -> most expensive; with two options 8bit and 16bit both map to the
                  expensive one)
  random_matched  random assignment with the fuzzy router's option mix (mean of 1000 draws)
  learned         logistic regression on the 5 complexity features predicting whether the
                  cheapest option answers correctly; 5-fold cross-fitted, so every prompt is
                  routed by a model that never saw it. Two options only.
  oracle          cheapest option that is correct; cheapest overall if none is

Usage:
  python training/scripts/session5_router_replay.py RUN_DIR [RUN_DIR ...] \
      --pool 1b_fp16,8b_awq4 --pool 8b_awq4,8b_fp8,8b_fp16 --batch-size 1
Outputs paper/analysis/session5_routing_<pool>_bs<bs>.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from session5_analysis import summarise  # noqa: E402

ROOT = HERE.parents[1]
GW = ROOT / "backend" / "src" / "green_weight"
PROMPTS = GW / "data" / "eval_prompts.jsonl"
ANALYSIS = ROOT / "paper" / "analysis"
FEATS_CACHE = ANALYSIS / "prompt_features.csv"
FEATURES = ["flesch_kincaid", "token_length", "entropy", "syntax_depth", "has_code_or_math"]
RNG = np.random.default_rng(42)


def prompt_features():
    """Complexity features + fuzzy tier per prompt_id, in the same order the benchmark uses."""
    if FEATS_CACHE.exists():
        return pd.read_csv(FEATS_CACHE)
    sys.path.insert(0, str(GW))
    from router.complexity_scorer import score
    from router.fuzzy_controller import FuzzyController
    from vllm_energy_benchmark import load_prompts
    fc = FuzzyController()
    rows = []
    for i, p in enumerate(load_prompts(None)):
        f = score(p["prompt"])
        tier, wp = fc.route(f)
        rows.append({"prompt_id": i, **{k: f[k] for k in FEATURES},
                     "fuzzy_tier": tier, "fuzzy_complexity": round(wp * 100, 2)})
    df = pd.DataFrame(rows)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATS_CACHE, index=False)
    return df


def tables(pp, run, pool, bs):
    """prompt_id x option matrices of energy and correctness for one run."""
    g = pp[(pp["run"] == run) & (pp["batch_size"] == bs) & pp["config"].isin(pool)]
    e = g.pivot(index="prompt_id", columns="config", values="energy_j")
    c = g.pivot(index="prompt_id", columns="config", values="correct")
    c = c.astype(str).apply(lambda s: s.str.lower().eq("true"))
    ids = e.dropna().index.intersection(c.dropna().index)
    return e.loc[ids], c.loc[ids]


def learned_choice(X, y_cheap_ok, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    p = np.full(len(y_cheap_ok), 0.5)
    if y_cheap_ok.min() == y_cheap_ok.max():
        return p
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y_cheap_ok):
        m = LogisticRegression(max_iter=1000).fit(X[tr], y_cheap_ok[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def evaluate(E, C, choice):
    """choice: array of column indices per prompt."""
    r = np.arange(len(choice))
    return E[r, choice], C[r, choice]


def replay(pp, feats, run, pool, bs, n_boot=2000):
    e, c = tables(pp, run, pool, bs)
    if e.empty:
        return pd.DataFrame()
    order = e.mean().sort_values().index.tolist()          # cheapest first
    E, C = e[order].to_numpy(), c[order].to_numpy()
    f = feats.set_index("prompt_id").loc[e.index]
    n, k = E.shape
    pol = {}
    for j, cfg in enumerate(order):
        pol[f"static:{cfg}"] = np.full(n, j)
    tier_to_ix = {"4bit": 0, "8bit": min(1, k - 1) if k > 2 else k - 1, "16bit": k - 1}
    pol["fuzzy"] = f["fuzzy_tier"].map(tier_to_ix).to_numpy()
    mix = np.bincount(pol["fuzzy"], minlength=k) / n
    pol["oracle"] = np.where(C.any(1), C.argmax(1), 0)
    if k == 2:
        p = learned_choice(f[FEATURES].to_numpy(), C[:, 0].astype(int))
        for th in (0.5, 0.7, 0.9):
            pol[f"learned@{th}"] = np.where(p >= th, 0, 1)

    best_static = max((s for s in pol if s.startswith("static:")),
                      key=lambda s: (C[np.arange(n), pol[s]].mean(), -E[np.arange(n), pol[s]].mean()))
    be, bc = evaluate(E, C, pol[best_static])
    rows = []

    def row(name, en, co, mixv):
        d_e, d_c = en - be, co.astype(float) - bc
        bi = RNG.integers(0, n, (n_boot, n))
        de_b, dc_b = d_e[bi].mean(1), d_c[bi].mean(1)
        rows.append({"run": run, "pool": "+".join(order), "batch_size": bs, "policy": name,
                     "n": n, "energy_j_per_req": round(en.mean(), 2), "proxy_acc": round(co.mean(), 4),
                     "vs": best_static,
                     "d_energy_j": round(d_e.mean(), 2),
                     "d_energy_ci": f"[{np.percentile(de_b, 2.5):.2f}, {np.percentile(de_b, 97.5):.2f}]",
                     "d_acc": round(d_c.mean(), 4),
                     "d_acc_ci": f"[{np.percentile(dc_b, 2.5):.4f}, {np.percentile(dc_b, 97.5):.4f}]",
                     "mix": " ".join(f"{order[j]}={m:.2f}" for j, m in enumerate(mixv))})

    for name, ch in pol.items():
        en, co = evaluate(E, C, ch)
        row(name, en, co, np.bincount(ch, minlength=k) / n)
    draws = [RNG.choice(k, n, p=mix) for _ in range(1000)]
    en = np.mean([evaluate(E, C, d)[0] for d in draws], 0)
    co = np.mean([evaluate(E, C, d)[1] for d in draws], 0)
    row("random_matched", en, co, mix)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--pool", action="append", required=True,
                    help="comma-separated configs, e.g. 1b_fp16,8b_awq4 (repeatable)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--min-util", type=float, default=50.0)
    args = ap.parse_args()

    summary, pp = summarise(args.run_dirs, args.min_util)
    bad = summary[summary["flag_disrupted"]]
    if not bad.empty:
        print("excluding disrupted:", bad[["run", "config", "batch_size"]].to_dict("records"))
        key = ["run", "config", "batch_size"]
        pp = pp[~pp.set_index(key).index.isin(bad.set_index(key).index)]
    feats = prompt_features()
    pd.set_option("display.width", 220)
    for pool in args.pool:
        cfgs = pool.split(",")
        res = pd.concat([replay(pp, feats, run, cfgs, args.batch_size)
                         for run in sorted(pp["run"].unique())], ignore_index=True)
        if res.empty:
            print(f"\n{pool}: no complete data")
            continue
        out = ANALYSIS / f"session5_routing_{pool.replace(',', '+')}_bs{args.batch_size}.csv"
        res.to_csv(out, index=False)
        print(f"\n=== pool {pool}  bs={args.batch_size}  -> {out.name}")
        print(res[["run", "policy", "energy_j_per_req", "proxy_acc", "d_energy_j", "d_energy_ci",
                   "d_acc", "d_acc_ci", "mix"]].to_string(index=False))


if __name__ == "__main__":
    main()
