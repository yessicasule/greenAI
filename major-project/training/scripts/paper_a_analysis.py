"""Paper A figures and routing case-study analysis (CPU only).

Reads the committed Session 1 and Session 4 CSVs, writes figures to
paper/figures/ and tables to paper/analysis/. Run from major-project/:
    python training/scripts/paper_a_analysis.py [--clean-runs 5 6]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
OUT = ROOT / "paper" / "analysis"
TIERS = ["4bit", "8bit", "16bit"]
TIER_LABEL = {"4bit": "4-bit (NF4)", "8bit": "8-bit (int8)", "16bit": "16-bit (fp16)"}
MIN_TOK = 16
N_BOOT = 2000
RNG = np.random.default_rng(42)

# validated categorical slots (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def per_prompt(run):
    d = pd.read_csv(ROOT / f"routing_run{run}_per_prompt.csv")
    d["tps"] = d.tokens_out / d.latency_s
    d["watts"] = d.energy_j / d.latency_s
    return d


def session1():
    d = pd.read_csv(ROOT / "results" / "energy_per_inference.csv")
    d["tps"] = d.tokens_out / d.latency_s
    d["watts"] = d.energy_j / d.latency_s
    return d


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.png/.pdf")


# ---------------------------------------------------------------- figure 1
def fig_disruption(runs):
    """16-bit throughput over the course of each run: disrupted vs clean."""
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    colors = {r: c for r, c in zip(runs, [ORANGE, BLUE, AQUA, YELLOW])}
    for r in runs:
        s = per_prompt(r)
        s = s[(s.tier == "16bit") & (s.tokens_out >= MIN_TOK)].sort_values("prompt_id")
        ax.scatter(s.prompt_id, s.tps, s=8, color=colors[r], alpha=0.35, linewidths=0)
        roll = s.tps.rolling(25, center=True, min_periods=5).median()
        ax.plot(s.prompt_id, roll, color=colors[r], lw=2)
        ax.annotate(f"Run {r}: median {s.tps.median():.1f} tok/s, {s.watts.median():.0f} W",
                    xy=(s.prompt_id.iloc[-1], roll.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.set_xlabel("Prompt (execution order)")
    ax.set_ylabel("16-bit throughput (tokens/s)")
    ax.set_ylim(0, None)
    ax.set_title("Same model, same GPU: 16-bit throughput in a disrupted vs. clean run",
                 loc="left", fontsize=9.5, color=INK)
    save(fig, "fig1_16bit_disruption")


# ---------------------------------------------------------------- figure 2
def fig_tier_bars(runs):
    """Per-tier throughput, power and J/token for Session 1 and each run."""
    sources = [("Session 1\n(no adapters)", session1())] + \
              [(f"Run {r}", per_prompt(r)) for r in runs]
    rows = []
    for name, d in sources:
        d = d[d.tokens_out >= MIN_TOK]
        for t in TIERS:
            x = d[d.tier == t]
            rows.append({"source": name, "tier": t, "tps": x.tps.median(),
                         "watts": x.watts.median(), "jpt": x.j_per_token.median()})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "tier_medians.csv", index=False)

    metrics = [("tps", "Throughput (tokens/s)"), ("watts", "Mean GPU power (W)"),
               ("jpt", "Energy (J/token)")]
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    n = len(sources)
    w = 0.8 / n
    for ax, (m, label) in zip(axes, metrics):
        for i, (name, _) in enumerate(sources):
            vals = [tab[(tab.source == name) & (tab.tier == t)][m].item() for t in TIERS]
            xs = np.arange(3) + (i - (n - 1) / 2) * w
            ax.bar(xs, vals, w * 0.92, color=colors[i], label=name.replace("\n", " "))
            for x, v in zip(xs, vals):
                ax.text(x, v, f"{v:.0f}" if m != "jpt" else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=6, color=INK2)
        ax.set_xticks(range(3), ["4-bit", "8-bit", "16-bit"])
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.75, -0.14), ncol=n, fontsize=8)
    fig.tight_layout()
    save(fig, "fig2_tier_medians")
    return tab


# ------------------------------------------------------------ routing study
def condition_rows(d, cond, tier_of):
    """One row per prompt: the per-tier measurement the condition selects."""
    pick = d.set_index(["prompt_id", "tier"])
    idx = [(p, tier_of[p]) for p in sorted(tier_of)]
    return pick.loc[idx].reset_index()


def oracle_tiers(d):
    """Cheapest tier (by this run's measured energy) that answers correctly;
    if no tier is correct, the cheapest tier overall. Differs from the
    experiment script's oracle, which assumes fewer bits = cheaper (false
    for 8-bit here); accuracy is identical, energy is the true lower bound."""
    out = {}
    for p, g in d.groupby("prompt_id"):
        ok = g[g.correct]
        out[p] = (ok if len(ok) else g).sort_values("energy_j").tier.iloc[0]
    return out


def boot_ci(x):
    x = np.asarray(x, dtype=float)
    means = RNG.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def routing_study(run):
    d = per_prompt(run)
    summary = pd.read_csv(ROOT / f"routing_run{run}_conditions.csv").set_index("condition")
    prompts = sorted(d.prompt_id.unique())
    first = d.drop_duplicates("prompt_id").set_index("prompt_id")

    conds = {f"static_{t}": {p: t for p in prompts} for t in TIERS}
    conds["fuzzy_router"] = first.final_tier.to_dict()
    conds["oracle"] = oracle_tiers(d)

    rows, picked = [], {}
    for name, tier_of in conds.items():
        r = condition_rows(d, name, tier_of)
        picked[name] = r
        acc_ci, e_ci = boot_ci(r.correct), boot_ci(r.energy_j)
        rows.append({"condition": name, "accuracy": r.correct.mean(),
                     "acc_ci_lo": acc_ci[0], "acc_ci_hi": acc_ci[1],
                     "j_per_request": r.energy_j.mean(),
                     "j_ci_lo": e_ci[0], "j_ci_hi": e_ci[1],
                     **{f"pct_{t}": (r.tier == t).mean() for t in TIERS}})
    tab = pd.DataFrame(rows).set_index("condition")

    # our recomputation must reproduce the script's own summary
    for c in tab.index:
        if c in summary.index:
            assert abs(tab.loc[c, "accuracy"] - summary.loc[c, "accuracy"]) < 1e-3, c
            if c != "oracle":
                assert abs(tab.loc[c, "j_per_request"] - summary.loc[c, "j_per_request"]) < 0.5, c
    for c in ("random_matched", "threshold_router"):
        tab.loc[c, ["accuracy", "j_per_request"]] = summary.loc[c, ["accuracy", "j_per_request"]]

    # paired bootstrap: fuzzy vs static 16-bit, per-prompt differences
    f, s16 = picked["fuzzy_router"], picked["static_16bit"]
    de = f.energy_j.values - s16.energy_j.values
    da = f.correct.values.astype(float) - s16.correct.values.astype(float)
    paired = pd.DataFrame([
        {"comparison": "fuzzy - static_16bit", "metric": "J/request",
         "mean_diff": de.mean(), "ci_lo": boot_ci(de)[0], "ci_hi": boot_ci(de)[1]},
        {"comparison": "fuzzy - static_16bit", "metric": "accuracy",
         "mean_diff": da.mean(), "ci_lo": boot_ci(da)[0], "ci_hi": boot_ci(da)[1]},
    ])

    # by difficulty
    by_diff = []
    for name in ("static_4bit", "static_8bit", "static_16bit", "fuzzy_router", "oracle"):
        r = picked[name].merge(first[["difficulty"]], left_on="prompt_id", right_index=True,
                               suffixes=("_x", ""))
        for diff, g in r.groupby("difficulty"):
            by_diff.append({"condition": name, "difficulty": diff, "n": len(g),
                            "accuracy": g.correct.mean(), "j_per_request": g.energy_j.mean(),
                            **{f"pct_{t}": (g.tier == t).mean() for t in TIERS}})
    by_diff = pd.DataFrame(by_diff)

    # does the fuzzy router pick the tier the oracle would?
    agree = pd.crosstab(pd.Series(conds["fuzzy_router"], name="fuzzy"),
                        pd.Series(conds["oracle"], name="oracle"))
    solvable = d.groupby("prompt_id").correct.any()
    agreement = {
        "fuzzy_equals_oracle_all": np.mean([conds["fuzzy_router"][p] == conds["oracle"][p]
                                            for p in prompts]),
        "fuzzy_equals_oracle_solvable": np.mean([conds["fuzzy_router"][p] == conds["oracle"][p]
                                                 for p in prompts if solvable[p]]),
        "n_solvable_by_any_tier": int(solvable.sum()),
    }

    tab.round(4).to_csv(OUT / f"routing_run{run}_conditions_ci.csv")
    paired.round(4).to_csv(OUT / f"routing_run{run}_paired.csv", index=False)
    by_diff.round(4).to_csv(OUT / f"routing_run{run}_by_difficulty.csv", index=False)
    agree.to_csv(OUT / f"routing_run{run}_fuzzy_vs_oracle.csv")
    pd.Series(agreement).to_csv(OUT / f"routing_run{run}_agreement.csv", header=["value"])
    print(f"  wrote paper/analysis/routing_run{run}_*.csv")
    return tab, paired, by_diff, agree, agreement


def fig_routing(run, tab):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    order = ["static_4bit", "static_8bit", "static_16bit", "fuzzy_router",
             "random_matched", "threshold_router", "oracle"]
    nice = {"static_4bit": "Static 4-bit", "static_8bit": "Static 8-bit",
            "static_16bit": "Static 16-bit", "fuzzy_router": "Fuzzy router",
            "random_matched": "Random (matched mix)", "threshold_router": "Threshold router",
            "oracle": "Oracle"}
    for c in order:
        r = tab.loc[c]
        routed = c in ("fuzzy_router", "random_matched", "threshold_router")
        color = ORANGE if c == "fuzzy_router" else (BLUE if not routed else AQUA)
        ax.scatter(r.j_per_request, r.accuracy, s=64, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
        if not np.isnan(r.get("acc_ci_lo", np.nan)):
            ax.errorbar(r.j_per_request, r.accuracy,
                        yerr=[[r.accuracy - r.acc_ci_lo], [r.acc_ci_hi - r.accuracy]],
                        xerr=[[r.j_per_request - r.j_ci_lo], [r.j_ci_hi - r.j_per_request]],
                        fmt="none", ecolor=color, elinewidth=1, alpha=0.6, zorder=2)
        offset = {"fuzzy_router": (-8, -12), "random_matched": (7, 5)}.get(c, (7, 4))
        ax.annotate(nice[c], (r.j_per_request, r.accuracy), xytext=offset,
                    ha="right" if offset[0] < 0 else "left",
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("Energy per request (J)")
    ax.set_ylabel("Accuracy (reference-match proxy)")
    ax.set_title(f"Routing conditions, run {run} (95% bootstrap CIs)",
                 loc="left", fontsize=9.5, color=INK)
    save(fig, f"fig3_routing_run{run}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disruption-runs", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--clean-runs", type=int, nargs="+", default=[5])
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    fig_disruption(args.disruption_runs)
    tiers = fig_tier_bars(args.disruption_runs)
    print(tiers.round(2).to_string(index=False))
    for run in args.clean_runs:
        tab, paired, by_diff, agree, agreement = routing_study(run)
        fig_routing(run, tab)
        print(f"\n=== run {run} conditions ===\n{tab.round(3).to_string()}")
        print(f"\n=== paired ===\n{paired.round(3).to_string(index=False)}")
        print(f"\n=== by difficulty ===\n{by_diff.round(3).to_string(index=False)}")
        print(f"\n=== fuzzy vs oracle tiers ===\n{agree}")
        print(f"\n{agreement}")


if __name__ == "__main__":
    main()
