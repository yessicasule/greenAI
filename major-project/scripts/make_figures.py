"""
Make Figures — Session 5 of the research plan (runs locally, CPU only)
=====================================================================

Reads the CSVs downloaded from Kaggle Sessions 1, 2 and 4 and produces
the paper's figures (300 dpi PNG + PDF). Generates whichever figures its
inputs exist for and skips the rest with a warning.

Expected inputs (put them in green_weight/results/):
  energy_logs/energy_per_inference.csv     (Session 1)
  energy_logs/energy_summary.csv           (Session 1)
  accuracy_logs/accuracy_summary.csv       (Session 2)
  routing_logs/routing_conditions_summary.csv  (Session 4)

Usage:
    python scripts/make_figures.py [--results-dir green_weight/results]
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- palette (validated reference palette; light surface) ----
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"                      # print surface
TIER_RAMP = {"4bit": "#86b6ef", "8bit": "#2a78d6", "16bit": "#0d366b"}  # ordinal blue
COND_COLORS = {                          # categorical, fixed assignment
    "static_4bit": TIER_RAMP["4bit"],
    "static_8bit": TIER_RAMP["8bit"],
    "static_16bit": TIER_RAMP["16bit"],
    "fuzzy_router": "#e34948",           # ours — most salient
    "threshold_router": "#4a3aa7",
    "random_matched": "#898781",
    "oracle": "#1baf7a",
    "oracle_cascade": "#1baf7a",
}
COND_LABELS = {
    "static_4bit": "Static 4-bit", "static_8bit": "Static 8-bit",
    "static_16bit": "Static fp16", "fuzzy_router": "Fuzzy router (ours)",
    "threshold_router": "Threshold router", "random_matched": "Random (matched mix)",
    "oracle": "Oracle", "oracle_cascade": "Oracle cascade",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.dpi": 300,
})


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {name}.png / .pdf")


def bootstrap_ci(values, n_boot=2000, seed=42):
    """95% bootstrap CI of the mean."""
    rng = random.Random(seed)
    if len(values) < 2:
        return 0.0
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (hi - lo) / 2


# ---------------------------------------------------- Fig 1: J/token bars
def fig_energy_per_tier(results_dir, outdir):
    path = results_dir / "energy_logs" / "energy_per_inference.csv"
    if not path.exists():
        print(f"[skip] {path} missing — run Session 1 first")
        return
    rows = read_csv(path)
    tiers = ["4bit", "8bit", "16bit"]
    data = {t: [float(r["j_per_token"]) for r in rows if r["tier"] == t] for t in tiers}

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    xs = range(len(tiers))
    means = [sum(data[t]) / len(data[t]) for t in tiers]
    cis = [bootstrap_ci(data[t]) for t in tiers]
    ax.bar(xs, means, width=0.55, color=[TIER_RAMP[t] for t in tiers],
           edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.errorbar(xs, means, yerr=cis, fmt="none", ecolor=INK,
                elinewidth=1, capsize=3, zorder=4)
    for x, m, c in zip(xs, means, cis):
        ax.annotate(f"{m:.2f}", (x, m + c), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["4-bit (NF4)", "8-bit (int8)", "fp16"])
    ax.set_ylabel("Energy (J / generated token)")
    ax.set_title("Measured GPU energy per token by precision tier",
                 fontsize=10, loc="left")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    n = min(len(v) for v in data.values())
    ax.annotate(f"mean ± 95% bootstrap CI, n≥{n} per tier, Llama-3.2-1B, T4",
                (0, -0.28), xycoords="axes fraction", fontsize=7, color=MUTED)
    save(fig, outdir, "fig1_energy_per_tier")


# ------------------------------------------------ Fig 2: Pareto scatter
def fig_pareto(results_dir, outdir):
    path = results_dir / "routing_logs" / "routing_conditions_summary.csv"
    if not path.exists():
        print(f"[skip] {path} missing — run Session 4 first")
        return
    rows = read_csv(path)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    pts = [(float(r["j_per_request"]), float(r["accuracy"]), r["condition"])
           for r in rows]
    x_span = max(p[0] for p in pts) - min(p[0] for p in pts) or 1.0
    y_span = max(p[1] for p in pts) - min(p[1] for p in pts) or 1.0

    labeled = []   # (x, y, extra_offset_pts) of labels already placed
    for x, y, cond in pts:
        color = COND_COLORS.get(cond, MUTED)
        marker = "*" if cond == "fuzzy_router" else (
            "^" if cond.startswith("oracle") else "o")
        size = 140 if cond == "fuzzy_router" else 55
        face = "none" if cond.startswith("oracle") else color
        ax.scatter(x, y, s=size, marker=marker, facecolor=face,
                   edgecolor=color, linewidth=1.4, zorder=3)
        # stack labels of coincident/near points instead of overprinting
        n_close = sum(1 for lx, ly, _ in labeled
                      if abs(x - lx) / x_span < 0.06
                      and abs(y - ly) / y_span < 0.06)
        dy = 5 - 11 * n_close
        ax.annotate(COND_LABELS.get(cond, cond), (x, y),
                    textcoords="offset points", xytext=(8, dy),
                    fontsize=7.5, color=INK)
        labeled.append((x, y, dy))
    ax.set_xlabel("Energy (J / request)  —  lower is better")
    ax.set_ylabel("Accuracy (reference-match)")
    ax.set_title("Accuracy vs measured energy per request", fontsize=10, loc="left")
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    save(fig, outdir, "fig2_pareto")


# --------------------------------------- Fig 3: tier distribution stacks
def fig_tier_distribution(results_dir, outdir):
    path = results_dir / "routing_logs" / "routing_conditions_summary.csv"
    if not path.exists():
        print(f"[skip] {path} missing — run Session 4 first")
        return
    rows = [r for r in read_csv(path)
            if r["condition"] in ("fuzzy_router", "threshold_router",
                                  "random_matched", "oracle")]
    if not rows:
        return
    tiers = ["4bit", "8bit", "16bit"]

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ys = range(len(rows))
    for yi, r in zip(ys, rows):
        left = 0.0
        for t in tiers:
            v = float(r[f"pct_{t}"]) * 100
            ax.barh(yi, v, left=left, height=0.55, color=TIER_RAMP[t],
                    edgecolor=SURFACE, linewidth=2, zorder=3)
            if v >= 8:
                ax.annotate(f"{v:.0f}%", (left + v / 2, yi), ha="center",
                            va="center", fontsize=7.5,
                            color=SURFACE if t == "16bit" else INK)
            left += v
    ax.set_yticks(list(ys))
    ax.set_yticklabels([COND_LABELS.get(r["condition"], r["condition"])
                        for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of prompts routed to tier (%)")
    ax.set_title("Precision-tier selection by routing policy", fontsize=10, loc="left")
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_RAMP[t]) for t in tiers]
    ax.legend(handles, ["4-bit", "8-bit", "fp16"], loc="lower right",
              fontsize=7.5, frameon=False, ncol=3, bbox_to_anchor=(1, -0.55))
    save(fig, outdir, "fig3_tier_distribution")


# ------------------------------------ Fig 4: per-tier accuracy, base vs QAT
def fig_accuracy_per_tier(results_dir, outdir):
    path = results_dir / "accuracy_logs" / "accuracy_summary.csv"
    if not path.exists():
        print(f"[skip] {path} missing — run Session 2 first")
        return
    rows = [r for r in read_csv(path)
            if r["metric"].startswith("acc") and "stderr" not in r["metric"]]
    if not rows:
        print("[skip] no accuracy metrics found in accuracy_summary.csv")
        return
    tasks = sorted({r["task"] for r in rows})
    tiers = ["4bit", "8bit", "16bit"]
    variants = [("base", "#2a78d6"), ("qat_adapter", "#1baf7a")]

    fig, axes = plt.subplots(1, len(tasks), figsize=(2.6 * len(tasks), 2.8),
                             sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        for vi, (variant, color) in enumerate(variants):
            vals = []
            for t in tiers:
                match = [float(r["value"]) for r in rows
                         if r["task"] == task and r["tier"] == t
                         and r["variant"] == variant]
                vals.append(match[0] if match else float("nan"))
            xs = [i + (vi - 0.5) * 0.36 for i in range(len(tiers))]
            ax.bar(xs, vals, width=0.34, color=color,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(["4-bit", "8-bit", "fp16"], fontsize=8)
        ax.set_title(task, fontsize=9, loc="left")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in variants]
    fig.legend(handles, ["Base", "QAT adapter"], fontsize=8, frameon=False,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Per-tier benchmark accuracy: base vs QAT adapters",
                 fontsize=10, x=0.02, y=1.06, ha="left")
    save(fig, outdir, "fig4_accuracy_per_tier")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="green_weight/results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    outdir = results_dir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    fig_energy_per_tier(results_dir, outdir)
    fig_pareto(results_dir, outdir)
    fig_tier_distribution(results_dir, outdir)
    fig_accuracy_per_tier(results_dir, outdir)
    print(f"\nFigures written to {outdir}")


if __name__ == "__main__":
    main()
