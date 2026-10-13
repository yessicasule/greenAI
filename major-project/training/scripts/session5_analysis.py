"""
Session 5 analysis: per-token energy of vLLM precision tiers.

Reads one or more run directories written by session5_vllm.sh
(~/session5_run<ID>_j<JOB>/<label>_bs<bs>/{per_prompt,per_batch,monitor}.csv + info.json)
and writes:
  paper/analysis/session5_summary.csv   one row per (run, config, batch size)
  paper/analysis/session5_paired.csv    paired J/token ratio vs the same model's fp16, with bootstrap CI
  paper/figures/fig_s5_jpt.pdf          J/token per config, bs=1 vs bs=8, all runs overlaid

Disruption flags (per run/config): the earlier bitsandbytes runs were wrecked by a
host-side slowdown that showed up as low GPU utilisation with normal per-token work.
A config is flagged if its median GPU utilisation during generation is below
--min-util, or if any co-tenant PID was seen on our GPU.

Usage:
  python training/scripts/session5_analysis.py results/session5_run1_j1857 [more run dirs]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "paper" / "analysis"
FIGS = ROOT / "paper" / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
ORDER = ["1b_fp16", "1b_fp8", "8b_fp16", "8b_fp8", "8b_awq4"]
RNG = np.random.default_rng(42)


def load_config_dir(d: Path):
    info = json.loads((d / "info.json").read_text())
    pp = pd.read_csv(d / "per_prompt.csv")
    pb = pd.read_csv(d / "per_batch.csv")
    mon = pd.read_csv(d / "monitor.csv") if (d / "monitor.csv").exists() else pd.DataFrame()
    return info, pp, pb, mon


def run_label(run_dir: Path):
    m = re.search(r"session5_run([^_]+)_j(\d+)", run_dir.name)
    return f"run{m.group(1)}" if m else run_dir.name


def summarise(run_dirs, min_util):
    rows, per_prompt = [], []
    for rd in run_dirs:
        run = run_label(rd)
        for d in sorted(p for p in rd.iterdir() if (p / "info.json").exists()):
            info, pp, pb, mon = load_config_dir(d)
            # GPU utilisation only while generating: drop samples with near-idle power
            # (model load, warmup gaps) by keeping the busiest 80% of samples.
            util = np.nan
            if not mon.empty:
                busy = mon[mon["power_w"] >= mon["power_w"].quantile(0.2)]
                util = float(busy["gpu_util"].median())
            cot = info.get("cotenant_pids_seen_during_run", [])
            s = info["summary"]
            rows.append({
                "run": run, "config": info["tier"], "batch_size": info["batch_size"],
                "n_prompts": info["n_prompts"],
                "j_per_token_total": s["j_per_token"],
                "j_per_token_median_batch": float(pb["j_per_token"].median()),
                "tok_per_s": s["tok_per_s"], "watts": s["watts"],
                "proxy_acc": float(pp["correct"].astype(str).str.lower().eq("true").mean()),
                "gpu_util_median": util,
                "load1_median": float(mon["load1"].median()) if not mon.empty else np.nan,
                "mhz_our_cpus_median": float(mon["mhz_our_cpus"].median()) if not mon.empty else np.nan,
                "cotenants": " ".join(map(str, cot)),
                "flag_disrupted": bool(cot) or (not np.isnan(util) and util < min_util),
                "quant_resolved": info.get("quantization_resolved"),
                "job": info.get("slurm_job_id"), "timestamp_utc": info.get("timestamp_utc"),
            })
            pp = pp.assign(run=run, config=info["tier"])
            per_prompt.append(pp)
    return pd.DataFrame(rows), pd.concat(per_prompt, ignore_index=True)


def paired(pp, n_boot=2000):
    """Ratio of total energy / total tokens vs the same model's fp16, paired on prompt_id."""
    out = []
    pp = pp.assign(model=pp["config"].str.split("_").str[0])
    for (run, model, bs), g in pp.groupby(["run", "model", "batch_size"]):
        base_cfg = f"{model}_fp16"
        base = g[g["config"] == base_cfg].set_index("prompt_id")
        if base.empty:
            continue
        for cfg in sorted(set(g["config"]) - {base_cfg}):
            q = g[g["config"] == cfg].set_index("prompt_id")
            ids = base.index.intersection(q.index).to_numpy()
            if len(ids) == 0:
                continue
            eb, tb = base.loc[ids, "energy_j"].to_numpy(), base.loc[ids, "tokens_out"].to_numpy()
            eq, tq = q.loc[ids, "energy_j"].to_numpy(), q.loc[ids, "tokens_out"].to_numpy()

            def ratio(ix):
                return (eq[ix].sum() / tq[ix].sum()) / (eb[ix].sum() / tb[ix].sum())
            point = ratio(np.arange(len(ids)))
            boots = [ratio(RNG.integers(0, len(ids), len(ids))) for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            ab = base.loc[ids, "correct"].astype(str).str.lower().eq("true").to_numpy()
            aq = q.loc[ids, "correct"].astype(str).str.lower().eq("true").to_numpy()
            out.append({"run": run, "model": model, "batch_size": bs, "config": cfg,
                        "baseline": base_cfg, "n_paired": len(ids),
                        "jpt_ratio": round(point, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "energy_saving_pct": round(100 * (1 - point), 1),
                        "proxy_acc_base": round(ab.mean(), 3), "proxy_acc_cfg": round(aq.mean(), 3),
                        "same_answer_pct": round(100 * (base.loc[ids, "response"].to_numpy()
                                                        == q.loc[ids, "response"].to_numpy()).mean(), 1)})
    return pd.DataFrame(out)


def figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clean = summary[~summary["flag_disrupted"]]
    cfgs = [c for c in ORDER if c in set(clean["config"])] + \
        sorted(set(clean["config"]) - set(ORDER))
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    x = np.arange(len(cfgs))
    for off, bs, col in [(-0.18, 1, BLUE), (0.18, 8, ORANGE)]:
        sub = clean[clean["batch_size"] == bs]
        med = [sub[sub["config"] == c]["j_per_token_total"].median() for c in cfgs]
        ax.bar(x + off, med, 0.34, color=col, label=f"batch {bs}")
        for i, c in enumerate(cfgs):
            pts = sub[sub["config"] == c]["j_per_token_total"]
            ax.scatter([x[i] + off] * len(pts), pts, s=6, color="black", zorder=3)
    ax.set_xticks(x, [c.replace("_", "\n") for c in cfgs], fontsize=7)
    ax.set_ylabel("J / output token", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_s5_jpt.pdf")
    fig.savefig(FIGS / "fig_s5_jpt.png", dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--min-util", type=float, default=50.0,
                    help="flag a config whose median GPU util while busy is below this (%%)")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    summary, pp = summarise(args.run_dirs, args.min_util)
    pairs = paired(pp[pp.set_index(["run", "config", "batch_size"]).index.isin(
        summary[~summary["flag_disrupted"]].set_index(["run", "config", "batch_size"]).index)])
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ANALYSIS / "session5_summary.csv", index=False)
    pairs.to_csv(ANALYSIS / "session5_paired.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary[["run", "config", "batch_size", "j_per_token_total", "tok_per_s", "watts",
                   "proxy_acc", "gpu_util_median", "flag_disrupted"]].to_string(index=False))
    if not pairs.empty:
        print("\nvs same-model fp16 (clean configs only):")
        print(pairs[["run", "batch_size", "config", "jpt_ratio", "ci_lo", "ci_hi",
                     "energy_saving_pct", "proxy_acc_base", "proxy_acc_cfg"]].to_string(index=False))
    if not args.no_fig:
        figure(summary)
        print(f"\nfigure: {FIGS / 'fig_s5_jpt.pdf'}")


if __name__ == "__main__":
    main()
