"""
Verify Results — automated credibility gate
===========================================

Runs integrity and plausibility checks over the CSVs downloaded from the
Kaggle sessions and writes `results_validation.md` (referenced by
CREDIBILITY_REPORT.md). A result may only be quoted in the paper if this
script reports PASS (or WARN with the warning acknowledged in the report).

Checks are deliberately conservative: they catch the failure modes this
project has already had once — empty logs, zero measurements, modeled
numbers masquerading as measured ones.

Usage:
    python scripts/verify_results.py [--results-dir green_weight/results]
"""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

# Plausibility window for a 1B model on a T4 (idle-subtraction not applied):
# T4 board power 25-70 W at 5-60 tokens/s => roughly 0.3-15 J/token.
JPT_MIN, JPT_MAX = 0.05, 50.0
MIN_N_PER_TIER = 30          # fewer than this is anecdote, not measurement
CI_WARN_FRACTION = 0.25      # CI half-width > 25% of mean => noisy

findings = []                # (level, area, message)


def check(level, area, ok, msg_pass, msg_fail):
    if ok:
        findings.append(("PASS", area, msg_pass))
    else:
        findings.append((level, area, msg_fail))
    return ok


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------- Session 1
def verify_energy(results_dir):
    area = "energy (Session 1)"
    per_inf = results_dir / "energy_logs" / "energy_per_inference.csv"
    hw_file = results_dir / "energy_logs" / "hardware_info.json"

    if not check("FAIL", area, per_inf.exists(),
                 f"{per_inf.name} present",
                 f"{per_inf} missing — no verified energy data exists"):
        return None

    rows = read_csv(per_inf)
    check("FAIL", area, len(rows) > 0, f"{len(rows)} measurement rows",
          "energy file is EMPTY (header only) — the exact failure of May 2026")

    by_tier = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)

    for tier in ["4bit", "8bit", "16bit"]:
        trs = by_tier.get(tier, [])
        check("FAIL", area, len(trs) >= MIN_N_PER_TIER,
              f"{tier}: n={len(trs)} (>= {MIN_N_PER_TIER})",
              f"{tier}: only n={len(trs)} measurements (< {MIN_N_PER_TIER})")
        if not trs:
            continue
        energies = [float(r["energy_j"]) for r in trs]
        tokens = [int(r["tokens_out"]) for r in trs]
        jpt = [float(r["j_per_token"]) for r in trs]

        check("FAIL", area, all(e > 0 for e in energies),
              f"{tier}: all energy readings positive",
              f"{tier}: contains zero/negative energy readings — meter failure")
        check("FAIL", area, all(t > 0 for t in tokens),
              f"{tier}: all token counts positive",
              f"{tier}: zero token counts present")
        mean_jpt = statistics.mean(jpt)
        check("FAIL", area, JPT_MIN <= mean_jpt <= JPT_MAX,
              f"{tier}: mean {mean_jpt:.3f} J/token within plausibility window "
              f"[{JPT_MIN}, {JPT_MAX}]",
              f"{tier}: mean {mean_jpt:.3f} J/token OUTSIDE plausible T4 range "
              f"[{JPT_MIN}, {JPT_MAX}] — check meter units")
        if len(jpt) >= 2:
            ci = 1.96 * statistics.stdev(jpt) / (len(jpt) ** 0.5)
            check("WARN", area, ci <= CI_WARN_FRACTION * mean_jpt,
                  f"{tier}: 95% CI ±{ci:.3f} is tight ({100*ci/mean_jpt:.0f}% of mean)",
                  f"{tier}: 95% CI ±{ci:.3f} is {100*ci/mean_jpt:.0f}% of mean — "
                  f"noisy; add runs or prompts")

    runs = {r.get("run", "1") for r in rows}
    check("WARN", area, len(runs) >= 3,
          f"{len(runs)} repeated runs (>= 3)",
          f"only {len(runs)} run(s) — repeat 3x for defensible CIs")

    if check("WARN", area, hw_file.exists(),
             "hardware_info.json present",
             "hardware_info.json missing — hardware provenance unrecorded"):
        hw = json.loads(hw_file.read_text())
        check("WARN", area, bool(hw.get("energy_counter")),
              f"NVML hardware energy counter used ({hw.get('gpu', '?')})",
              "power-sampling fallback was used, not the hardware counter — "
              "note this in threats-to-validity")
        return hw
    return None


# ---------------------------------------------------------- Session 2
def verify_accuracy(results_dir):
    area = "accuracy (Session 2)"
    path = results_dir / "accuracy_logs" / "accuracy_summary.csv"
    if not check("FAIL", area, path.exists(),
                 f"{path.name} present",
                 f"{path} missing — per-tier accuracy unverified"):
        return

    rows = [r for r in read_csv(path)
            if r["metric"].startswith("acc") and "stderr" not in r["metric"]]
    check("FAIL", area, len(rows) > 0, f"{len(rows)} accuracy entries",
          "no accuracy metrics in file")

    vals = [float(r["value"]) for r in rows]
    check("FAIL", area, all(0.0 <= v <= 1.0 for v in vals),
          "all accuracy values in [0, 1]", "accuracy values outside [0, 1]")

    # sanity: fp16 should not be beaten by 4-bit across the board
    for task in {r["task"] for r in rows}:
        base = {r["tier"]: float(r["value"]) for r in rows
                if r["task"] == task and r["variant"] == "base"}
        if "16bit" in base and "4bit" in base:
            check("WARN", area, base["16bit"] >= base["4bit"] - 0.02,
                  f"{task}: fp16 >= 4-bit as expected "
                  f"({base['16bit']:.3f} vs {base['4bit']:.3f})",
                  f"{task}: 4-bit ({base['4bit']:.3f}) beats fp16 "
                  f"({base['16bit']:.3f}) by >2pts — verify eval setup")


# ---------------------------------------------------------- Session 4
def verify_routing(results_dir):
    area = "routing (Session 4)"
    summary = results_dir / "routing_logs" / "routing_conditions_summary.csv"
    per_prompt = results_dir / "routing_logs" / "routing_per_prompt.csv"
    if not check("FAIL", area, summary.exists(),
                 f"{summary.name} present",
                 f"{summary} missing — main experiment not run"):
        return

    rows = {r["condition"]: r for r in read_csv(summary)}
    expected = ["static_4bit", "static_8bit", "static_16bit", "fuzzy_router",
                "random_matched", "threshold_router", "oracle", "oracle_cascade"]
    for cond in expected:
        check("FAIL", area, cond in rows, f"condition {cond} present",
              f"condition {cond} MISSING from summary")
    if not all(c in rows for c in expected):
        return

    acc = {c: float(rows[c]["accuracy"]) for c in expected}
    energy = {c: float(rows[c]["j_per_request"]) for c in expected}

    check("FAIL", area, all(v > 0 for v in energy.values()),
          "all conditions have positive measured energy",
          "zero-energy condition present — measurement failure")
    check("WARN", area, acc["oracle"] >= max(acc[c] for c in expected if c != "oracle") - 1e-9,
          "oracle accuracy is the maximum (as it must be by construction)",
          "oracle is NOT the accuracy maximum — derivation bug")
    check("WARN", area, energy["static_4bit"] <= energy["static_16bit"],
          "4-bit uses less energy/request than fp16",
          "4-bit uses MORE energy than fp16 — the §2 risk materialized; "
          "switch low tiers to GPTQ/AWQ kernels or reframe (see RESEARCH_PLAN.md)")
    check("WARN", area, acc["fuzzy_router"] >= acc["random_matched"],
          "fuzzy router beats random with matched tier mix",
          "fuzzy router does NOT beat matched random — routing adds no "
          "intelligence; report honestly and investigate the sensor")

    if check("WARN", area, per_prompt.exists(),
             "per-prompt raw data present",
             "routing_per_prompt.csv missing — raw artifact needed for release"):
        praws = read_csv(per_prompt)
        n_prompts = len({r["prompt_id"] for r in praws})
        check("FAIL", area, len(praws) == 3 * n_prompts,
              f"{n_prompts} prompts x 3 tiers = {len(praws)} rows, complete grid",
              f"incomplete measurement grid: {len(praws)} rows for "
              f"{n_prompts} prompts")
        check("WARN", area, n_prompts >= 300,
              f"n={n_prompts} prompts (>= 300)",
              f"only n={n_prompts} prompts — plan calls for >= 300")


# ------------------------------------------------------------- report
def write_report(results_dir, hw):
    out = results_dir / "results_validation.md"
    n_pass = sum(1 for f in findings if f[0] == "PASS")
    n_warn = sum(1 for f in findings if f[0] == "WARN")
    n_fail = sum(1 for f in findings if f[0] == "FAIL")
    verdict = ("NOT PUBLISHABLE — failures present" if n_fail
               else ("PUBLISHABLE WITH CAVEATS — acknowledge warnings" if n_warn
                     else "ALL CHECKS PASSED"))

    lines = [
        "# Automated Results Validation",
        "",
        f"Generated by `scripts/verify_results.py` on "
        f"{datetime.now(timezone.utc).isoformat()} (UTC).",
        "",
        f"**Verdict: {verdict}**  (PASS: {n_pass}, WARN: {n_warn}, FAIL: {n_fail})",
        "",
    ]
    if hw:
        lines += [f"Hardware: {hw.get('gpu', '?')}, driver {hw.get('driver', '?')}, "
                  f"NVML energy counter: {hw.get('energy_counter', '?')}", ""]
    lines += ["| Level | Area | Finding |", "|---|---|---|"]
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    for level, area, msg in sorted(findings, key=lambda f: order[f[0]]):
        lines.append(f"| {level} | {area} | {msg} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[OK] wrote {out}")
    print(f"VERDICT: {verdict}")


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="green_weight/results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    hw = verify_energy(results_dir)
    verify_accuracy(results_dir)
    verify_routing(results_dir)

    for level, area, msg in findings:
        print(f"  [{level}] {area}: {msg}")
    write_report(results_dir, hw)


if __name__ == "__main__":
    main()
