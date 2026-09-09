"""
calibrate_breakpoints.py — derive fuzzy membership breakpoints from data
=======================================================================

Why this exists
---------------
`config.yaml`'s `fuzzy_controller.*_breakpoints` decide where each feature's
LOW / MEDIUM / HIGH bands begin. Until 2026-09-09 they were hand-picked
round numbers, and nothing checked them against the distribution of prompts
the router actually sees. They were badly wrong in a way that silently
disabled the router:

    feature          HIGH band began at   % of real prompts reaching it
    flesch_kincaid   0.667 (normalized)               10%
    entropy          0.471                            10%
    token_length     0.500                             0%   <-- observed max 0.13
    syntax_depth     0.417                             0%   <-- observed max 0.33

With no prompt ever entering a HIGH band, every rule with a HIGH antecedent
fired at ~zero strength and the defuzzified complexity score collapsed onto
~50 for nearly everything — 8-bit for 22 of 30 eval prompts, 16-bit for 1.

The fix is to stop guessing. This script reads the eval prompt set, scores
every prompt with the real complexity sensor, and puts the two breakpoints
at the **empirical terciles** of each feature. That guarantees each band is
populated by construction, on whatever prompt distribution is actually in
use.

Terciles use only the feature distribution, never the difficulty labels, so
this is a calibration step and not label fitting — the routing decision
still comes entirely from the rule base.

Output units match config.yaml: raw units for flesch_kincaid (grade),
entropy (bits) and syntax_depth (parse depth), already-normalized 0-1 for
token_length. See complexity_scorer's *_RANGE constants.

Usage
-----
    cd backend/src/green_weight        # bare same-directory imports
    python ../../../training/scripts/calibrate_breakpoints.py
    python ../../../training/scripts/calibrate_breakpoints.py --prompts data/eval_prompts.jsonl

Prints a config.yaml-ready block. It does NOT edit config.yaml — applying
the values is a deliberate, reviewable step, because breakpoints are a
paper-relevant calibration choice.

IMPORTANT: re-run this against the authoritative 500-prompt set
(`prepare_eval_dataset.py`, seed 42) on the cluster before any measurement
session. The laptop copy of data/eval_prompts.jsonl is a stale 30-prompt
sample (see NEW.md Phase 1), so numbers derived from it are provisional.
"""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from router import complexity_scorer as cs  # noqa: E402

# Raw-unit accessor per feature: how to recover the pre-normalization value
# from a prompt, so the emitted breakpoints match config.yaml's units.
RAW_EXTRACTORS = {
    "flesch_kincaid": lambda p: _fk_grade(p),
    "entropy": lambda p: cs.shannon_entropy(p),
    "syntax_depth": lambda p: float(cs.get_parse_depth(p)),
    # token_length's breakpoints are stored already-normalized (0-1).
    "token_length": lambda p: cs.normalize_to_01(len(p) / 4.0, *cs.TOKEN_LENGTH_RANGE),
}

ROUNDING = {
    "flesch_kincaid": 1,
    "entropy": 2,
    "syntax_depth": 0,
    "token_length": 3,
}


def _fk_grade(prompt: str) -> float:
    import textstat
    return float(textstat.flesch_kincaid_grade(prompt))


def load_prompts(path: Path):
    prompts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("prompt") or row.get("text") or ""
            if text:
                prompts.append(text)
    return prompts


def terciles(values):
    """Lower and upper tercile — the two breakpoints, in ascending order."""
    ordered = sorted(values)
    lo = statistics.quantiles(ordered, n=3)[0]
    hi = statistics.quantiles(ordered, n=3)[1]
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="data/eval_prompts.jsonl",
                    help="JSONL eval set (default: data/eval_prompts.jsonl)")
    args = ap.parse_args()

    logging.disable(logging.INFO)
    path = Path(args.prompts)
    if not path.exists():
        sys.exit(f"error: prompt file not found: {path.resolve()}")

    prompts = load_prompts(path)
    if len(prompts) < 3:
        sys.exit(f"error: need >= 3 prompts to compute terciles, got {len(prompts)}")

    print(f"# Calibrated from {path} (n={len(prompts)} prompts), empirical terciles.")
    if len(prompts) < 100:
        print(f"# WARNING: n={len(prompts)} is small. Re-run on the authoritative")
        print("# 500-prompt set before trusting these for a measurement session.")
    print("  fuzzy_controller:")

    for feature, extract in RAW_EXTRACTORS.items():
        raw = [extract(p) for p in prompts]
        lo, hi = terciles(raw)
        nd = ROUNDING[feature]
        lo_r = round(lo, nd) if nd else int(round(lo))
        hi_r = round(hi, nd) if nd else int(round(hi))
        if lo_r == hi_r:                      # degenerate: no width to MEDIUM
            step = 10 ** -nd if nd else 1
            hi_r = lo_r + step
        span = f"observed {min(raw):.2f}-{max(raw):.2f}"
        print(f"    {feature}_breakpoints: [{lo_r}, {hi_r}]  # terciles, {span}")

    print("\n# Band occupancy at these breakpoints (share of prompts per band):")
    for feature, extract in RAW_EXTRACTORS.items():
        raw = [extract(p) for p in prompts]
        lo, hi = terciles(raw)
        n = len(raw)
        low = sum(v < lo for v in raw) / n
        med = sum(lo <= v <= hi for v in raw) / n
        high = sum(v > hi for v in raw) / n
        print(f"#   {feature:16s} low={low:.0%}  medium={med:.0%}  high={high:.0%}")


if __name__ == "__main__":
    main()
