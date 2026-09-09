"""
router_quality_report.py — classification quality of the routing decision
=========================================================================

Computes per-tier precision / recall / F1 for the router's tier choice and
writes them to a JSON artifact the dashboard reads
(`results/router_quality.json`). No GPU, no model loading — this measures
the *routing decision* only.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
The reference labels are the eval set's `difficulty_label`, which is
**dataset provenance**, not measured quantization sensitivity:

    easy   = TriviaQA          -> expected tier 4bit
    medium = Alpaca            -> expected tier 8bit
    hard   = GSM8K/CodeAlpaca  -> expected tier 16bit

That mapping is an assumption. It says "a harder prompt should get more
precision", which is the system's design intent, but it is NOT evidence
that a 4-bit model actually answers TriviaQA correctly or that fp16 is
actually required for GSM8K. Only Session 2 (per-tier benchmark accuracy)
and Session 4 (per-prompt correctness at every tier) can establish that.

So: these numbers are a legitimate measure of whether the sensor
discriminates prompt difficulty at all, and a legitimate regression guard.
They are NOT a result about energy or model accuracy, they do not belong
in `paper/results.md`, and anything displaying them must label them as a
routing-decision proxy. The emitted JSON carries
`"provenance": "DERIVED"` and a `"caveat"` string so a consumer cannot
render them as a measured result by accident.

Usage
-----
    cd backend/src/green_weight          # bare same-directory imports
    python ../../../training/scripts/router_quality_report.py

Deterministic: same eval set + same config.yaml => same numbers.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from router import complexity_scorer as cs          # noqa: E402
from router.fuzzy_controller import FuzzyController  # noqa: E402
from router.routellm_bridge import RouteLLMBridge    # noqa: E402

# difficulty label -> the tier the system is designed to pick for it
EXPECTED_TIER = {"easy": "4bit", "medium": "8bit", "hard": "16bit"}
TIERS = ["4bit", "8bit", "16bit"]
TIER_INDEX = {t: i for i, t in enumerate(TIERS)}
DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}


def spearman(xs, ys):
    """Spearman rank correlation, average ranks for ties. Implemented here
    rather than pulled from scipy so this script has no dependency the
    backend does not already have."""
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="data/eval_prompts.jsonl")
    ap.add_argument("--out", default="results/router_quality.json")
    args = ap.parse_args()

    logging.disable(logging.INFO)
    prompt_path = Path(args.prompts)
    if not prompt_path.exists():
        sys.exit(f"error: prompt file not found: {prompt_path.resolve()}")

    rows = []
    with open(prompt_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get("prompt") and r.get("difficulty_label")]
    if not rows:
        sys.exit("error: no labelled prompts found — cannot score the router")

    controller = FuzzyController()
    bridge = RouteLLMBridge()

    scores, predicted, truth = [], [], []
    for r in rows:
        features = cs.score(r["prompt"])
        _fuzzy_tier, win_prob = controller.route(features)
        predicted.append(bridge.decide(r["prompt"], win_prob))
        scores.append(win_prob * 100.0)
        truth.append(EXPECTED_TIER[r["difficulty_label"]])

    n = len(rows)

    per_tier = {}
    for tier in TIERS:
        tp = sum(p == tier and t == tier for p, t in zip(predicted, truth))
        fp = sum(p == tier and t != tier for p, t in zip(predicted, truth))
        fn = sum(p != tier and t == tier for p, t in zip(predicted, truth))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_tier[tier] = {
            "support": sum(t == tier for t in truth),
            "predicted": sum(p == tier for p in predicted),
            "true_positives": tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    correct = sum(p == t for p, t in zip(predicted, truth))
    within_one = sum(abs(TIER_INDEX[p] - TIER_INDEX[t]) <= 1
                     for p, t in zip(predicted, truth))
    macro_f1 = sum(v["f1"] for v in per_tier.values()) / len(TIERS)
    macro_precision = sum(v["precision"] for v in per_tier.values()) / len(TIERS)
    macro_recall = sum(v["recall"] for v in per_tier.values()) / len(TIERS)

    confusion = {}
    for label in ("easy", "medium", "hard"):
        got = Counter(p for p, r in zip(predicted, rows)
                      if r["difficulty_label"] == label)
        confusion[label] = {t: got.get(t, 0) for t in TIERS}

    by_difficulty = {}
    for label in ("easy", "medium", "hard"):
        vals = [s for s, r in zip(scores, rows) if r["difficulty_label"] == label]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        by_difficulty[label] = {
            "n": len(vals),
            "mean_complexity": round(mean, 2),
            "sd": round(var ** 0.5, 2),
        }

    rho = spearman(scores, [DIFFICULTY_RANK[r["difficulty_label"]] for r in rows])
    neutral = sum(abs(s - 50.0) < 1e-9 for s in scores)
    mix = Counter(predicted)

    report = {
        "provenance": "DERIVED",
        "caveat": (
            "Reference labels are dataset provenance (TriviaQA=easy, "
            "Alpaca=medium, GSM8K/CodeAlpaca=hard), not measured "
            "quantization sensitivity. These figures describe the routing "
            "DECISION only — they are not model accuracy and not an energy "
            "result. Establishing which tier can actually answer each "
            "prompt requires Session 2 and Session 4."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_set": str(prompt_path),
        "n_prompts": n,
        "overall": {
            "exact_agreement": round(correct / n, 4),
            "within_one_tier": round(within_one / n, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "spearman_rho": round(rho, 4),
            "neutral_score_share": round(neutral / n, 4),
        },
        "per_tier": per_tier,
        "tier_mix": {t: round(mix.get(t, 0) / n, 4) for t in TIERS},
        "confusion": confusion,
        "complexity_by_difficulty": by_difficulty,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"n={n} prompts from {prompt_path}")
    print(f"exact agreement {report['overall']['exact_agreement']:.1%}   "
          f"macro-P {macro_precision:.3f}  macro-R {macro_recall:.3f}  "
          f"macro-F1 {macro_f1:.3f}   rho {rho:.3f}")
    for tier in TIERS:
        v = per_tier[tier]
        print(f"  {tier:6s} precision={v['precision']:.3f} "
              f"recall={v['recall']:.3f} f1={v['f1']:.3f} "
              f"(support {v['support']}, predicted {v['predicted']})")
    print(f"written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
