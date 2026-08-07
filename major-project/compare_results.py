"""
Compare Results
===============

Compares accuracy and energy results between two pipeline runs (baseline vs ablation).

Usage (run from major-project/):
    python compare_results.py \
        --baseline backend/src/green_weight/results/accuracy_logs/accuracy_results_baseline.json \
        --ablation backend/src/green_weight/results/accuracy_logs/accuracy_results_ablation1.json
"""

import argparse
import json
from pathlib import Path


def load_results(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare(baseline: dict, ablation: dict, label_a: str, label_b: str) -> None:
    print(f"\n{'='*60}")
    print(f"COMPARISON: {label_a}  vs  {label_b}")
    print(f"{'='*60}")

    acc_a = baseline.get("accuracy_by_condition", {})
    acc_b = ablation.get("accuracy_by_condition", {})

    print("\n[Accuracy by Condition]")
    print(f"{'Condition':<20} {label_a:>12} {label_b:>12} {'Delta':>10}")
    print("-" * 56)
    conditions = sorted(set(acc_a) | set(acc_b))
    for cond in conditions:
        oa = acc_a.get(cond, {}).get("overall", 0.0)
        ob = acc_b.get(cond, {}).get("overall", 0.0)
        delta = ob - oa
        arrow = "+" if delta >= 0 else ""
        print(f"{cond:<20} {oa:>12.4f} {ob:>12.4f} {arrow}{delta:>9.4f}")

    met_a = baseline.get("routellm_metrics", {})
    met_b = ablation.get("routellm_metrics", {})

    if met_a or met_b:
        print("\n[RouteLLM Metrics]")
        print(f"{'Metric':<20} {label_a:>12} {label_b:>12} {'Delta':>10}")
        print("-" * 56)
        metrics = sorted(set(met_a) | set(met_b))
        for m in metrics:
            va = met_a.get(m, 0.0)
            vb = met_b.get(m, 0.0)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = vb - va
                arrow = "+" if delta >= 0 else ""
                print(f"{m:<20} {va:>12.4f} {vb:>12.4f} {arrow}{delta:>9.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare two pipeline result files")
    parser.add_argument("--baseline", required=True,
                        help="Path to baseline accuracy_results.json")
    parser.add_argument("--ablation", required=True,
                        help="Path to ablation accuracy_results.json")
    parser.add_argument("--label-a", default="baseline",
                        help="Label for baseline (default: baseline)")
    parser.add_argument("--label-b", default="ablation",
                        help="Label for ablation (default: ablation)")
    args = parser.parse_args()

    path_a = Path(args.baseline)
    path_b = Path(args.ablation)

    for p in (path_a, path_b):
        if not p.exists():
            print(f"[FAIL] File not found: {p}")
            raise SystemExit(1)

    baseline = load_results(path_a)
    ablation = load_results(path_b)
    compare(baseline, ablation, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
