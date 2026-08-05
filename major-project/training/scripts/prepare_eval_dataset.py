"""
Prepare Evaluation Dataset
==========================

Builds the stratified routing-evaluation set (default 500 prompts:
200 easy / 150 medium / 150 hard) from public datasets:

  easy   - TriviaQA (short factual questions, unambiguous answers)
  medium - Alpaca instructions without input (explain / summarize / describe)
  hard   - GSM8K math word problems + CodeAlpaca coding tasks

Guarantees for publication credibility:
  - No prompt appears twice (exact-match dedup across ALL difficulty pools).
  - No repetition-padding: if a pool is too small the script FAILS loudly
    rather than silently duplicating prompts.
  - Deterministic under --seed.

Usage (needs `pip install datasets`; run locally or in a Kaggle cell):
    python scripts/prepare_eval_dataset.py --num-prompts 500 \
        --output green_weight/data/eval_prompts.jsonl
"""

import argparse
import json
import random
from pathlib import Path

# difficulty -> (fraction of total, list of (dataset_id, config, extractor))
SPLIT_FRACTIONS = {"easy": 0.4, "medium": 0.3, "hard": 0.3}


def _from_triviaqa(row):
    q = (row.get("question") or "").strip()
    ans = row.get("answer") or {}
    a = (ans.get("value") or "").strip() if isinstance(ans, dict) else ""
    if q and a and len(q.split()) <= 40:
        return {"prompt": q, "reference_answer": a}
    return None


def _from_alpaca(row):
    # Only instructions with no extra input context, mid-length
    if (row.get("input") or "").strip():
        return None
    q = (row.get("instruction") or "").strip()
    a = (row.get("output") or "").strip()
    if q and a and 5 <= len(q.split()) <= 60:
        return {"prompt": q, "reference_answer": a}
    return None


def _from_gsm8k(row):
    q = (row.get("question") or "").strip()
    a = (row.get("answer") or "").strip()
    if q and a:
        # keep the final numeric answer after '####' as the reference
        final = a.split("####")[-1].strip() if "####" in a else a
        return {"prompt": q, "reference_answer": final, "full_solution": a}
    return None


def _from_codealpaca(row):
    if (row.get("input") or "").strip():
        return None
    q = (row.get("instruction") or "").strip()
    a = (row.get("output") or "").strip()
    if q and a:
        return {"prompt": q, "reference_answer": a}
    return None


SOURCES = {
    "easy": [("mandarjoshi/trivia_qa", "rc.nocontext", "validation", _from_triviaqa)],
    "medium": [("tatsu-lab/alpaca", None, "train", _from_alpaca)],
    "hard": [
        ("openai/gsm8k", "main", "test", _from_gsm8k),
        ("sahil2801/CodeAlpaca-20k", None, "train", _from_codealpaca),
    ],
}


def collect(difficulty: str, needed: int, seen: set, rng: random.Random) -> list:
    """Stream rows from each source for a difficulty until `needed` unique prompts."""
    from datasets import load_dataset

    specs = SOURCES[difficulty]
    per_source = needed // len(specs)
    quotas = [per_source] * len(specs)
    quotas[-1] += needed - sum(quotas)

    out = []
    for (ds_id, config, split, extract), quota in zip(specs, quotas):
        print(f"  [{difficulty}] {ds_id} -> want {quota}")
        ds = load_dataset(ds_id, config, split=split, streaming=True)
        # take a generous buffer, then sample deterministically
        buffer = []
        for row in ds:
            entry = extract(row)
            if entry and entry["prompt"] not in seen:
                seen.add(entry["prompt"])
                entry["difficulty_label"] = difficulty
                entry["source"] = ds_id
                buffer.append(entry)
            if len(buffer) >= quota * 3:
                break
        if len(buffer) < quota:
            raise RuntimeError(
                f"{ds_id} yielded only {len(buffer)} unique prompts, need {quota}. "
                f"Refusing to pad with duplicates."
            )
        out.extend(rng.sample(buffer, quota))
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare stratified evaluation dataset")
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--output", type=str,
                        default="green_weight/data/eval_prompts.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    counts = {d: int(args.num_prompts * f) for d, f in SPLIT_FRACTIONS.items()}
    counts["easy"] += args.num_prompts - sum(counts.values())  # rounding remainder

    print(f"Target: {counts} (total {args.num_prompts}), seed={args.seed}")

    seen: set = set()
    prompts = []
    for difficulty in ["easy", "medium", "hard"]:
        prompts.extend(collect(difficulty, counts[difficulty], seen, rng))

    rng.shuffle(prompts)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    assert len({p["prompt"] for p in prompts}) == len(prompts), "duplicate prompts!"
    print(f"[OK] Wrote {len(prompts)} unique prompts to {output_path}")
    for d in ["easy", "medium", "hard"]:
        n = sum(1 for p in prompts if p["difficulty_label"] == d)
        print(f"     {d}: {n}")


if __name__ == "__main__":
    main()
