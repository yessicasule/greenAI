"""
Prepare Evaluation Dataset
==========================

Generates eval_prompts.jsonl from a HuggingFace dataset (e.g., OpenOrca).

Usage:
    python scripts/prepare_eval_dataset.py --num-prompts 500 \
        --output green_weight/data/eval_prompts.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path


def generate_synthetic_prompts(n: int, seed: int = 42) -> list:
    """Generate a balanced synthetic dataset when HuggingFace is unavailable."""
    random.seed(seed)

    easy = [
        ("What is the capital of France?", "Paris"),
        ("What is 2 + 2?", "4"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("How many days are in a week?", "7"),
        ("What color is the sky on a clear day?", "Blue"),
        ("What is the boiling point of water in Celsius?", "100 degrees Celsius"),
        ("Name the primary colors.", "Red, blue, and yellow"),
        ("How many continents are there on Earth?", "7"),
        ("What is the speed of light?", "Approximately 299,792,458 meters per second"),
    ]

    medium = [
        ("Summarize the plot of Harry Potter and the Philosopher's Stone.",
         "A young wizard attends Hogwarts and defeats a dark wizard trying to steal the Philosopher's Stone."),
        ("Explain the concept of photosynthesis.",
         "The process by which plants convert sunlight into glucose using carbon dioxide and water."),
        ("What are the main causes of climate change?",
         "Greenhouse gas emissions from fossil fuels, deforestation, and industrial activities."),
        ("Describe the water cycle.",
         "The continuous movement of water through evaporation, condensation, precipitation, and collection."),
        ("What is the difference between RAM and ROM?",
         "RAM is volatile temporary memory; ROM is non-volatile permanent memory."),
        ("Explain the theory of natural selection.",
         "Organisms with traits better suited to their environment survive and reproduce more successfully."),
        ("What is machine learning?",
         "A subset of AI where systems learn from data to make predictions or decisions."),
        ("Describe the difference between a virus and a bacterium.",
         "Viruses need a host to replicate; bacteria are self-replicating single-celled organisms."),
        ("What is the significance of the Magna Carta?",
         "A 1215 charter limiting the king's power and establishing rule of law in England."),
        ("Explain supply and demand.",
         "Price is determined by the balance between how much is available and how much is wanted."),
    ]

    hard = [
        ("Prove that the square root of 2 is irrational.",
         "Assume sqrt(2) = p/q in lowest terms. Then 2q^2 = p^2, so p is even, leading to contradiction."),
        ("Explain quantum entanglement and its implications for quantum computing.",
         "Entangled qubits share quantum states, enabling correlated computations across distances."),
        ("Design an algorithm to find the longest common subsequence of two strings.",
         "Use dynamic programming with dp[i][j] tracking LCS of s1[:i] and s2[:j]. O(mn) time."),
        ("Discuss the philosophical implications of Godel's incompleteness theorems.",
         "No consistent formal system can prove all truths about arithmetic; some truths are unprovable."),
        ("Write a Python function that implements quicksort and analyze its time complexity.",
         "Recursive pivot-based sort: O(n log n) average, O(n^2) worst case. Partition in O(n)."),
        ("Explain the Fourier transform and its applications.",
         "Decomposes a signal into frequency components. Used in signal processing, image compression, physics."),
        ("What is the CAP theorem in distributed systems?",
         "A distributed system can guarantee at most two of: Consistency, Availability, Partition tolerance."),
        ("Describe the P vs NP problem.",
         "Whether every problem whose solution can be verified in polynomial time can also be solved in polynomial time."),
        ("Explain Bayes' theorem with an example.",
         "P(A|B) = P(B|A)*P(A)/P(B). Example: disease testing with false positive rates."),
        ("Write a recursive algorithm for the Tower of Hanoi and derive its time complexity.",
         "Move n-1 disks to auxiliary, move largest to target, move n-1 back. T(n)=2T(n-1)+1, so O(2^n)."),
    ]

    easy_count = n // 3
    medium_count = n // 3
    hard_count = n - easy_count - medium_count

    def sample_with_repeat(pool, count):
        if len(pool) >= count:
            return random.sample(pool, count)
        reps = count // len(pool) + 1
        return (pool * reps)[:count]

    prompts = []
    for prompt, answer in sample_with_repeat(easy, easy_count):
        prompts.append({"prompt": prompt, "reference_answer": answer, "difficulty_label": "easy"})
    for prompt, answer in sample_with_repeat(medium, medium_count):
        prompts.append({"prompt": prompt, "reference_answer": answer, "difficulty_label": "medium"})
    for prompt, answer in sample_with_repeat(hard, hard_count):
        prompts.append({"prompt": prompt, "reference_answer": answer, "difficulty_label": "hard"})

    random.shuffle(prompts)
    return prompts


_PROMPT_KEYS = ("question", "instruction", "prompt")
_ANSWER_KEYS = ("response", "output", "answer")


def _extract_prompt_entry(row: dict) -> dict | None:
    prompt = next((row[k] for k in _PROMPT_KEYS if row.get(k)), None)
    if not prompt:
        return None
    answer = next((row[k] for k in _ANSWER_KEYS if row.get(k)), "")
    return {
        "prompt": prompt,
        "reference_answer": answer,
        "difficulty_label": row.get("difficulty", "medium"),
    }


def load_from_hf_dataset(dataset_name: str, n: int) -> list:
    """Load prompts from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
        print(f"Loading {n} prompts from HuggingFace dataset: {dataset_name}...")
        ds = load_dataset(dataset_name, split="train", streaming=True)
        prompts = []
        skipped = 0
        for i, row in enumerate(ds):
            if len(prompts) >= n:
                break
            entry = _extract_prompt_entry(row)
            if entry:
                prompts.append(entry)
            else:
                skipped += 1
        if skipped:
            print(f"[WARN] Skipped {skipped} rows with missing prompt field")
        return prompts
    except Exception as e:
        print(f"[WARN] Failed to load from HuggingFace ({e}). Falling back to synthetic prompts.")
        return []


def main():
    parser = argparse.ArgumentParser(description="Prepare evaluation dataset")
    parser.add_argument("--num-prompts", type=int, default=500,
                        help="Number of prompts to generate (default: 500)")
    parser.add_argument("--output", type=str,
                        default="green_weight/data/eval_prompts.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name (e.g., 'Open-Orca/OpenOrca'). "
                             "Falls back to synthetic if omitted or unavailable.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        prompts = load_from_hf_dataset(args.dataset, args.num_prompts)
    else:
        prompts = []

    if not prompts:
        print(f"Generating {args.num_prompts} synthetic prompts...")
        prompts = generate_synthetic_prompts(args.num_prompts, seed=args.seed)

    with open(output_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    easy = sum(1 for p in prompts if p["difficulty_label"] == "easy")
    medium = sum(1 for p in prompts if p["difficulty_label"] == "medium")
    hard = sum(1 for p in prompts if p["difficulty_label"] == "hard")

    print(f"[OK] Wrote {len(prompts)} prompts to {output_path}")
    print(f"     easy={easy}, medium={medium}, hard={hard}")


if __name__ == "__main__":
    main()
