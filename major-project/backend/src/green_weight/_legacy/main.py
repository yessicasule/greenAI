"""Green-Weight Energy Gearbox - Main entry point."""

import argparse
import sys
from pathlib import Path

from green_weight.config import BitWidth
from green_weight.controllers.fuzzy_gearbox import FuzzyGearbox
from green_weight.core.dynamic_inference import DynamicInferenceEngine
from green_weight.core.prompt_complexity import ComplexityScorer
from green_weight.evaluation.benchmark import EnergyBenchmark, AccuracyBenchmark


def demo_prompt_analysis():
    """Demonstrate the complexity sensor and fuzzy gearbox."""
    print("=" * 60)
    print("GREEN-WEIGHT ENERGY GEARBOX DEMO")
    print("=" * 60)
    print()

    scorer = ComplexityScorer()
    gearbox = FuzzyGearbox()

    test_prompts = [
        "Hi there!",
        "What time is it?",
        "Summarize this email about the meeting.",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list using quicksort.",
        "Prove that the sum of angles in a triangle equals 180 degrees.",
        "Analyze the economic implications of AI on developing nations.",
    ]

    print("Prompt Analysis:")
    print("-" * 60)
    print(f"{'Prompt':<50} {'Score':>8} {'Gear':>6}")
    print("-" * 60)

    for prompt in test_prompts:
        complexity = scorer.calculate_complexity(prompt)
        decision = gearbox.decide(complexity)
        gear = decision.bit_width

        # Truncate long prompts
        display = prompt[:47] + "..." if len(prompt) > 50 else prompt
        print(f"{display:<50} {complexity:>7.1f} {gear:>5}-bit")

    print()


def demo_inference():
    """Demonstrate dynamic inference."""
    print("=" * 60)
    print("DYNAMIC INFERENCE DEMO")
    print("=" * 60)
    print()

    engine = DynamicInferenceEngine()

    prompts = [
        "Hello!",
        "Write a function to calculate fibonacci numbers.",
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        print("-" * 40)

        result = engine.generate(prompt)

        print(f"Complexity Score: {result.complexity_score:.1f}/100")
        print(f"Selected Gear: {result.bit_width}-bit")
        print(f"Confidence: {result.gear_decision.confidence:.2f}")
        print(f"Inference Time: {result.inference_time_ms:.2f}ms")
        print(f"Tokens: {result.tokens_generated}")
        print()
        print(f"Response:\n{result.response}")
        print()
        print("=" * 60)
        print()


def run_benchmark(num_prompts: int = 100):
    """Run energy benchmark."""
    print("=" * 60)
    print("ENERGY BENCHMARK")
    print("=" * 60)
    print()

    engine = DynamicInferenceEngine()
    benchmark = EnergyBenchmark(engine)

    # Generate synthetic test prompts of varying complexity
    test_prompts = [
        "Hi!",
        "What is 2+2?",
        "Tell me a joke.",
        "Summarize this article.",
        "Explain machine learning.",
        "Write a Python script.",
        "Solve this differential equation.",
        "Analyze the geopolitical situation.",
    ] * (num_prompts // 8)

    print(f"Running benchmark with {len(test_prompts)} prompts...")
    print()

    results = benchmark.run_comparison(test_prompts)
    report = benchmark.generate_report(results)
    print(report)


def compare_all_gears():
    """Compare all gears on the same prompts."""
    print("=" * 60)
    print("GEAR COMPARISON")
    print("=" * 60)
    print()

    engine = DynamicInferenceEngine()

    prompts = [
        "Hello, how are you?",
        "Write a Python function to reverse a string.",
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        print("-" * 40)

        comparisons = engine.compare_gears(prompt)

        for gear, result in comparisons.items():
            print(f"\n{gear.value}-bit Mode:")
            print(f"  Time: {result.inference_time_ms:.2f}ms")
            print(f"  Tokens: {result.tokens_generated}")

        print()
        print("=" * 60)
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Green-Weight Energy Gearbox for AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m green_weight.main demo          # Run full demo
  python -m green_weight.main analyze       # Analyze prompts only
  python -m green_weight.main benchmark     # Run energy benchmark
  python -m green_weight.main compare       # Compare all gears
        """,
    )

    parser.add_argument(
        "command",
        choices=["demo", "analyze", "benchmark", "compare"],
        default="demo",
        nargs="?",
        help="Command to run",
    )

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts for benchmark (default: 100)",
    )

    args = parser.parse_args()

    if args.command == "demo":
        demo_prompt_analysis()
        demo_inference()
    elif args.command == "analyze":
        demo_prompt_analysis()
    elif args.command == "benchmark":
        run_benchmark(args.num_prompts)
    elif args.command == "compare":
        compare_all_gears()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
