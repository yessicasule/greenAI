"""Energy and accuracy benchmarking for the Energy Gearbox."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from green_weight.config import BitWidth
from green_weight.core.dynamic_inference import DynamicInferenceEngine, InferenceResult


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmarking."""

    total_prompts: int = 0
    total_time_ms: float = 0.0
    total_tokens: int = 0
    energy_joules_estimate: float = 0.0

    # Per-bit-width stats
    prompts_by_gear: dict[BitWidth, int] = field(default_factory=dict)
    time_by_gear: dict[BitWidth, float] = field(default_factory=dict)
    tokens_by_gear: dict[BitWidth, int] = field(default_factory=dict)

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(self.total_prompts, 1)

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / max(self.total_prompts, 1)

    @property
    def throughput_tokens_per_sec(self) -> float:
        return (self.total_tokens / self.total_time_ms) * 1000


def estimate_energy_joules(
    bit_width: BitWidth,
    tokens: int,
    model_params: int = 7_000_000_000,  # 7B model
) -> float:
    """
    Estimate energy consumption in Joules.

    Rough approximation based on:
    - 4-bit: ~0.25x energy of 16-bit
    - 8-bit: ~0.5x energy of 16-bit
    - 16-bit: Baseline

    Formula: E = ops * energy_per_op
    ops ≈ 2 * params * tokens for forward pass
    """
    base_energy_per_op = 1e-12  # 1 pJ per op (rough estimate)
    ops = 2 * model_params * tokens

    # Bit-width scaling factors
    scaling = {
        BitWidth.LOW: 0.25,
        BitWidth.MEDIUM: 0.5,
        BitWidth.HIGH: 1.0,
    }

    return ops * base_energy_per_op * scaling[bit_width]


class EnergyBenchmark:
    """Benchmark energy consumption and accuracy."""

    def __init__(
        self,
        engine: DynamicInferenceEngine,
        energy_fn: Callable[[BitWidth, int], float] = estimate_energy_joules,
    ):
        self.engine = engine
        self.energy_fn = energy_fn

    def run_single(
        self,
        prompt: str,
        reference: str | None = None,
    ) -> InferenceResult:
        """Run single benchmark and collect metrics."""
        result = self.engine.generate(prompt)
        return result

    def run_comparison(
        self,
        prompts: list[str],
        references: list[str] | None = None,
    ) -> dict[str, BenchmarkMetrics]:
        """
        Compare fuzzy vs static (always 16-bit) approach.

        Returns metrics for both approaches.
        """
        fuzzy_metrics = BenchmarkMetrics()
        static_metrics = BenchmarkMetrics()

        for i, prompt in enumerate(prompts):
            # Fuzzy approach
            fuzzy_result = self.engine.generate(prompt)
            fuzzy_metrics.total_prompts += 1
            fuzzy_metrics.total_time_ms += fuzzy_result.inference_time_ms
            fuzzy_metrics.total_tokens += fuzzy_result.tokens_generated
            fuzzy_metrics.energy_joules_estimate += self.energy_fn(
                fuzzy_result.bit_width,
                fuzzy_result.tokens_generated,
            )

            gear = fuzzy_result.bit_width
            fuzzy_metrics.prompts_by_gear[gear] = fuzzy_metrics.prompts_by_gear.get(gear, 0) + 1
            fuzzy_metrics.time_by_gear[gear] = fuzzy_metrics.time_by_gear.get(gear, 0) + fuzzy_result.inference_time_ms
            fuzzy_metrics.tokens_by_gear[gear] = fuzzy_metrics.tokens_by_gear.get(gear, 0) + fuzzy_result.tokens_generated

            # Static approach (always 16-bit)
            static_result = self.engine.generate(prompt, force_gear=BitWidth.HIGH)
            static_metrics.total_prompts += 1
            static_metrics.total_time_ms += static_result.inference_time_ms
            static_metrics.total_tokens += static_result.tokens_generated
            static_metrics.energy_joules_estimate += self.energy_fn(
                BitWidth.HIGH,
                static_result.tokens_generated,
            )

        return {
            "fuzzy": fuzzy_metrics,
            "static_16bit": static_metrics,
        }

    def calculate_savings(
        self,
        fuzzy_metrics: BenchmarkMetrics,
        static_metrics: BenchmarkMetrics,
    ) -> dict[str, float]:
        """Calculate energy and time savings."""
        energy_savings = (
            (static_metrics.energy_joules_estimate - fuzzy_metrics.energy_joules_estimate)
            / static_metrics.energy_joules_estimate
            * 100
        )
        time_savings = (
            (static_metrics.total_time_ms - fuzzy_metrics.total_time_ms)
            / static_metrics.total_time_ms
            * 100
        )

        return {
            "energy_savings_percent": energy_savings,
            "time_savings_percent": time_savings,
            "energy_joules_saved": static_metrics.energy_joules_estimate - fuzzy_metrics.energy_joules_estimate,
            "time_ms_saved": static_metrics.total_time_ms - fuzzy_metrics.total_time_ms,
        }

    def generate_report(
        self,
        results: dict[str, BenchmarkMetrics],
        output_path: Path | None = None,
    ) -> str:
        """Generate human-readable benchmark report."""
        fuzzy = results["fuzzy"]
        static = results["static_16bit"]
        savings = self.calculate_savings(fuzzy, static)

        report = []
        report.append("=" * 60)
        report.append("GREEN-WEIGHT ENERGY GEARBOX BENCHMARK REPORT")
        report.append("=" * 60)
        report.append("")

        # Fuzzy approach stats
        report.append("FUZZY DYNAMIC APPROACH:")
        report.append(f"  Total prompts: {fuzzy.total_prompts}")
        report.append(f"  Avg time: {fuzzy.avg_time_ms:.2f} ms")
        report.append(f"  Avg tokens: {fuzzy.avg_tokens:.1f}")
        report.append(f"  Throughput: {fuzzy.throughput_tokens_per_sec:.1f} tokens/sec")
        report.append(f"  Est. energy: {fuzzy.energy_joules_estimate:.4f} J")
        report.append("")
        report.append("  Gear distribution:")
        for gear in BitWidth:
            count = fuzzy.prompts_by_gear.get(gear, 0)
            pct = count / max(fuzzy.total_prompts, 1) * 100
            report.append(f"    {gear.value}-bit: {count} ({pct:.1f}%)")
        report.append("")

        # Static approach stats
        report.append("STATIC 16-BIT APPROACH:")
        report.append(f"  Total prompts: {static.total_prompts}")
        report.append(f"  Avg time: {static.avg_time_ms:.2f} ms")
        report.append(f"  Est. energy: {static.energy_joules_estimate:.4f} J")
        report.append("")

        # Savings
        report.append("SAVINGS (FUZZY vs STATIC):")
        report.append(f"  Energy saved: {savings['energy_savings_percent']:.1f}%")
        report.append(f"  Time saved: {savings['time_savings_percent']:.1f}%")
        report.append(f"  Joules saved: {savings['energy_joules_saved']:.6f} J")
        report.append("")
        report.append("=" * 60)

        report_text = "\n".join(report)

        if output_path:
            output_path.write_text(report_text)
            print(f"Report saved to {output_path}")

        return report_text


class AccuracyBenchmark:
    """Benchmark accuracy across different bit-widths."""

    def __init__(self, engine: DynamicInferenceEngine):
        self.engine = engine

    def evaluate_on_dataset(
        self,
        dataset: list[dict[str, str]],  # [{"prompt": ..., "reference": ...}]
    ) -> dict[BitWidth, float]:
        """
        Evaluate accuracy for each bit-width.

        Returns accuracy score for each gear.
        """
        results = {}

        for gear in BitWidth:
            correct = 0
            total = 0

            for item in dataset:
                result = self.engine.generate(item["prompt"], force_gear=gear)
                # Simple accuracy: does response contain expected keywords?
                # In practice, use BLEU/ROUGE or human evaluation
                if self._check_accuracy(result.response, item.get("reference", "")):
                    correct += 1
                total += 1

            results[gear] = correct / max(total, 1)

        return results

    def _check_accuracy(self, response: str, reference: str) -> bool:
        """Simple accuracy check - can be replaced with proper metrics."""
        # Placeholder: check if key words from reference appear in response
        ref_words = set(reference.lower().split())
        resp_words = set(response.lower().split())
        overlap = len(ref_words & resp_words)
        return overlap / max(len(ref_words), 1) > 0.3

    def plot_tradeoff_curve(
        self,
        energy_results: dict[BitWidth, float],
        accuracy_results: dict[BitWidth, float],
        output_path: Path | None = None,
    ):
        """Generate energy-accuracy tradeoff visualization."""
        # This would create a plot showing the Pareto frontier
        # For now, just print the data
        print("\nEnergy-Accuracy Tradeoff:")
        print("-" * 40)
        print(f"{'Bit-Width':<10} {'Energy (J)':<15} {'Accuracy':<10}")
        print("-" * 40)
        for gear in BitWidth:
            energy = energy_results.get(gear, 0)
            acc = accuracy_results.get(gear, 0)
            print(f"{gear.value:<10} {energy:<15.6f} {acc:<10.2%}")
