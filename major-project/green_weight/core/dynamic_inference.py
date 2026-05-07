"""Dynamic Inference Engine - The main orchestrator."""

import time
from dataclasses import dataclass
from typing import Iterator

from green_weight.config import BitWidth, GEARS
from green_weight.controllers.fuzzy_gearbox import FuzzyGearbox, GearDecision
from green_weight.core.prompt_complexity import ComplexityScorer


@dataclass
class InferenceResult:
    """Result of a dynamic inference call."""

    prompt: str
    complexity_score: float
    gear_decision: GearDecision
    bit_width: BitWidth
    response: str
    inference_time_ms: float
    tokens_generated: int


class DynamicInferenceEngine:
    """
    Main inference engine that uses the Energy Gearbox.

    Pipeline:
    1. Score prompt complexity (Sensor)
    2. Select gear via fuzzy logic (Gearbox)
    3. Run inference at selected bit-width
    4. Return result with metadata
    """

    def __init__(
        self,
        model=None,  # Placeholder for actual model
        scorer: ComplexityScorer | None = None,
        gearbox: FuzzyGearbox | None = None,
    ):
        self.model = model
        self.scorer = scorer or ComplexityScorer()
        self.gearbox = gearbox or FuzzyGearbox()

    def analyze_prompt(self, prompt: str) -> tuple[float, GearDecision]:
        """Analyze prompt and decide on gear without running inference."""
        complexity = self.scorer.calculate_complexity(prompt)
        decision = self.gearbox.decide(complexity)
        return complexity, decision

    def generate(
        self,
        prompt: str,
        force_gear: BitWidth | None = None,
    ) -> InferenceResult:
        """
        Generate response with dynamic bit-width selection.

        Args:
            prompt: Input text
            force_gear: Optional fixed gear (for testing/comparison)

        Returns:
            InferenceResult with full metadata
        """
        # Step 1: Analyze complexity
        complexity, decision = self.analyze_prompt(prompt)

        # Step 2: Select gear
        if force_gear is not None:
            bit_width = force_gear
            decision = GearDecision(
                bit_width=force_gear,
                confidence=1.0,
                simple_membership=0.0,
                medium_membership=0.0,
                complex_membership=0.0,
            )
        else:
            bit_width = decision.bit_width

        gear_config = GEARS[bit_width]

        # Step 3: Run inference (placeholder)
        start_time = time.perf_counter()
        response = self._run_inference(prompt, gear_config)
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Estimate tokens (rough approximation)
        tokens_generated = len(response.split()) * 1.3  # Rough token estimate

        return InferenceResult(
            prompt=prompt,
            complexity_score=complexity,
            gear_decision=decision,
            bit_width=bit_width,
            response=response,
            inference_time_ms=inference_time_ms,
            tokens_generated=int(tokens_generated),
        )

    def _run_inference(self, prompt: str, gear_config) -> str:
        """
        Placeholder for actual model inference.

        In production, this would:
        1. Load the appropriate quantized model/adapter
        2. Run generation with gear_config parameters
        3. Return the generated text
        """
        # Placeholder response for demonstration
        return (
            f"[Simulated response using {gear_config.bit_width}-bit precision]\n"
            f"Config: {gear_config.description}\n"
            f"Max tokens: {gear_config.max_tokens}, Temp: {gear_config.temperature}"
        )

    def stream_generate(
        self,
        prompt: str,
        force_gear: BitWidth | None = None,
    ) -> Iterator[str]:
        """Stream generation with dynamic gear selection."""
        # Analyze first
        complexity, decision = self.analyze_prompt(prompt)
        bit_width = force_gear or decision.bit_width

        # Yield analysis info
        yield f"[Gear selected: {bit_width}-bit | Complexity: {complexity:.1f}]\n\n"

        # Stream actual response (placeholder)
        response = self._run_inference(prompt, GEARS[bit_width])
        for word in response.split():
            yield word + " "
            time.sleep(0.01)  # Simulate streaming

    def compare_gears(self, prompt: str) -> dict[BitWidth, InferenceResult]:
        """Run same prompt through all gears for comparison."""
        return {
            gear: self.generate(prompt, force_gear=gear)
            for gear in BitWidth
        }
