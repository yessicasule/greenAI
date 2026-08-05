"""Fuzzy Logic Controller - The Gearbox for the Energy Gearbox system."""

from dataclasses import dataclass
from typing import Callable

from green_weight.config import BitWidth, FUZZY, GEARS


@dataclass(frozen=True)
class GearDecision:
    """Output from the fuzzy controller."""

    bit_width: BitWidth
    confidence: float  # How confident the decision is (0-1)
    simple_membership: float
    medium_membership: float
    complex_membership: float


class FuzzyGearbox:
    """
    Fuzzy Logic Controller for dynamic bit-width selection.

    Instead of binary decisions, uses fuzzy membership to smoothly
    transition between gears based on prompt complexity.
    """

    def __init__(self, config: FUZZY.__class__ = FUZZY):
        self.config = config

    def _triangular_membership(
        self,
        x: float,
        center: float,
        width: float,
    ) -> float:
        """
        Triangular membership function.

        Returns degree of membership (0-1) for value x in fuzzy set.
        """
        if x < center - width or x > center + width:
            return 0.0
        if x < center:
            return (x - (center - width)) / width
        return ((center + width) - x) / width

    def _calculate_memberships(self, complexity: float) -> dict[str, float]:
        """
        Calculate fuzzy membership degrees for each complexity class.

        Returns dict with membership values for simple, medium, complex.
        """
        width = (self.config.complexity_max - self.config.complexity_min) / 6

        simple = self._triangular_membership(
            complexity, self.config.simple_center, width
        )
        medium = self._triangular_membership(
            complexity, self.config.medium_center, width
        )
        complex_m = self._triangular_membership(
            complexity, self.config.complex_center, width
        )

        # Normalize to ensure sum = 1
        total = simple + medium + complex_m
        if total > 0:
            return {
                "simple": simple / total,
                "medium": medium / total,
                "complex": complex_m / total,
            }
        return {"simple": 0.33, "medium": 0.34, "complex": 0.33}

    def _defuzzify(self, memberships: dict[str, float]) -> float:
        """
        Convert fuzzy memberships to crisp bit-width value.

        Uses weighted average (centroid method).
        """
        simple_weight = memberships["simple"] * self.config.simple_output
        medium_weight = memberships["medium"] * self.config.medium_output
        complex_weight = memberships["complex"] * self.config.complex_output

        total_weight = memberships["simple"] + memberships["medium"] + memberships["complex"]

        if total_weight == 0:
            return float(BitWidth.MEDIUM)

        return (simple_weight + medium_weight + complex_weight) / total_weight

    def _select_gear(self, crisp_value: float) -> BitWidth:
        """Select final gear based on defuzzified value."""
        # Find closest bit-width
        candidates = [BitWidth.LOW, BitWidth.MEDIUM, BitWidth.HIGH]
        return min(candidates, key=lambda b: abs(float(b) - crisp_value))

    def decide(self, complexity_score: float) -> GearDecision:
        """
        Main entry point: decide which gear to use for given complexity.

        Args:
            complexity_score: 0-100 complexity value from the sensor

        Returns:
            GearDecision with selected bit-width and membership info
        """
        # Clamp to valid range
        complexity = max(
            self.config.complexity_min,
            min(complexity_score, self.config.complexity_max)
        )

        # Calculate fuzzy memberships
        memberships = self._calculate_memberships(complexity)

        # Defuzzify to get crisp bit-width
        crisp = self._defuzzify(memberships)

        # Select gear
        gear = self._select_gear(crisp)

        # Calculate confidence (highest membership value)
        confidence = max(memberships.values())

        return GearDecision(
            bit_width=gear,
            confidence=confidence,
            simple_membership=memberships["simple"],
            medium_membership=memberships["medium"],
            complex_membership=memberships["complex"],
        )

    def get_gear_config(self, decision: GearDecision) -> dict:
        """Get the configuration for a selected gear."""
        return GEARS[decision.bit_width]
