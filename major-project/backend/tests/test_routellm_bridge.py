"""
Tests for router/routellm_bridge.py — confirmed by its own module docstring
to be a documented, provisional threshold pass-through, NOT a real RouteLLM
integration (no checkpoint is loaded/called). These tests exercise
`decide()` against known win_probability inputs; they deliberately do not
touch a real RouteLLM checkpoint.

Zone boundaries come from config.yaml's tier_thresholds
(4bit_upper=33, 8bit_upper=66 -> mid_zone_lower=0.33, mid_zone_upper=0.66).
"""

import pytest

from router.routellm_bridge import RouteLLMBridge, route


@pytest.fixture(scope="module")
def bridge():
    return RouteLLMBridge()


class TestDecide:
    def test_low_probability_routes_4bit(self, bridge):
        assert bridge.decide("irrelevant", 0.1) == "4bit"

    def test_just_below_lower_zone_routes_4bit(self, bridge):
        assert bridge.decide("irrelevant", 0.329999) == "4bit"

    def test_high_probability_routes_16bit(self, bridge):
        assert bridge.decide("irrelevant", 0.9) == "16bit"

    def test_just_above_upper_zone_routes_16bit(self, bridge):
        assert bridge.decide("irrelevant", 0.660001) == "16bit"

    def test_mid_zone_routes_8bit_across_the_whole_band(self, bridge):
        # The entire MID zone maps to 8-bit. This used to be split at 0.5,
        # with the upper half escalated to 16-bit; that tie-break was
        # removed 2026-09-09 after it was measured to collapse the 8-bit
        # tier from 42.8% of prompts to 4.2% and to drop agreement with
        # the eval set's difficulty labels from 49.8% to 38.4%. See
        # ROUTER_DIAGNOSIS.md and routellm_bridge.decide()'s docstring.
        #
        # 0.5 specifically is what the fuzzy controller emits when its
        # MEDIUM term fires alone — "no rule discriminated" — so it must
        # resolve to the middle tier, not the most expensive one.
        for wp in (0.34, 0.4, 0.5, 0.6, 0.66):
            assert bridge.decide("irrelevant", wp) == "8bit", wp

    def test_mid_zone_low_half_routes_8bit(self, bridge):
        # < 0.5 within the mid zone -> 8bit
        assert bridge.decide("irrelevant", 0.4) == "8bit"

    def test_lower_boundary_inclusive_is_mid_zone(self, bridge):
        # win_probability == mid_zone_lower is NOT < lower -> falls into MID
        assert bridge.decide("irrelevant", 0.33) == "8bit"  # 0.33 < 0.5

    def test_upper_boundary_inclusive_is_mid_zone(self, bridge):
        # win_probability == mid_zone_upper is NOT > upper -> falls into MID,
        # and the whole MID zone is 8-bit (see above). Only a score strictly
        # above the upper boundary reaches 16-bit.
        assert bridge.decide("irrelevant", 0.66) == "8bit"
        assert bridge.decide("irrelevant", 0.660001) == "16bit"

    def test_zero_routes_4bit(self, bridge):
        assert bridge.decide("irrelevant", 0.0) == "4bit"

    def test_one_routes_16bit(self, bridge):
        assert bridge.decide("irrelevant", 1.0) == "16bit"

    def test_prompt_argument_is_unused(self, bridge):
        # Documented: prompt is accepted but ignored today.
        assert bridge.decide("A", 0.1) == bridge.decide("completely different text", 0.1)


class TestModuleLevelRoute:
    def test_route_convenience_function_matches_bridge(self):
        assert route("irrelevant", 0.1) == "4bit"
        assert route("irrelevant", 0.9) == "16bit"


class TestZoneBoundaries:
    def test_zones_loaded_from_config(self, bridge):
        assert bridge.mid_zone_lower == pytest.approx(0.33)
        assert bridge.mid_zone_upper == pytest.approx(0.66)
