"""
Tests for router/fuzzy_controller.py — the scikit-fuzzy-based routing
controller. Construction and route() both run for real (scikit-fuzzy,
numpy are installed; no GPU/model needed).

Sanity-prompt tier expectations use only the two clear-cut cases that are
stable regardless of config.yaml breakpoint calibration: a trivial factual
question should route to 4bit, and an explicit code-generation request
(which sets has_code_or_math=1, and every rule that fires on that pins
complexity["high"]) should route to 16bit.
"""

import numpy as np
import pytest
import skfuzzy as fuzz
import skfuzzy.control as ctrl

from router import complexity_scorer as cs
from router.fuzzy_controller import FuzzyController, _trimf_low_mid_high


class TestFuzzyControllerConstruction:
    def test_construction_succeeds(self):
        controller = FuzzyController()
        assert controller is not None
        assert controller.system is not None
        assert controller.simulator is not None

    def test_tier_thresholds_loaded_from_config(self):
        controller = FuzzyController()
        assert controller.tier_4bit_upper == 33
        assert controller.tier_8bit_upper == 66
        assert controller.tier_16bit_lower == 67


class TestRoute:
    @staticmethod
    @pytest.fixture(scope="class")
    def controller():
        return FuzzyController()

    def test_trivial_factual_prompt_routes_to_4bit(self, controller):
        features = cs.score("What is 2 + 2?")
        tier, win_prob = controller.route(features)
        assert tier == "4bit"
        assert 0.0 <= win_prob <= 1.0

    def test_code_generation_prompt_routes_to_16bit(self, controller):
        features = cs.score("Write a Python function that sorts a list using quicksort.")
        tier, win_prob = controller.route(features)
        assert tier == "16bit"
        assert 0.0 <= win_prob <= 1.0

    @pytest.mark.parametrize("prompt", [
        "What is 2 + 2?",
        "Write a Python function that sorts a list using quicksort.",
        "Explain quantum computing in detail, including superposition, "
        "entanglement, and quantum gates.",
        "Hi",
        "Implement a recursive algorithm for the Fibonacci sequence in C++.",
    ])
    def test_route_returns_valid_tier_and_probability(self, controller, prompt):
        features = cs.score(prompt)
        tier, win_prob = controller.route(features)
        assert tier in {"4bit", "8bit", "16bit"}
        assert 0.0 <= win_prob <= 1.0

    def test_missing_features_default_gracefully(self, controller):
        # route() uses .get() with defaults for every feature key, so an
        # empty dict must not raise.
        tier, win_prob = controller.route({})
        assert tier in {"4bit", "8bit", "16bit"}
        assert 0.0 <= win_prob <= 1.0


class TestTrimfLowMidHigh:
    """Locks in the documented, intentional behavior: only bps[0]/bps[1]
    are ever read. A 3rd breakpoint value (if present in a config list) is
    silently ignored -- this is not a bug to "fix", it's current, relied-on
    behavior."""

    def _build(self, bps):
        universe = np.arange(0, 1.01, 0.01)
        antecedent = ctrl.Antecedent(universe, "feature")
        _trimf_low_mid_high(antecedent, bps)
        return antecedent

    def test_third_breakpoint_is_ignored(self):
        two_bp = self._build([0.2, 0.5])
        three_bp = self._build([0.2, 0.5, 0.9])  # 3rd value (0.9) must be inert

        for label in ("low", "medium", "high"):
            np.testing.assert_array_equal(
                two_bp[label].mf, three_bp[label].mf,
                err_msg=f"membership function '{label}' differs when a 3rd "
                        f"breakpoint is present -- it should be ignored",
            )

    def test_medium_peak_is_midpoint_of_lo_hi(self):
        antecedent = self._build([0.2, 0.6])
        expected_mid = (0.2 + 0.6) / 2.0
        # medium is trimf([lo, mid, hi]); peak (mf value 1.0) sits at `mid`.
        universe = antecedent.universe
        medium_mf = antecedent["medium"].mf
        peak_idx = int(np.argmax(medium_mf))
        assert universe[peak_idx] == pytest.approx(expected_mid, abs=0.01)

    def test_empty_breakpoints_uses_defaults(self):
        antecedent = self._build([])
        universe = antecedent.universe
        medium_mf = antecedent["medium"].mf
        peak_idx = int(np.argmax(medium_mf))
        expected_mid = (0.33 + 0.66) / 2.0
        assert universe[peak_idx] == pytest.approx(expected_mid, abs=0.01)

    def test_single_breakpoint_uses_default_hi(self):
        antecedent = self._build([0.1])
        universe = antecedent.universe
        medium_mf = antecedent["medium"].mf
        peak_idx = int(np.argmax(medium_mf))
        expected_mid = (0.1 + 0.66) / 2.0
        assert universe[peak_idx] == pytest.approx(expected_mid, abs=0.01)
