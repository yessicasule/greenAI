"""
Tests for router/complexity_scorer.py — the "sensor" layer that turns a raw
prompt into 5 normalized [0,1] features for the fuzzy controller.

spaCy's en_core_web_sm model is available in this environment (verified
before writing this suite), so get_parse_depth()/score() run for real
rather than being mocked out.
"""

import math

import pytest

from router import complexity_scorer as cs


# --------------------------------------------------------------------------
# normalize_to_01
# --------------------------------------------------------------------------

class TestNormalizeTo01:
    def test_min_maps_to_zero(self):
        assert cs.normalize_to_01(0, 0, 10) == 0.0

    def test_max_maps_to_one(self):
        assert cs.normalize_to_01(10, 0, 10) == 1.0

    def test_midpoint_maps_to_half(self):
        assert cs.normalize_to_01(5, 0, 10) == 0.5

    def test_below_min_is_clipped_to_zero(self):
        assert cs.normalize_to_01(-100, 0, 10) == 0.0

    def test_above_max_is_clipped_to_one(self):
        assert cs.normalize_to_01(1000, 0, 10) == 1.0

    def test_degenerate_max_equals_min_returns_half(self):
        assert cs.normalize_to_01(5, 3, 3) == 0.5

    def test_degenerate_max_less_than_min_returns_half(self):
        assert cs.normalize_to_01(5, 10, 3) == 0.5


# --------------------------------------------------------------------------
# shannon_entropy
# --------------------------------------------------------------------------

class TestShannonEntropy:
    def test_empty_string_is_zero(self):
        assert cs.shannon_entropy("") == 0.0

    def test_uniform_string_is_zero(self):
        # Single repeated symbol -> no uncertainty.
        assert cs.shannon_entropy("aaaaaaaa") == 0.0

    def test_two_symbols_equal_probability_is_one_bit(self):
        # "aabb": p(a)=p(b)=0.5 -> H = 1 bit
        assert cs.shannon_entropy("aabb") == pytest.approx(1.0)

    def test_four_symbols_equal_probability_is_two_bits(self):
        # "abcd": p=0.25 each -> H = log2(4) = 2 bits
        assert cs.shannon_entropy("abcd") == pytest.approx(2.0)

    def test_result_is_never_negative(self):
        assert cs.shannon_entropy("mississippi") >= 0.0


# --------------------------------------------------------------------------
# has_code_or_math
# --------------------------------------------------------------------------

class TestHasCodeOrMath:
    def test_fenced_code_block_detected(self):
        assert cs.has_code_or_math("Here:\n```python\nprint(1)\n```\n") is True

    def test_inline_latex_detected(self):
        assert cs.has_code_or_math("Solve $x^2 + 1 = 0$ for x.") is True

    def test_math_operator_symbol_detected(self):
        assert cs.has_code_or_math("Compute ∑ from 1 to 10.") is True

    def test_natural_language_code_request_detected(self):
        assert cs.has_code_or_math("Write a function to reverse a string.") is True

    def test_plain_prompt_not_detected(self):
        assert cs.has_code_or_math("What is the capital of France?") is False

    def test_plain_conversational_prompt_not_detected(self):
        assert cs.has_code_or_math("Tell me about your day.") is False


# --------------------------------------------------------------------------
# Range constants sanity (own boundaries normalize to 0.0 / 1.0)
# --------------------------------------------------------------------------

class TestRangeConstantBoundaries:
    @pytest.mark.parametrize("range_const", [
        "FLESCH_KINCAID_RANGE",
        "ENTROPY_RANGE",
        "SYNTAX_DEPTH_RANGE",
        "TOKEN_LENGTH_RANGE",
    ])
    def test_own_min_normalizes_to_zero(self, range_const):
        lo, hi = getattr(cs, range_const)
        assert cs.normalize_to_01(lo, lo, hi) == 0.0

    @pytest.mark.parametrize("range_const", [
        "FLESCH_KINCAID_RANGE",
        "ENTROPY_RANGE",
        "SYNTAX_DEPTH_RANGE",
        "TOKEN_LENGTH_RANGE",
    ])
    def test_own_max_normalizes_to_one(self, range_const):
        lo, hi = getattr(cs, range_const)
        assert cs.normalize_to_01(hi, lo, hi) == 1.0

    def test_ranges_are_well_formed(self):
        for lo, hi in (cs.FLESCH_KINCAID_RANGE, cs.ENTROPY_RANGE,
                       cs.SYNTAX_DEPTH_RANGE, cs.TOKEN_LENGTH_RANGE):
            assert hi > lo

    def test_token_length_range_matches_real_eval_set_calibration(self):
        # Calibrated 2026-08-22 against data/eval_prompts.jsonl's real
        # observed max (153.75 approx-tokens, rounded up to 154) — replaced
        # the old hardcoded 512-token cap, which made token_length
        # structurally near-dead as a routing signal (see
        # CREDIBILITY_REPORT.md / NEW.md Phase 4).
        assert cs.TOKEN_LENGTH_RANGE == (0, 154)


# --------------------------------------------------------------------------
# score()
# --------------------------------------------------------------------------

class TestScore:
    EXPECTED_KEYS = {"flesch_kincaid", "token_length", "entropy", "syntax_depth", "has_code_or_math"}

    def test_returns_all_five_keys(self):
        features = cs.score("What is the capital of France?")
        assert set(features.keys()) == self.EXPECTED_KEYS

    def test_all_values_in_zero_one_range(self):
        features = cs.score(
            "Write a Python function that implements quicksort with recursion, "
            "and explain its time complexity using big-O notation."
        )
        for key, val in features.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_has_code_or_math_is_binary(self):
        features = cs.score("Write a function to sort a list.")
        assert features["has_code_or_math"] in (0.0, 1.0)

        features2 = cs.score("What is your favorite color?")
        assert features2["has_code_or_math"] in (0.0, 1.0)

    def test_code_prompt_flags_has_code_or_math(self):
        features = cs.score("Write a Python function that sorts a list using quicksort.")
        assert features["has_code_or_math"] == 1.0

    def test_plain_prompt_does_not_flag_has_code_or_math(self):
        features = cs.score("What is the capital of France?")
        assert features["has_code_or_math"] == 0.0

    def test_empty_prompt_does_not_crash(self):
        features = cs.score("")
        assert set(features.keys()) == self.EXPECTED_KEYS
        for val in features.values():
            assert 0.0 <= val <= 1.0

    def test_token_length_increases_with_prompt_length(self):
        short = cs.score("Hi")["token_length"]
        long = cs.score("word " * 200)["token_length"]
        assert long > short
