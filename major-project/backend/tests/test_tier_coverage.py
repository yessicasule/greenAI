"""
End-to-end tier-coverage tests for the router (sensor + fuzzy controller
+ bridge), over a fixed 15-prompt set spanning all three precision tiers.

Why this file exists
--------------------
Every other router test checks a component in isolation. Nothing checked
the property that actually matters for the system as a whole: that all
three tiers stay *reachable*. Twice now a tier has silently gone dead
without a single unit test noticing.

  - Before 2026-09-09 the HIGH membership term was a triangle peaking at a
    feature value of 1.0, which no real prompt reaches. Every rule with a
    HIGH antecedent fired at ~zero strength, so the defuzzified score
    pinned near the neutral centroid and 16-bit took only 7% of the
    500-prompt eval set.
  - The bridge's `win_probability >= 0.5 -> 16bit` tie-break then did the
    reverse to the middle tier, collapsing 8-bit from 42.8% of prompts to
    4.2%.

Both were routing-quality failures invisible to a suite that only asserts
"returns one of the three valid tier strings". These tests fail loudly if
any tier becomes unreachable again.

The 15 prompts are 5 per expected tier, chosen to be clear-cut cases of
each kind: trivial factual lookups, mid-difficulty explanation tasks, and
code/derivation/proof tasks. Expected tiers are the author's labels, not
measured quantization sensitivity — a distinction that bounds what these
tests may claim. They pin routing *behaviour*, not ground truth about
which tier can actually answer each prompt correctly; that needs Session
2/4 data.
"""

import pytest

from router import complexity_scorer as cs
from router.fuzzy_controller import FuzzyController
from router.routellm_bridge import RouteLLMBridge

# (prompt, expected tier). 5 per tier.
PROMPTS = [
    # ---- trivial factual / arithmetic: the cheapest tier should do ----
    ("What is 2 + 2?", "4bit"),
    ("Who wrote Romeo and Juliet?", "4bit"),
    ("What is the capital of France?", "4bit"),
    ("How many days are in a week?", "4bit"),
    ("What is the speed of light?", "4bit"),
    # ---- moderate explanation / summarisation ----
    ("Explain the water cycle.", "8bit"),
    ("Summarize the causes of World War I.", "8bit"),
    ("Describe how photosynthesis works in plants.", "8bit"),
    ("What are the main differences between mitosis and meiosis?", "8bit"),
    ("Explain supply and demand in economics.", "8bit"),
    # ---- code / derivation / proof: needs full precision ----
    ("Write a Python function that sorts a list using quicksort.", "16bit"),
    ("Implement a balanced binary search tree with insertion and deletion in C++.", "16bit"),
    ("Discuss the philosophical implications of Godel's incompleteness theorems "
     "for the foundations of mathematics.", "16bit"),
    ("Derive the closed-form solution for ridge regression and explain the role "
     "of the regularization parameter.", "16bit"),
    ("Prove that the halting problem is undecidable using a diagonalization argument.", "16bit"),
]

TIER_ORDER = {"4bit": 0, "8bit": 1, "16bit": 2}

# Known misses as of 2026-09-09, kept in the set deliberately rather than
# quietly dropped — removing the prompts the router gets wrong would make
# the accuracy floor below meaningless:
#   "Who wrote Romeo and Juliet?"          -> 8bit  (expected 4bit)
#   "Summarize the causes of World War I." -> 4bit  (expected 8bit)
# Both are single-tier errors on genuinely borderline prompts. The floor
# is set at 12/15 to leave one prompt of slack beyond today's 13/15.
MIN_EXACT = 12


@pytest.fixture(scope="module")
def routed():
    """Route all 15 prompts once through the real sensor + controller +
    bridge, exactly as api.py's /route endpoint does."""
    controller = FuzzyController()
    bridge = RouteLLMBridge()
    out = []
    for prompt, expected in PROMPTS:
        features = cs.score(prompt)
        fuzzy_tier, win_prob = controller.route(features)
        final_tier = bridge.decide(prompt, win_prob)
        out.append({
            "prompt": prompt,
            "expected": expected,
            "fuzzy_tier": fuzzy_tier,
            "final_tier": final_tier,
            "score": win_prob * 100,
        })
    return out


class TestTierReachability:
    """The regression guard: no tier may go structurally unreachable."""

    @pytest.mark.parametrize("tier", ["4bit", "8bit", "16bit"])
    def test_every_tier_is_reachable(self, routed, tier):
        chosen = [r["final_tier"] for r in routed]
        assert tier in chosen, (
            "No prompt routed to {}. A precision tier has become "
            "unreachable — check membership-function shape, config.yaml "
            "breakpoints, and the bridge's zone mapping. Got: {}".format(tier, chosen)
        )

    @pytest.mark.parametrize("tier", ["4bit", "8bit", "16bit"])
    def test_no_tier_swallows_the_whole_set(self, routed, tier):
        chosen = [r["final_tier"] for r in routed]
        share = chosen.count(tier) / len(chosen)
        assert share <= 0.6, (
            "{} took {:.0%} of a set built to be evenly split across three "
            "tiers — the router has collapsed onto one tier.".format(tier, share)
        )

    def test_bridge_does_not_starve_the_middle_tier(self, routed):
        """The bridge is a documented pass-through over the fuzzy tier. It
        must not systematically move prompts off 8-bit, which is what the
        removed `>= 0.5 -> 16bit` tie-break did (42.8% -> 4.2% measured
        over the 500-prompt eval set)."""
        fuzzy_8 = sum(r["fuzzy_tier"] == "8bit" for r in routed)
        final_8 = sum(r["final_tier"] == "8bit" for r in routed)
        assert final_8 >= fuzzy_8, (
            "bridge reduced 8-bit from {} to {} prompts".format(fuzzy_8, final_8)
        )


class TestRoutingQuality:
    def test_meets_exact_accuracy_floor(self, routed):
        hits = [r for r in routed if r["final_tier"] == r["expected"]]
        misses = [r for r in routed if r["final_tier"] != r["expected"]]
        detail = ", ".join(
            "{!r} -> {} (expected {})".format(r["prompt"][:40], r["final_tier"], r["expected"])
            for r in misses
        )
        assert len(hits) >= MIN_EXACT, (
            "only {}/{} prompts routed to their expected tier (floor {}). "
            "Misses: {}".format(len(hits), len(routed), MIN_EXACT, detail)
        )

    def test_no_two_tier_errors(self, routed):
        """A 4-bit prompt reaching fp16, or vice versa, is a different
        class of failure from a one-tier borderline call."""
        for r in routed:
            gap = abs(TIER_ORDER[r["final_tier"]] - TIER_ORDER[r["expected"]])
            assert gap <= 1, (
                "{!r} expected {}, got {} — two tiers off".format(
                    r["prompt"][:50], r["expected"], r["final_tier"])
            )

    def test_complexity_score_orders_the_three_groups(self, routed):
        """Structural property, independent of where the tier thresholds
        happen to sit: harder prompt groups must score higher on average."""
        def mean_for(tier):
            vals = [r["score"] for r in routed if r["expected"] == tier]
            return sum(vals) / len(vals)

        easy, mid, hard = mean_for("4bit"), mean_for("8bit"), mean_for("16bit")
        assert easy < mid < hard, (
            "complexity score does not order the groups: 4bit-group={:.1f}, "
            "8bit-group={:.1f}, 16bit-group={:.1f}".format(easy, mid, hard)
        )

    def test_code_prompts_reach_the_top_tier(self, routed):
        """has_code_or_math is the sensor's strongest single signal; an
        explicit code-generation request must reach fp16."""
        for r in routed:
            if r["prompt"].startswith(("Write a Python", "Implement a balanced")):
                assert r["final_tier"] == "16bit", (
                    "{!r} routed to {}".format(r["prompt"][:50], r["final_tier"])
                )
