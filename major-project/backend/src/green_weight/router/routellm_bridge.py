"""
RouteLLM Bridge — Phase 3–4 Bridge
==================================

STATUS: provisional threshold pass-through, NOT a real RouteLLM integration.

Why: RouteLLM's pretrained checkpoints (e.g. `routellm/mf_gpt4_augmented`)
are trained on human-preference battles between specific named models
(GPT-4, Mixtral, ...) drawn from a closed `MODEL_IDS` set
(see RouteLLM/routellm/routers/routers.py). They have no learned signal for
"4-bit vs 8-bit vs 16-bit Llama-3.2-1B" — loading one and feeding it our
tiers would not error out, it would just silently return a win-probability
number with no real relationship to our system. That's worse than not
having it, so this bridge does not attempt it.

What this bridge actually does today: applies fixed thresholds to the fuzzy
controller's `win_probability` (itself `complexity_score / 100` — see
fuzzy_controller.py) to pick a tier. No RouteLLM model is loaded or called.

What replaces this: a real tier-preference router, trained on Session 4's
`routing_per_prompt.csv` (once it exists) — comparing our own tiers'
outputs the way RouteLLM's training methodology compares model outputs.
That is the actual RouteLLM-methodology contribution; see RESEARCH_PLAN.md
RQ3. This class will be extended (or replaced) to load that router once
trained.
"""

import logging

logger = logging.getLogger(__name__)

from config import get_config


class RouteLLMBridge:
    """
    Threshold-based tier refinement. See module docstring — this is a
    provisional pass-through, not a RouteLLM classifier.
    """

    def __init__(self):
        zones = get_config().get_routing_zone_boundaries()
        self.mid_zone_lower = zones["mid_zone_lower"]
        self.mid_zone_upper = zones["mid_zone_upper"]

    def decide(self, prompt: str, win_probability: float) -> str:
        """
        Route based on win_probability zones. `prompt` is unused today —
        kept in the signature because the real tier-preference router
        (see module docstring) will need the actual text to score, and
        callers already pass it.
        < 0.33  -> 4-bit (simple)
        0.33-0.66 -> 8-bit (medium), upper half (>=0.5) -> 16-bit (complex)
        > 0.66  -> 16-bit (complex)
        """
        if win_probability < self.mid_zone_lower:
            logger.debug(f"Win probability {win_probability:.3f} -> 4bit")
            return "4bit"
        elif win_probability > self.mid_zone_upper:
            logger.debug(f"Win probability {win_probability:.3f} -> 16bit")
            return "16bit"
        else:
            # MID zone: >= 0.5 goes to 16-bit (complex-leaning), < 0.5 stays 8-bit
            if win_probability >= 0.5:
                logger.debug(f"Win probability {win_probability:.3f} MID-high -> 16bit")
                return "16bit"
            logger.debug(f"Win probability {win_probability:.3f} MID-low -> 8bit")
            return "8bit"


def route(prompt: str, fuzzy_win_probability: float) -> str:
    """
    Convenience function: make a routing decision using the bridge.

    Args:
        prompt: Input text
        fuzzy_win_probability: Win probability from fuzzy controller (0–1)

    Returns:
        Tier: "4bit", "8bit", or "16bit"
    """
    bridge = RouteLLMBridge()
    return bridge.decide(prompt, fuzzy_win_probability)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.DEBUG)

    test_prompts = [
        ("What is 2 + 2?", 0.2),  # Simple -> should route to 4-bit
        ("Explain quantum computing.", 0.5),  # Medium -> 8-bit (MID zone)
        ("Write a complex algorithm in Python.", 0.85),  # Complex -> should route to 16-bit
    ]

    bridge = RouteLLMBridge()
    for prompt, win_prob in test_prompts:
        tier = bridge.decide(prompt, win_prob)
        print(f"\nPrompt: {prompt[:40]}... (win_prob={win_prob:.2f})")
        print(f"  -> Tier: {tier}")
