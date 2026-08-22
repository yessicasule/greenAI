"""
Fuzzy Controller - Phase 3–4 Bridge
===================================

Purpose: The novel contribution. Takes five complexity features and uses scikit-fuzzy
to produce a crisp routing decision — a tier label and a win probability float that
RouteLLM can consume as a threshold.

What it does:
- Defines three fuzzy membership functions (triangular/trapezoidal) for each feature
- Encodes a rule base of fuzzy IF-THEN rules
- Runs ControlSystemSimulation to defuzzify output to crisp tier + win probability
- Membership breakpoints are read from config.yaml for easy ablation studies
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np

try:
    import skfuzzy as fuzz
    import skfuzzy.control as ctrl
except ImportError:
    fuzz = None
    ctrl = None

from config import get_config
from router import complexity_scorer

logger = logging.getLogger(__name__)


def _trimf_low_mid_high(antecedent, bps, default_lo=0.33, default_hi=0.66):
    """Build symmetric LOW/MEDIUM/HIGH triangular membership functions from
    two edge breakpoints (lo, hi); MEDIUM's peak is their midpoint.

    Only reads bps[0]/bps[1]. A 3rd breakpoint value, if present (kept in
    config.yaml purely as a documented LOW/HIGH edge pair per the comments
    there), is intentionally ignored — see the 2026-08-22 note in
    config.yaml's fuzzy_controller section for why a true 3-breakpoint
    asymmetric partition was considered and deferred.
    """
    lo = bps[0] if bps else default_lo
    hi = bps[1] if len(bps) > 1 else default_hi
    mid = (lo + hi) / 2.0
    antecedent["low"] = fuzz.trimf(antecedent.universe, [0, 0, lo])
    antecedent["medium"] = fuzz.trimf(antecedent.universe, [lo, mid, hi])
    antecedent["high"] = fuzz.trimf(antecedent.universe, [hi, 1, 1])


class FuzzyController:
    """
    Fuzzy logic controller for three-way precision routing.
    
    Inputs:
    - flesch_kincaid (0–1): Readability
    - token_length (0–1): Prompt length
    - entropy (0–1): Character entropy
    - syntax_depth (0–1): Parse tree depth
    - has_code_or_math (0 or 1): Binary code/math detection
    
    Output:
    - complexity (0–100): Defuzzified complexity score
    - Mapped to tier: 4bit (0–33), 8bit (34–66), 16bit (67–100)
    - Also returns win_probability (0–1) for RouteLLM threshold
    """
    
    def __init__(self):
        """Initialize fuzzy system with membership functions and rules."""
        if fuzz is None or ctrl is None:
            raise ImportError(
                "scikit-fuzzy not installed. Install with: pip install scikit-fuzzy"
            )
        
        config = get_config()
        
        # Define input antecedents with membership functions
        self.flesch_kincaid = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "flesch_kincaid")
        self.token_length = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "token_length")
        self.entropy = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "entropy")
        self.syntax_depth = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "syntax_depth")
        self.has_code_or_math = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "has_code_or_math")
        
        # Define output consequent
        self.complexity = ctrl.Consequent(np.arange(0, 101, 1), "complexity")
        
        # Load breakpoints from config (stored in raw/absolute units, e.g.
        # FK grade level, entropy bits, parse-tree depth).
        fk_breakpoints = config.get_fuzzy_membership_breakpoints("flesch_kincaid")
        tl_breakpoints = config.get_fuzzy_membership_breakpoints("token_length")
        ent_breakpoints = config.get_fuzzy_membership_breakpoints("entropy")
        sd_breakpoints = config.get_fuzzy_membership_breakpoints("syntax_depth")

        # Normalize breakpoints into the SAME [0, 1] space as the actual
        # feature values. This must use complexity_scorer.normalize_to_01
        # with the identical (min, max) bounds it uses for the features
        # themselves — previously this reimplemented a diverging formula
        # (plain division, no min-offset) with different bounds (2-8 for
        # entropy, 1-20 for syntax_depth) than complexity_scorer.py used,
        # which silently misaligned config.yaml's breakpoints against the
        # real feature space (fixed 2026-08-05, see complexity_scorer.py's
        # FLESCH_KINCAID_RANGE/ENTROPY_RANGE/SYNTAX_DEPTH_RANGE).
        fk_breakpoints = [
            complexity_scorer.normalize_to_01(bp, *complexity_scorer.FLESCH_KINCAID_RANGE)
            for bp in fk_breakpoints
        ]
        tl_breakpoints = [min(1.0, bp) for bp in tl_breakpoints]  # Already 0–1
        ent_breakpoints = [
            complexity_scorer.normalize_to_01(bp, *complexity_scorer.ENTROPY_RANGE)
            for bp in ent_breakpoints
        ]
        sd_breakpoints = [
            complexity_scorer.normalize_to_01(bp, *complexity_scorer.SYNTAX_DEPTH_RANGE)
            for bp in sd_breakpoints
        ]
        
        _trimf_low_mid_high(self.flesch_kincaid, fk_breakpoints)
        _trimf_low_mid_high(self.token_length, tl_breakpoints)
        _trimf_low_mid_high(self.entropy, ent_breakpoints)
        _trimf_low_mid_high(self.syntax_depth, sd_breakpoints)
        
        # Code/math (essentially binary, but fuzzy membership helps smooth transitions)
        self.has_code_or_math["no"] = fuzz.trimf(self.has_code_or_math.universe, [0, 0, 0.5])
        self.has_code_or_math["yes"] = fuzz.trimf(self.has_code_or_math.universe, [0.5, 1, 1])
        
        # Output: complexity (0–100, mapped to LOW/MEDIUM/HIGH)
        self.complexity["low"] = fuzz.trimf(self.complexity.universe, [0, 0, 33])
        self.complexity["medium"] = fuzz.trimf(self.complexity.universe, [25, 50, 75])
        self.complexity["high"] = fuzz.trimf(self.complexity.universe, [67, 100, 100])
        
        # Define fuzzy rules
        # Rule base: heuristics for routing
        rules = [
            # Simple prompts (low on most features) -> use 4-bit
            ctrl.Rule(
                self.flesch_kincaid["low"] & self.token_length["low"] & 
                self.syntax_depth["low"] & self.has_code_or_math["no"],
                self.complexity["low"]
            ),
            
            # Complex reasoning (high on multiple features) -> use 16-bit
            ctrl.Rule(
                (self.syntax_depth["high"] | self.entropy["high"]) & 
                self.flesch_kincaid["high"],
                self.complexity["high"]
            ),
            
            # Code/math detected -> lean toward 16-bit
            ctrl.Rule(
                self.has_code_or_math["yes"],
                self.complexity["high"]
            ),
            
            # Long prompts -> lean toward 8-bit or 16-bit
            ctrl.Rule(
                self.token_length["high"],
                self.complexity["medium"]
            ),
            
            # High entropy (diverse vocabulary) -> lean toward 16-bit
            ctrl.Rule(
                self.entropy["high"] & self.flesch_kincaid["high"],
                self.complexity["high"]
            ),
            
            # Default/fallback: medium complexity for mid-range features
            ctrl.Rule(
                self.token_length["medium"] | self.syntax_depth["medium"],
                self.complexity["medium"]
            ),
            
            # Low on everything else -> favor 4-bit
            ctrl.Rule(
                self.flesch_kincaid["low"] & self.token_length["low"],
                self.complexity["low"]
            ),
        ]
        
        # Create control system
        self.system = ctrl.ControlSystem(rules)
        self.simulator = ctrl.ControlSystemSimulation(self.system)
        
        # Load tier thresholds from config
        thresholds = config.get_fuzzy_tier_thresholds()
        self.tier_4bit_upper = thresholds.get("4bit_upper", 33)
        self.tier_8bit_upper = thresholds.get("8bit_upper", 66)
        self.tier_16bit_lower = thresholds.get("16bit_lower", 67)
        
        logger.info("[OK] Fuzzy controller initialized")
    
    def route(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Route a prompt based on complexity features.
        
        Args:
            features: Dict with keys from complexity_scorer.score():
                     flesch_kincaid, token_length, entropy, syntax_depth, has_code_or_math
        
        Returns:
            Tuple of (tier_label, win_probability)
            - tier_label: "4bit", "8bit", or "16bit"
            - win_probability: float in [0, 1]. Despite the name (kept for
              API/frontend compatibility), this is `complexity_score / 100`
              from this fuzzy system — NOT a RouteLLM classifier output.
              See router/routellm_bridge.py's module docstring for why a
              real RouteLLM checkpoint can't be used here.
        """
        # Set input values
        self.simulator.input["flesch_kincaid"] = features.get("flesch_kincaid", 0.5)
        self.simulator.input["token_length"] = features.get("token_length", 0.5)
        self.simulator.input["entropy"] = features.get("entropy", 0.5)
        self.simulator.input["syntax_depth"] = features.get("syntax_depth", 0.5)
        self.simulator.input["has_code_or_math"] = features.get("has_code_or_math", 0.0)
        
        # Compute
        try:
            self.simulator.compute()
        except Exception as e:
            logger.error(f"Fuzzy computation failed: {e}. Defaulting to 8-bit.")
            return "8bit", 0.5
        
        # Get defuzzified complexity output
        complexity_score = self.simulator.output.get("complexity", 50.0)
        
        # Map complexity to tier
        if complexity_score <= self.tier_4bit_upper:
            tier = "4bit"
        elif complexity_score <= self.tier_8bit_upper:
            tier = "8bit"
        else:
            tier = "16bit"
        
        # Compute win probability (normalized complexity to [0, 1])
        win_probability = complexity_score / 100.0
        
        logger.debug(
            f"Route decision: complexity={complexity_score:.1f} -> tier={tier}, "
            f"win_prob={win_probability:.3f}"
        )
        
        return tier, win_probability


def route_prompt(prompt: str) -> Tuple[str, float]:
    """
    Convenience function: score a prompt and route it in one call.
    
    Args:
        prompt: Input text
    
    Returns:
        Tuple of (tier, win_probability)
    """
    # Score the prompt
    features = complexity_scorer.score(prompt)
    
    # Create controller and route (creates a new instance each time;
    # in production, you'd create once and reuse)
    controller = FuzzyController()
    tier, win_prob = controller.route(features)
    
    return tier, win_prob


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.DEBUG)
    
    test_prompts = [
        "What is 2 + 2?",
        "Write a Python quicksort function.",
        "Explain quantum computing in detail.",
    ]
    
    controller = FuzzyController()
    for prompt in test_prompts:
        features = complexity_scorer.score(prompt)
        tier, win_prob = controller.route(features)
        print(f"\nPrompt: {prompt[:40]}...")
        print(f"  Tier: {tier}, Win prob: {win_prob:.3f}")
