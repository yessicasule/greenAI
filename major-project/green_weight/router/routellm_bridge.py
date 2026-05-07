"""
RouteLLM Bridge - Phase 3–4 Bridge
==================================

Purpose: Translate the fuzzy controller's output into RouteLLM's API contract
so that RouteLLM's Controller class makes the binary strong/weak decision
consistent with the fuzzy tier assignment.

What it does:
- Instantiates a RouteLLM Controller using the router type from config
- Maps three tiers to RouteLLM's binary strong/weak model pair:
  * "16bit" ↔ strong_model
  * "4bit" ↔ weak_model
  * "8bit" ↔ intermediate (bypasses RouteLLM's binary decision)
- When fuzzy output is in MID zone (0.33–0.66), returns 8-bit directly
- When outside MID zone, defers to RouteLLM with the win-probability as threshold
"""

import logging
from typing import Tuple, Optional
import torch

logger = logging.getLogger(__name__)

# Try to import RouteLLM
try:
    from routellm.controller import Controller
    ROUTELLM_AVAILABLE = True
except ImportError:
    ROUTELLM_AVAILABLE = False
    logger.warning("RouteLLM not installed. Install with: pip install routellm")

from config import get_config


class RouteLLMBridge:
    """
    Adapter between fuzzy controller and RouteLLM's binary router.
    """
    
    def __init__(self):
        """Initialize the RouteLLM bridge."""
        zones = get_config().get_routing_zone_boundaries()
        self.mid_zone_lower = zones["mid_zone_lower"]
        self.mid_zone_upper = zones["mid_zone_upper"]

        if not ROUTELLM_AVAILABLE:
            logger.warning(
                "RouteLLM not available. Will fallback to fuzzy routing only. "
                "To enable RouteLLM, install with: pip install routellm"
            )
            self.controller = None
            return
        
        config = get_config()
        
        # Get configuration
        router_type = config.get_router_type()
        mf_checkpoint = config.get_mf_checkpoint()
        
        try:
            # Initialize RouteLLM controller with the specified router type
            # The controller expects a pre-trained router (mf, bert, or sw_ranking)
            logger.info(f"Loading RouteLLM controller (type={router_type})...")
            
            # For 'mf' router (matrix factorization), load from huggingface
            if router_type == "mf":
                # The RouteLLM paper shows that the pre-trained mf router
                # generalizes well to new model pairs without retraining
                self.controller = Controller(
                    model_name_or_path=mf_checkpoint,  # e.g., "routellm/mf_gpt4_augmented"
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            elif router_type == "bert":
                self.controller = Controller(
                    model_name_or_path="routellm/bert_gpt4_augmented",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            elif router_type == "sw_ranking":
                self.controller = Controller(
                    model_name_or_path="routellm/sw_ranking_gpt4_augmented",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                raise ValueError(f"Unknown router type: {router_type}")
            
            logger.info(f"[OK] RouteLLM controller initialized (router_type={router_type})")
            
        except Exception as e:
            logger.error(f"Failed to initialize RouteLLM controller: {e}")
            logger.warning("Falling back to fuzzy routing only.")
            self.controller = None
        
    def decide(self, prompt: str, win_probability: float) -> str:
        """
        Make a routing decision using RouteLLM if in extreme zones,
        or use 8-bit directly if in middle zone.
        
        Args:
            prompt: Input text
            win_probability: Fuzzy controller's win probability (0–1)
        
        Returns:
            Tier: "4bit", "8bit", or "16bit"
        """
        # If in middle zone, use 8-bit directly (bypass RouteLLM)
        if self.mid_zone_lower <= win_probability <= self.mid_zone_upper:
            logger.debug(
                f"Win probability {win_probability:.3f} in MID zone -> using 8bit directly"
            )
            return "8bit"
        
        # If RouteLLM not available, use fuzzy decision
        if self.controller is None:
            logger.debug("RouteLLM not available, using fuzzy mapping")
            if win_probability < 0.5:
                return "4bit"
            elif win_probability < 0.75:
                return "8bit"
            else:
                return "16bit"
        
        # Use RouteLLM's router with win_probability as threshold
        # The router outputs 0 (weak) or 1 (strong)
        # We interpret:
        #   0 (weak) -> 4-bit
        #   1 (strong) -> 16-bit
        try:
            # RouteLLM's Controller.route() expects:
            #   prompt: str
            #   threshold: float (decision boundary)
            # Returns: 0 (weak) or 1 (strong)
            decision = self.controller.route(prompt, threshold=win_probability)
            
            tier = "16bit" if decision == 1 else "4bit"
            logger.debug(
                f"RouteLLM decision: {decision} (threshold={win_probability:.3f}) -> {tier}"
            )
            return tier
            
        except Exception as e:
            logger.error(f"RouteLLM routing failed: {e}. Falling back to fuzzy mapping.")
            if win_probability < 0.5:
                return "4bit"
            elif win_probability < 0.75:
                return "8bit"
            else:
                return "16bit"
    
    def get_controller(self):
        """Get the underlying RouteLLM controller (for advanced use)."""
        return self.controller


def route(prompt: str, fuzzy_win_probability: float) -> str:
    """
    Convenience function: make a routing decision using RouteLLM bridge.
    
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
