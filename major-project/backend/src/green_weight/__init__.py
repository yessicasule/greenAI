"""
Green-Weight: Dynamic Routing + Quantized Cascade + Energy Tracking

A system for Green AI that combines:
- FrugalGPT: Cascade inference with generation judger
- RouteLLM: Pre-trained binary router for model selection
- Fuzzy Logic: Novel three-way precision router

The goal: Extreme energy efficiency while maintaining LLM accuracy.
"""

__version__ = "0.1.0"
__author__ = "Green AI Team"

# Import key components for easy access
try:
    from config import get_config, Config
    from models.model_pool import load_pool, infer, warmup
    from models.local_llm_adapter import LocalLLMforAll
    from router.complexity_scorer import score as score_complexity
    from router.fuzzy_controller import FuzzyController, route_prompt
    from router.routellm_bridge import RouteLLMBridge
    from cascade.frugal_cascade import FrugalCascade, run_cascade
    from benchmark.energy_tracker import EnergyTracker
    from benchmark.accuracy_eval import AccuracyEvaluator
    from benchmark.tradeoff_plotter import TradeoffPlotter
except ImportError as e:
    print(f"Warning: Could not import all components: {e}")

__all__ = [
    "get_config",
    "Config",
    "load_pool",
    "infer",
    "warmup",
    "LocalLLMforAll",
    "score_complexity",
    "FuzzyController",
    "route_prompt",
    "RouteLLMBridge",
    "FrugalCascade",
    "run_cascade",
    "EnergyTracker",
    "AccuracyEvaluator",
    "TradeoffPlotter",
]
from green_weight.controllers.fuzzy_gearbox import FuzzyGearbox
from green_weight.core.dynamic_inference import DynamicInferenceEngine
from green_weight.core.prompt_complexity import ComplexityScorer

__all__ = [
    "BitWidth",
    "GEARS",
    "FuzzyGearbox",
    "DynamicInferenceEngine",
    "ComplexityScorer",
]
