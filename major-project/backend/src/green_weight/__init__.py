"""
Green-Weight: Dynamic Routing + Quantized Cascade + Energy Tracking

A system for Green AI that combines:
- FrugalGPT: Cascade inference with generation judger
- RouteLLM: precision-tier threshold refinement (provisional pass-through
  today, not a real RouteLLM classifier — see router/routellm_bridge.py)
- Fuzzy Logic: Novel three-way precision router (router/fuzzy_controller.py)

The goal: Extreme energy efficiency while maintaining LLM accuracy.

Note on imports: the modules below (config.py, router/, models/, etc.) use
bare same-directory imports (e.g. `from config import get_config`), matching
the convention `api.py` and `run_pipeline.py` rely on when run with cwd
INSIDE this package directory (`cd backend/src/green_weight && uvicorn
api:app`). That means `import green_weight` from outside this directory
(e.g. `backend/src` as cwd) will NOT resolve these bare imports — the
try/except below exists so that failure is silent and diagnosable rather
than a crash, not because it's expected to succeed today.
"""

__version__ = "0.1.0"
__author__ = "Green AI Team"

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
    print(f"Warning: package-level imports need cwd inside green_weight/ to resolve: {e}")

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
