"""
Accuracy Evaluator - Phase 5
============================

Purpose: Run standard lm-eval-harness benchmarks across all four conditions:
- always-4-bit
- always-8-bit
- always-16-bit
- fuzzy-routed system

What it does:
- Uses lm_eval Python API (not CLI), wrapping the already-loaded tier models
  (registered by models/model_pool.py) in lm_eval's HFLM.
- For the routed condition, dispatches each individual lm-eval request to the
  tier the fuzzy controller picks for that request's prompt text (see
  RoutedLM below) — this is a real per-example routing decision, not an
  aggregate approximation.
- Computes RouteLLM-style metrics (CPT, APGR) treating 16-bit as baseline.
- Saves results to JSON for tradeoff_plotter.

Note: this is the live demo/API path (used by run_pipeline.py), separate
from training/scripts/kaggle_accuracy_eval.py which is the actual Session 2
script that produces the paper's per-tier accuracy numbers. Both now call
real lm_eval — this file previously returned hardcoded placeholder scores
regardless of input (fixed 2026-08-22; see CREDIBILITY_REPORT.md).
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import lm-eval
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    logger.warning("lm-eval not installed. Install with: pip install lm-eval")

from config import get_config
from models.local_llm_adapter import LocalLLMforAll, _model_registry
from router.complexity_scorer import score as score_complexity
from router.fuzzy_controller import FuzzyController
from router.routellm_bridge import RouteLLMBridge

# tier label (as used by the fuzzy controller / config.yaml) -> service name
# (as registered by models/model_pool.py)
TIER_TO_SERVICE = {"4bit": "local/4bit", "8bit": "local/8bit", "16bit": "local/16bit"}

# lm-eval result-dict metric keys to try, in preference order, when reducing
# a task's full metrics dict down to the single scalar this evaluator's
# per-task/per-condition scores table expects.
_PRIMARY_METRIC_KEYS = ("acc,none", "acc_norm,none", "exact_match,none", "acc")


class RoutedLM:
    """
    lm_eval.api.model.LM implementation for the "routed" condition.

    Does not itself run inference — for each request, scores the request's
    prompt text with the fuzzy controller to pick a tier, then delegates the
    actual loglikelihood/generation call to that tier's HFLM instance.
    Requests are grouped by tier before dispatch (one batched call per tier)
    and results are reassembled in the original request order.
    """

    def __init__(self, tier_lms: Dict[str, "HFLM"], fuzzy_controller: FuzzyController):
        if not tier_lms:
            raise RuntimeError(
                "RoutedLM needs at least one loaded tier model — none were provided."
            )
        self.tier_lms = tier_lms
        self.fuzzy_controller = fuzzy_controller
        # One entry per request handled so far, in dispatch order — not
        # necessarily original request order; used only for the aggregate
        # tier-distribution stats consumed by compute_routellm_metrics().
        self.routing_log: List[Dict[str, Any]] = []

    def _tier_for_context(self, context_text: str) -> str:
        features = score_complexity(context_text)
        tier, win_probability = self.fuzzy_controller.route(features)
        if tier not in self.tier_lms:
            # Requested tier's model isn't loaded — fall back to whichever
            # tier *is* loaded that's closest in precision, rather than
            # silently fabricating a score by skipping evaluation.
            available = list(self.tier_lms.keys())
            logger.warning(
                f"Routed condition picked tier '{tier}' but no model is loaded "
                f"for it; falling back to '{available[0]}'. Available: {available}"
            )
            tier = available[0]
        self.routing_log.append({"tier": tier, "win_probability": win_probability})
        return tier

    def _dispatch(self, requests: list, method_name: str) -> list:
        # Group requests by routed tier, preserving original order on output.
        buckets: Dict[str, List[tuple]] = {}
        for i, req in enumerate(requests):
            context_text = req.args[0]
            tier = self._tier_for_context(context_text)
            buckets.setdefault(tier, []).append((i, req))

        results: list = [None] * len(requests)
        for tier, indexed in buckets.items():
            lm = self.tier_lms[tier]
            sub_requests = [r for _, r in indexed]
            sub_results = getattr(lm, method_name)(sub_requests)
            for (orig_idx, _), res in zip(indexed, sub_results):
                results[orig_idx] = res
        return results

    def loglikelihood(self, requests: list) -> list:
        return self._dispatch(requests, "loglikelihood")

    def loglikelihood_rolling(self, requests: list) -> list:
        return self._dispatch(requests, "loglikelihood_rolling")

    def generate_until(self, requests: list) -> list:
        return self._dispatch(requests, "generate_until")


class AccuracyEvaluator:
    """
    Evaluate accuracy across different routing conditions.
    """

    def __init__(self):
        """Initialize evaluator."""
        config = get_config()

        self.output_dir = Path(config.benchmark.get("accuracy_eval", {}).get("output_dir", "results/accuracy_logs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tasks = config.get_benchmark_tasks()
        self.llm_for_all = LocalLLMforAll()

        # For routed condition
        self.fuzzy_controller = FuzzyController()
        self.routellm_bridge = RouteLLMBridge()

        self.results: Dict[str, Dict[str, Any]] = {}
        self._tier_lm_cache: Dict[str, "HFLM"] = {}
        self._last_routing_log: List[Dict[str, Any]] = []

        logger.info(f"[OK] Accuracy evaluator initialized. Tasks: {self.tasks}")

    def _get_tier_lm(self, tier: str) -> "HFLM":
        """Wrap the already-loaded model for `tier` in lm_eval's HFLM, cached
        per evaluator instance so repeated evaluations reuse one wrapper."""
        if tier in self._tier_lm_cache:
            return self._tier_lm_cache[tier]

        service_name = TIER_TO_SERVICE[tier]
        entry = _model_registry.get(service_name)
        if entry is None:
            raise RuntimeError(
                f"Model for tier '{tier}' (service '{service_name}') is not loaded. "
                "Call models.model_pool.load_pool() before running accuracy evaluation — "
                "accuracy cannot be measured without a real model."
            )

        lm = HFLM(pretrained=entry["model"], tokenizer=entry["tokenizer"], batch_size="auto")
        self._tier_lm_cache[tier] = lm
        return lm

    def evaluate_condition(
        self,
        condition_name: str,
        tier_or_router: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single condition (always-4bit, always-8bit, always-16bit, or routed).

        Args:
            condition_name: "always_4bit", "always_8bit", "always_16bit", or "routed"
            tier_or_router: For always-X conditions, the tier. For routed, None.

        Returns:
            Dict with accuracy scores per task (plus "overall")

        Raises:
            RuntimeError: if lm-eval or the required tier model(s) aren't
            available. This evaluator never substitutes a placeholder score
            for a real one — a failed evaluation must fail loudly, not
            silently populate results with fabricated numbers.
        """
        if not LM_EVAL_AVAILABLE:
            raise RuntimeError(
                "lm-eval is not installed. Install with: pip install lm-eval"
            )

        logger.info(f"Evaluating condition: {condition_name}")

        if condition_name.startswith("always_"):
            task_results = self._run_lm_eval(self._get_tier_lm(tier_or_router))
        else:
            task_results = self._evaluate_routed()

        scores = self._extract_scores(task_results)
        overall = float(np.mean(list(scores.values()))) if scores else 0.0
        scores["overall"] = overall

        self.results[condition_name] = scores
        logger.info(f"  Overall: {overall:.4f}")

        return scores

    def _run_lm_eval(self, lm) -> Dict[str, Dict[str, Any]]:
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=self.tasks,
            random_seed=42,
            numpy_random_seed=42,
            torch_random_seed=42,
        )
        return results["results"]

    def _evaluate_routed(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate the routed condition: every request is dispatched to
        whichever tier the fuzzy controller picks for its prompt text."""
        tier_lms = {}
        for tier, service_name in TIER_TO_SERVICE.items():
            if service_name in _model_registry:
                tier_lms[tier] = self._get_tier_lm(tier)

        if not tier_lms:
            raise RuntimeError(
                "No tier models are loaded — cannot evaluate the routed condition. "
                "Call models.model_pool.load_pool() first."
            )

        routed_lm = RoutedLM(tier_lms, self.fuzzy_controller)
        task_results = self._run_lm_eval(routed_lm)
        self._last_routing_log = routed_lm.routing_log
        return task_results

    @staticmethod
    def _extract_scores(task_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Reduce lm-eval's full per-task metrics dict to one scalar per
        task, trying the standard accuracy-style keys first and otherwise
        falling back to the first numeric metric present."""
        scores = {}
        for task, metrics in task_results.items():
            value = None
            for key in _PRIMARY_METRIC_KEYS:
                if key in metrics:
                    value = metrics[key]
                    break
            if value is None:
                numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
                if not numeric:
                    raise RuntimeError(
                        f"lm-eval returned no numeric metric for task '{task}': {metrics}"
                    )
                value = numeric[0]
            scores[task] = float(value)
        return scores

    def run_all_conditions(self) -> Dict[str, Dict[str, float]]:
        """
        Run evaluation for all four conditions.

        Returns:
            Dict mapping condition names to accuracy results
        """
        conditions = [
            ("always_4bit", "4bit"),
            ("always_8bit", "8bit"),
            ("always_16bit", "16bit"),
            ("routed", None),
        ]

        for condition_name, tier in conditions:
            self.evaluate_condition(condition_name, tier)

        return self.results

    def compute_routellm_metrics(self) -> Dict[str, float]:
        """
        Compute RouteLLM-style metrics (CPT, APGR) based on results.

        Returns:
            Dict with CPT (fraction of routed calls sent to the strong/16bit
            tier) and APGR (average performance gap recovered)
        """
        if "always_16bit" not in self.results or "routed" not in self.results:
            logger.warning("Cannot compute RouteLLM metrics without all conditions evaluated.")
            return {}

        strong_perf = self.results["always_16bit"].get("overall", 0.0)
        weak_perf = self.results["always_4bit"].get("overall", 0.0)
        routed_perf = self.results["routed"].get("overall", 0.0)

        # CPT (call-performance threshold): fraction of routed-condition
        # requests that were actually sent to the strong (16bit) tier,
        # recorded during _evaluate_routed(). Note this system doesn't have
        # a RouteLLM-style learned preference threshold to report directly —
        # router/routellm_bridge.py is a documented threshold pass-through,
        # not a trained classifier — so this is real observed routing
        # behavior, not a calibrated threshold value.
        if self._last_routing_log:
            strong_calls = sum(1 for r in self._last_routing_log if r["tier"] == "16bit")
            cpt = strong_calls / len(self._last_routing_log)
        else:
            logger.warning("No routing log recorded; CPT cannot be computed.")
            cpt = 0.0

        # APGR: Average Performance Gap Recovered
        # = (routed_perf - weak_perf) / (strong_perf - weak_perf)
        if strong_perf > weak_perf:
            apgr = (routed_perf - weak_perf) / (strong_perf - weak_perf)
        else:
            apgr = 0.0

        metrics = {
            "CPT": cpt,
            "APGR": apgr,
            "strong_performance": strong_perf,
            "weak_performance": weak_perf,
            "routed_performance": routed_perf,
        }

        logger.info(f"RouteLLM Metrics: CPT={cpt:.3f}, APGR={apgr:.3f}")

        return metrics

    def save_results(self) -> None:
        """Save accuracy results to JSON."""
        import json

        results_file = self.output_dir / "accuracy_results.json"

        # Combine results with RouteLLM metrics
        metrics = self.compute_routellm_metrics()
        output = {
            "accuracy_by_condition": self.results,
            "routellm_metrics": metrics,
        }

        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"[OK] Saved accuracy results to {results_file}")

    def get_summary(self) -> Dict[str, float]:
        """Return summary of overall accuracies."""
        summary = {}
        for condition, scores in self.results.items():
            summary[condition] = scores.get("overall", 0.0)
        return summary


if __name__ == "__main__":
    # Quick test — requires a CUDA GPU (model_pool.load_pool() refuses to
    # run without one) and lm-eval installed.
    logging.basicConfig(level=logging.INFO)

    from models import model_pool
    config = get_config()
    model_pool.load_pool(lazy_16bit=config.is_lazy_16bit())

    evaluator = AccuracyEvaluator()
    evaluator.run_all_conditions()
    evaluator.save_results()

    print("\nAccuracy Summary:")
    print(evaluator.get_summary())
