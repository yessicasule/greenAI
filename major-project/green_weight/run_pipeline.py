"""
Run Pipeline - Main Orchestrator
================================

Purpose: The single entry point that wires everything together in the correct order.

What it does:
1. Loads config
2. Initializes model pool
3. Loads and processes prompts from eval_prompts.jsonl
4. For each prompt:
   - Scores complexity
   - Routes via fuzzy + RouteLLM
   - Runs cascade with energy tracking
   - Logs full trace
5. Runs accuracy evaluation across all conditions
6. Generates trade-off plots

Supports --dry-run flag for quick sanity checks.
"""

import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from config import get_config
from models import model_pool
from router.complexity_scorer import score as score_complexity
from router.fuzzy_controller import FuzzyController
from router.routellm_bridge import RouteLLMBridge
from benchmark.energy_tracker import EnergyTracker
from benchmark.accuracy_eval import AccuracyEvaluator
from benchmark.tradeoff_plotter import TradeoffPlotter

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> None:
    """Configure logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    logger.info(f"[OK] Logging initialized. Log file: {log_file}")


def load_eval_prompts(dataset_path: Path, limit: int = None) -> List[Dict[str, Any]]:
    """
    Load evaluation prompts from JSONL file.
    
    Args:
        dataset_path: Path to eval_prompts.jsonl
        limit: Max number of prompts to load
    
    Returns:
        List of prompt dicts with keys: prompt, reference_answer, difficulty_label
    """
    if not dataset_path.exists():
        logger.warning(f"Eval prompts file not found: {dataset_path}")
        logger.warning("Using synthetic prompts for testing.")
        return _synthetic_prompts(limit or 5)
    
    prompts = []
    try:
        with open(dataset_path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                prompts.append(json.loads(line.strip()))
    except Exception as e:
        logger.error(f"Failed to load prompts from {dataset_path}: {e}")
        return _synthetic_prompts(limit or 5)
    
    logger.info(f"[OK] Loaded {len(prompts)} prompts from {dataset_path}")
    return prompts


def _synthetic_prompts(count: int) -> List[Dict[str, Any]]:
    """Generate synthetic prompts for testing."""
    return [
        {
            "prompt": f"What is {i}+{i}?",
            "reference_answer": str(2*i),
            "difficulty_label": "easy"
        }
        for i in range(1, count+1)
    ]


def run_pipeline(args) -> None:
    """
    Main pipeline: orchestrate the entire system.
    
    Args:
        args: Parsed command-line arguments
    """
    config = get_config()
    
    # Setup logging
    log_dir = Path(config.get_log_output_dir())
    setup_logging(log_dir)
    
    logger.info("=" * 80)
    logger.info("GREEN-WEIGHT PIPELINE START")
    logger.info("=" * 80)
    logger.info(f"Dry-run mode: {args.dry_run}")
    logger.info(f"Routing-only mode: {args.routing_only}")

    # =========================================================================
    # Phase 1: Initialize Model Pool (skipped in routing-only mode)
    # =========================================================================
    logger.info("\n[Phase 1] Initializing model pool...")
    if args.routing_only:
        logger.info("[Phase 1] Skipped (routing-only mode - no GPU required)")
    else:
        try:
            model_pool.load_pool(lazy_16bit=config.is_lazy_16bit())

            if config.pipeline.get("warmup_enabled", True):
                model_pool.warmup(num_prompts=5)
        except Exception as e:
            logger.error(f"Failed to initialize model pool: {e}")
            return
    
    # =========================================================================
    # Phase 2: Load Evaluation Prompts
    # =========================================================================
    logger.info("\n[Phase 2] Loading evaluation prompts...")
    eval_dataset_path = Path(config.get_eval_dataset_path())
    
    limit_prompts = args.dry_run_count if args.dry_run else None
    prompts = load_eval_prompts(eval_dataset_path, limit=limit_prompts)
    logger.info(f"Will process {len(prompts)} prompts")
    
    # =========================================================================
    # Phase 3-4: Routing and Cascade
    # =========================================================================
    logger.info("\n[Phase 3-4] Routing and cascade inference...")
    
    energy_tracker = None if args.routing_only else EnergyTracker()
    fuzzy_controller = FuzzyController()
    routellm_bridge = RouteLLMBridge()
    
    pipeline_log = []  # Full trace for all prompts
    
    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]
        
        logger.info(f"\n[{i+1}/{len(prompts)}] Processing prompt: {prompt[:60]}...")
        
        try:
            # Step 1: Score complexity
            features = score_complexity(prompt)
            logger.debug(f"  Complexity scores: {features}")
            
            # Step 2: Route with fuzzy controller
            tier, win_prob = fuzzy_controller.route(features)
            logger.debug(f"  Fuzzy routing: tier={tier}, win_prob={win_prob:.3f}")
            
            # Step 3: Refine with RouteLLM bridge
            final_tier = routellm_bridge.decide(prompt, win_prob)
            logger.debug(f"  Final tier: {final_tier}")
            
            if args.routing_only:
                response = ""
                energy_info = {"joules": 0.0, "duration_s": 0.0}
            else:
                # Step 4: Run cascade with energy tracking
                response, energy_info = energy_tracker.track_inference(
                    prompt,
                    final_tier,
                    max_new_tokens=config.model.get("max_new_tokens", 256)
                )

            # Log entry
            log_entry = {
                "index": i,
                "prompt": prompt,
                "complexity_features": features,
                "tier": final_tier,
                "win_probability": win_prob,
                "response": response[:200],
                "energy_joules": energy_info.get("joules", 0.0),
                "duration_s": energy_info.get("duration_s", 0.0),
            }
            pipeline_log.append(log_entry)

            logger.info(f"  [OK] Complete: tier={final_tier}, energy={energy_info.get('joules', 0):.1f}J")
            
        except Exception as e:
            logger.error(f"  [FAIL] Failed: {e}")
            pipeline_log.append({
                "index": i,
                "prompt": prompt,
                "error": str(e)
            })
    
    # Save pipeline log
    pipeline_log_file = log_dir / "pipeline_trace.jsonl"
    with open(pipeline_log_file, "w") as f:
        for entry in pipeline_log:
            f.write(json.dumps(entry) + "\n")
    logger.info(f"[OK] Saved pipeline trace to {pipeline_log_file}")
    
    # Save energy logs
    if energy_tracker is not None:
        energy_tracker.save_logs()
        energy_stats = energy_tracker.get_summary_stats()
        logger.info(f"\nEnergy Summary:\n{json.dumps(energy_stats, indent=2)}")
    
    # =========================================================================
    # Phase 5: Accuracy Evaluation
    # =========================================================================
    if not args.dry_run:
        logger.info("\n[Phase 5] Accuracy evaluation...")
        try:
            evaluator = AccuracyEvaluator()
            evaluator.run_all_conditions()
            evaluator.save_results()
            
            summary = evaluator.get_summary()
            logger.info(f"Accuracy Summary:\n{json.dumps(summary, indent=2)}")
            
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
    else:
        logger.info("[Phase 5] Skipped (dry-run mode)")
    
    # =========================================================================
    # Phase 5b: Visualization
    # =========================================================================
    if not args.dry_run:
        logger.info("\n[Phase 5b] Generating trade-off plots...")
        try:
            plotter = TradeoffPlotter()
            plotter.generate()
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
    else:
        logger.info("[Phase 5b] Skipped (dry-run mode)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Processed {len(prompts)} prompts")
    logger.info(f"Logs saved to: {log_dir}")
    
    if not args.dry_run:
        results_dir = Path(config.get_figure_output_dir())
        logger.info(f"Results saved to: {results_dir}")


def main():
    """Parse arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description="Green-Weight Pipeline: Dynamic routing + quantized cascade + energy tracking"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only first N prompts (set via config) for quick testing"
    )
    parser.add_argument(
        "--dry-run-count",
        type=int,
        default=10,
        help="Number of prompts to process in dry-run mode (default: 10)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yaml (default: auto-detect)"
    )
    parser.add_argument(
        "--routing-only",
        action="store_true",
        help="Run routing layer only (no model loading, no inference). Useful for CPU-only systems."
    )
    
    args = parser.parse_args()
    
    # Load config if specified
    if args.config:
        config = get_config()
        config.load_config(args.config)
    
    # Run pipeline
    run_pipeline(args)


if __name__ == "__main__":
    main()
