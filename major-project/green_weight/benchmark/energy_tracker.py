"""
Energy Tracker - Phase 5
=======================

Purpose: Measure real energy consumption per inference call using CodeCarbon,
broken down by tier.

What it does:
- Wraps frugal_cascade.run() with CodeCarbon EmissionsTracker
- Records delta in joules per inference, along with metadata
- Computes summary statistics per tier (mean, std dev, joules-per-token)
- Outputs CSVs for the energy X-axis of the trade-off curve
"""

import logging
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Try to import CodeCarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    logger.warning("CodeCarbon not installed. Energy tracking will be disabled.")

from config import get_config
from cascade.frugal_cascade import run_cascade


class EnergyTracker:
    """
    Track energy consumption per inference using CodeCarbon.
    """
    
    def __init__(self):
        """Initialize energy tracker."""
        config = get_config()
        
        self.output_dir = Path(config.get_energy_output_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Energy log: list of (prompt, tier, joules, response_len, prompt_len)
        self.energy_log: List[Dict[str, Any]] = []
        
        # Per-tier statistics
        self.tier_stats: Dict[str, List[float]] = {
            "4bit": [],
            "8bit": [],
            "16bit": [],
        }
        
        logger.info(f"[OK] Energy tracker initialized. Output dir: {self.output_dir}")
    
    def track_inference(
        self,
        prompt: str,
        tier: str,
        max_new_tokens: int = 256,
        country_code: str = "USA"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run cascade inference and track energy consumption.
        
        Args:
            prompt: Input prompt
            tier: Starting tier
            max_new_tokens: Maximum tokens to generate
            country_code: Country code for CodeCarbon (e.g., "USA")
        
        Returns:
            Tuple of (response, energy_info)
            energy_info contains: joules, carbon_g, duration_s
        """
        if not CODECARBON_AVAILABLE:
            logger.warning("CodeCarbon not available. Running inference without energy tracking.")
            response, metadata = run_cascade(prompt, tier, max_new_tokens)
            return response, {
                "joules": 0.0,
                "carbon_g": 0.0,
                "duration_s": 0.0,
                "tracked": False
            }
        
        config = get_config()
        country = config.benchmark.get("energy_tracker", {}).get("country_code", country_code)
        
        try:
            # Create tracker
            tracker = EmissionsTracker(country_iso_code=country)
            tracker.start()
            
            # Run cascade
            start_time = datetime.now()
            response, metadata = run_cascade(prompt, tier, max_new_tokens)
            end_time = datetime.now()
            
            # Stop tracker and get emissions
            emissions_kg = tracker.stop()
            
            # Convert to joules (rough approximation from emissions)
            # Carbon intensity varies by country; for now use rough estimate
            # 1 kg CO2 ≈ 0.17 kWh = 612 kJ (approximate)
            carbon_g = emissions_kg * 1000 if emissions_kg else 0.0
            joules = carbon_g * 612 if carbon_g else 0.0  # Approximate
            
            duration_s = (end_time - start_time).total_seconds()
            
            energy_info = {
                "joules": joules,
                "carbon_g": carbon_g,
                "duration_s": duration_s,
                "tracked": True
            }
            
        except Exception as e:
            logger.error(f"Energy tracking failed: {e}")
            response, metadata = run_cascade(prompt, tier, max_new_tokens)
            energy_info = {
                "joules": 0.0,
                "carbon_g": 0.0,
                "duration_s": 0.0,
                "tracked": False,
                "error": str(e)
            }
        
        # Log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "prompt_len": len(prompt),
            "tier": tier,
            "response": response,
            "response_len": len(response),
            "joules": energy_info["joules"],
            "carbon_g": energy_info["carbon_g"],
            "duration_s": energy_info["duration_s"],
            "cascade_metadata": metadata
        }
        
        self.energy_log.append(log_entry)
        
        # Track per-tier statistics
        if energy_info["tracked"]:
            self.tier_stats[tier].append(energy_info["joules"])
        
        logger.debug(
            f"Tracked {tier}: {energy_info['joules']:.1f}J, "
            f"{energy_info['carbon_g']:.4f}g CO2, {energy_info['duration_s']:.2f}s"
        )
        
        return response, energy_info
    
    def save_logs(self) -> None:
        """Save energy logs to CSV files."""
        # Detailed log (one row per inference)
        detailed_csv = self.output_dir / "energy_detailed.csv"
        
        if self.energy_log:
            with open(detailed_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "timestamp", "tier", "prompt_len", "response_len",
                    "joules", "carbon_g", "duration_s"
                ])
                writer.writeheader()
                
                for entry in self.energy_log:
                    writer.writerow({
                        "timestamp": entry["timestamp"],
                        "tier": entry["tier"],
                        "prompt_len": entry["prompt_len"],
                        "response_len": entry["response_len"],
                        "joules": entry["joules"],
                        "carbon_g": entry["carbon_g"],
                        "duration_s": entry["duration_s"],
                    })
            
            logger.info(f"[OK] Saved detailed energy log to {detailed_csv}")
        
        # Summary statistics (one row per tier)
        summary_csv = self.output_dir / "energy_summary.csv"
        
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "tier", "count", "mean_joules", "std_joules",
                "total_joules", "mean_carbon_g", "joules_per_token"
            ])
            writer.writeheader()
            
            for tier in ["4bit", "8bit", "16bit"]:
                if self.tier_stats[tier]:
                    energies = self.tier_stats[tier]
                    count = len(energies)
                    mean = sum(energies) / count
                    std = (sum((e - mean) ** 2 for e in energies) / count) ** 0.5
                    total = sum(energies)
                    
                    # Estimate joules per output token (rough approximation)
                    # Assuming ~50 avg output tokens
                    joules_per_token = mean / 50.0 if mean > 0 else 0.0
                    
                    writer.writerow({
                        "tier": tier,
                        "count": count,
                        "mean_joules": mean,
                        "std_joules": std,
                        "total_joules": total,
                        "mean_carbon_g": 0.0,  # Placeholder
                        "joules_per_token": joules_per_token,
                    })
        
        logger.info(f"[OK] Saved energy summary to {summary_csv}")
    
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Return summary statistics per tier."""
        stats = {}
        for tier in ["4bit", "8bit", "16bit"]:
            if self.tier_stats[tier]:
                energies = self.tier_stats[tier]
                count = len(energies)
                mean = sum(energies) / count
                std = (sum((e - mean) ** 2 for e in energies) / count) ** 0.5
                
                stats[tier] = {
                    "count": count,
                    "mean_joules": mean,
                    "std_joules": std,
                    "total_joules": sum(energies),
                }
            else:
                stats[tier] = {
                    "count": 0,
                    "mean_joules": 0.0,
                    "std_joules": 0.0,
                    "total_joules": 0.0,
                }
        
        return stats


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    tracker = EnergyTracker()
    
    test_prompts = [
        ("What is 2+2?", "4bit"),
        ("Explain machine learning.", "8bit"),
    ]
    
    for prompt, tier in test_prompts:
        response, energy_info = tracker.track_inference(prompt, tier)
        print(f"\n{tier}: {energy_info['joules']:.1f}J")
        print(f"Response: {response[:80]}...")
    
    tracker.save_logs()
    print("\nSummary stats:")
    print(tracker.get_summary_stats())
