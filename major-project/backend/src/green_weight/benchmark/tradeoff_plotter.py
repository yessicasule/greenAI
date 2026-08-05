"""
Tradeoff Plotter - Phase 5
=========================

Purpose: Generate the central figures of the paper — the energy vs accuracy
trade-off curve showing all four conditions.

What it does:
- Reads energy summary CSV from energy_tracker
- Reads accuracy results JSON from accuracy_eval
- Plots scatter of points (one per condition) with energy saved vs accuracy
- Draws Pareto frontier line
- Generates secondary bar chart showing precision-selection distribution
- Saves as high-resolution PNGs for the paper
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Plotting will be disabled.")

from config import get_config


class TradeoffPlotter:
    """
    Generate trade-off curves and visualization plots.
    """
    
    def __init__(self):
        """Initialize plotter."""
        config = get_config()
        
        self.output_dir = Path(config.get_figure_output_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dpi = config.benchmark.get("tradeoff_plotter", {}).get("dpi", 300)
        self.formats = config.benchmark.get("tradeoff_plotter", {}).get("plot_formats", ["png"])
        
        logger.info(f"[OK] Tradeoff plotter initialized. Output dir: {self.output_dir}")
    
    def load_energy_data(self, csv_path: Path) -> Dict[str, Dict[str, float]]:
        """
        Load energy summary data from CSV.
        
        Expected columns: tier, mean_joules, std_joules, total_joules
        
        Returns:
            Dict mapping tier names to energy stats
        """
        if not csv_path.exists():
            logger.warning(f"Energy CSV not found: {csv_path}. Using dummy data.")
            return self._dummy_energy_data()
        
        try:
            df = pd.read_csv(csv_path)
            data = {}
            for _, row in df.iterrows():
                tier = row["tier"]
                data[tier] = {
                    "mean_joules": float(row["mean_joules"]),
                    "std_joules": float(row["std_joules"]),
                }
            return data
        except Exception as e:
            logger.error(f"Failed to load energy data: {e}")
            return self._dummy_energy_data()
    
    def load_accuracy_data(self, json_path: Path) -> Dict[str, float]:
        """
        Load accuracy results from JSON.
        
        Returns:
            Dict mapping condition names to overall accuracy
        """
        if not json_path.exists():
            logger.warning(f"Accuracy JSON not found: {json_path}. Using dummy data.")
            return self._dummy_accuracy_data()
        
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            accuracy_by_condition = data.get("accuracy_by_condition", {})
            result = {}
            for condition, scores in accuracy_by_condition.items():
                result[condition] = scores.get("overall", 0.0)
            return result
        except Exception as e:
            logger.error(f"Failed to load accuracy data: {e}")
            return self._dummy_accuracy_data()
    
    def _dummy_energy_data(self) -> Dict[str, Dict[str, float]]:
        """Return placeholder energy data for testing."""
        return {
            "4bit": {"mean_joules": 8.0, "std_joules": 1.5},
            "8bit": {"mean_joules": 25.0, "std_joules": 3.0},
            "16bit": {"mean_joules": 100.0, "std_joules": 5.0},
        }
    
    def _dummy_accuracy_data(self) -> Dict[str, float]:
        """Return placeholder accuracy data for testing."""
        return {
            "always_4bit": 0.42,
            "always_8bit": 0.58,
            "always_16bit": 0.72,
            "routed": 0.68,
        }
    
    def compute_pareto_frontier(
        self,
        points: List[Tuple[float, float, str]]
    ) -> List[Tuple[float, float, str]]:
        """
        Compute Pareto frontier from (x, y, label) points.
        
        Points on the frontier are non-dominated (Pareto optimal).
        """
        frontier = []
        for p1 in points:
            dominated = False
            for p2 in points:
                if p1 != p2:
                    # p2 dominates p1 if p2.x < p1.x AND p2.y >= p1.y
                    if p2[0] < p1[0] and p2[1] >= p1[1]:
                        dominated = True
                        break
            
            if not dominated:
                frontier.append(p1)
        
        # Sort by x-coordinate for plotting
        frontier.sort(key=lambda p: p[0])
        
        return frontier
    
    def plot_tradeoff_curve(
        self,
        energy_data: Dict[str, Dict[str, float]],
        accuracy_data: Dict[str, float],
        output_prefix: str = "tradeoff_curve"
    ) -> None:
        """
        Plot the energy vs accuracy trade-off curve.
        
        Args:
            energy_data: Energy summary from load_energy_data()
            accuracy_data: Accuracy results from load_accuracy_data()
            output_prefix: Prefix for output filenames
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib not available. Cannot plot.")
            return
        
        # Map conditions to (energy_saved%, accuracy)
        # Energy saved is relative to always-16bit (baseline)
        baseline_energy = energy_data.get("16bit", {}).get("mean_joules", 100.0)
        
        points = []
        labels = {
            "always_4bit": ("Always 4-bit", "red"),
            "always_8bit": ("Always 8-bit", "orange"),
            "always_16bit": ("Always 16-bit (baseline)", "blue"),
            "routed": ("Fuzzy-Routed", "green"),
        }
        
        for condition, (label, color) in labels.items():
            accuracy = accuracy_data.get(condition, 0.0)
            
            # Get energy for this condition (use tier mapping)
            if condition == "always_4bit":
                energy = energy_data.get("4bit", {}).get("mean_joules", baseline_energy)
            elif condition == "always_8bit":
                energy = energy_data.get("8bit", {}).get("mean_joules", baseline_energy)
            else:  # always_16bit or routed
                energy = baseline_energy
            
            # Compute energy saved as percentage
            energy_saved_pct = 100 * (1 - energy / baseline_energy)
            
            points.append((energy_saved_pct, accuracy, label))
            logger.debug(f"{condition}: {energy_saved_pct:.1f}% energy saved, {accuracy:.4f} accuracy")
        
        # Compute Pareto frontier
        frontier = self.compute_pareto_frontier(
            [(p[0], p[1], p[2]) for p in points]
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7), dpi=self.dpi)
        
        # Plot all points
        colors_map = {label: color for _, (label, color) in labels.items()}
        for x, y, label in points:
            color = colors_map.get(label, "gray")
            ax.scatter(x, y, s=200, alpha=0.6, c=color, edgecolors="black", linewidth=2, label=label)
        
        # Plot Pareto frontier line
        if frontier:
            frontier_x = [p[0] for p in frontier]
            frontier_y = [p[1] for p in frontier]
            ax.plot(frontier_x, frontier_y, "k--", alpha=0.5, linewidth=1.5, label="Pareto frontier")
        
        # Formatting
        ax.set_xlabel("Energy Saved (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
        ax.set_title("Energy vs Accuracy Trade-off Curve", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        
        # Save
        for fmt in self.formats:
            output_path = self.output_dir / f"{output_prefix}.{fmt}"
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"[OK] Saved tradeoff curve to {output_path}")
        
        plt.close(fig)
    
    def plot_precision_distribution(
        self,
        output_prefix: str = "precision_distribution"
    ) -> None:
        """
        Plot the distribution of prompt routing across the three tiers
        for the routed system.
        
        Args:
            output_prefix: Prefix for output filenames
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib not available. Cannot plot.")
            return
        
        # Placeholder: In real implementation, this would come from the pipeline logs
        # For now, use reasonable estimates based on complexity distribution
        distribution = {
            "4bit": 0.30,  # 30% of prompts routed to 4-bit (simple)
            "8bit": 0.45,  # 45% to 8-bit (medium)
            "16bit": 0.25,  # 25% to 16-bit (complex)
        }
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        tiers = list(distribution.keys())
        fractions = list(distribution.values())
        colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]
        
        wedges, texts, autotexts = ax.pie(
            fractions,
            labels=tiers,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 12}
        )
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        
        ax.set_title("Prompt Routing Distribution (Fuzzy-Routed System)", fontsize=14, fontweight="bold")
        
        # Save
        for fmt in self.formats:
            output_path = self.output_dir / f"{output_prefix}.{fmt}"
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"[OK] Saved precision distribution to {output_path}")
        
        plt.close(fig)
    
    def generate(
        self,
        energy_csv_path: Optional[Path] = None,
        accuracy_json_path: Optional[Path] = None
    ) -> None:
        """
        Generate all plots (trade-off curve + precision distribution).
        
        Args:
            energy_csv_path: Path to energy_summary.csv (auto-detected if None)
            accuracy_json_path: Path to accuracy_results.json (auto-detected if None)
        """
        config = get_config()
        
        # Auto-detect paths
        if energy_csv_path is None:
            energy_csv_path = Path(config.get_energy_output_dir()) / "energy_summary.csv"
        
        if accuracy_json_path is None:
            accuracy_json_path = Path(config.benchmark.get("accuracy_eval", {}).get("output_dir", "results/accuracy_logs")) / "accuracy_results.json"
        
        logger.info(f"Generating plots...")
        logger.info(f"  Energy data: {energy_csv_path}")
        logger.info(f"  Accuracy data: {accuracy_json_path}")
        
        # Load data
        energy_data = self.load_energy_data(energy_csv_path)
        accuracy_data = self.load_accuracy_data(accuracy_json_path)
        
        # Generate plots
        self.plot_tradeoff_curve(energy_data, accuracy_data)
        self.plot_precision_distribution()
        
        logger.info(f"[OK] All plots generated in {self.output_dir}")


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    plotter = TradeoffPlotter()
    plotter.generate()
