"""
Configuration loader and manager for green-weight.
Reads config.yaml and provides a singleton interface to access all tunable parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class QuantConfig:
    """Quantization settings for a single tier."""
    enabled: bool
    load_in_8bit: Optional[bool] = None
    load_in_4bit: Optional[bool] = None
    bnb_4bit_quant_type: Optional[str] = None
    bnb_4bit_use_double_quant: Optional[bool] = None
    bnb_4bit_compute_dtype: Optional[str] = None
    compute_dtype: Optional[str] = None
    dtype: Optional[str] = None


class Config:
    """
    Singleton configuration manager.
    Load once at startup, read everywhere.
    """
    
    _instance: Optional['Config'] = None
    _config_data: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize config from config.yaml if not already loaded."""
        if not self._config_data:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml. If None, searches in standard locations.
        """
        if config_path is None:
            # Search for config.yaml in standard locations
            candidates = [
                Path(__file__).parent / "config.yaml",
                Path.cwd() / "config.yaml",
                Path.cwd() / "green_weight" / "config.yaml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    config_path = str(candidate)
                    break
            
            if config_path is None:
                raise FileNotFoundError(
                    "config.yaml not found. Searched: " + ", ".join(str(c) for c in candidates)
                )
        
        with open(config_path, "r") as f:
            self._config_data = yaml.safe_load(f)
        
        print(f"[OK] Loaded config from {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by dot-notation key.
        E.g., get("model.base_model_id") or get("router.fuzzy_controller.tier_thresholds")
        """
        keys = key.split(".")
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    # Convenience accessors for major sections
    
    @property
    def model(self) -> Dict[str, Any]:
        return self._config_data.get("model", {})
    
    @property
    def router(self) -> Dict[str, Any]:
        return self._config_data.get("router", {})
    
    @property
    def cascade(self) -> Dict[str, Any]:
        return self._config_data.get("cascade", {})
    
    @property
    def benchmark(self) -> Dict[str, Any]:
        return self._config_data.get("benchmark", {})
    
    @property
    def pipeline(self) -> Dict[str, Any]:
        return self._config_data.get("pipeline", {})
    
    @property
    def hardware(self) -> Dict[str, Any]:
        return self._config_data.get("hardware", {})
    
    # Model-specific accessors
    
    def get_model_id(self) -> str:
        """Get the base HuggingFace model ID."""
        return self.model.get("base_model_id", "meta-llama/Llama-2-7b-hf")
    
    def get_quantization_config(self, tier: str) -> QuantConfig:
        """
        Get quantization settings for a specific tier (4bit, 8bit, or 16bit).
        
        Args:
            tier: "4bit", "8bit", or "16bit"
        
        Returns:
            QuantConfig object with the quantization parameters.
        """
        quant_dict = self.model.get("quantization", {}).get(tier, {})
        return QuantConfig(**quant_dict)
    
    def is_lazy_16bit(self) -> bool:
        """Check if lazy loading of 16-bit model is enabled."""
        return self.model.get("lazy_16bit", False)
    
    def get_device_map(self) -> Optional[str]:
        """Get the device map for model loading."""
        return self.model.get("device_map", "auto")
    
    def get_max_memory(self) -> Optional[Dict[str, str]]:
        """Get max memory per GPU if specified."""
        return self.model.get("max_memory_per_gpu")
    
    # Router-specific accessors
    
    def get_router_type(self) -> str:
        """Get RouteLLM router type (mf, bert, or sw_ranking)."""
        return self.router.get("routellm", {}).get("router_type", "mf")
    
    def get_mf_checkpoint(self) -> str:
        """Get RouteLLM MF checkpoint path."""
        return self.router.get("routellm", {}).get("mf_checkpoint", "routellm/mf_gpt4_augmented")
    
    def get_fuzzy_tier_thresholds(self) -> Dict[str, int]:
        """Get tier assignment thresholds from fuzzy controller."""
        return self.router.get("fuzzy_controller", {}).get("tier_thresholds", {
            "4bit_upper": 33,
            "8bit_upper": 66,
            "16bit_lower": 67,
        })
    
    def get_routing_zone_boundaries(self) -> Dict[str, float]:
        """Return normalized [0-1] mid-zone bounds used by RouteLLMBridge."""
        thresholds = self.get_fuzzy_tier_thresholds()
        return {
            "mid_zone_lower": thresholds.get("4bit_upper", 33) / 100.0,
            "mid_zone_upper": thresholds.get("8bit_upper", 66) / 100.0,
        }

    def get_fuzzy_membership_breakpoints(self, feature: str) -> list:
        """
        Get membership function breakpoints for a complexity feature.
        
        Args:
            feature: One of "flesch_kincaid", "token_length", "entropy", "syntax_depth"
        
        Returns:
            List of breakpoint values for LOW/MID/HIGH transitions.
        """
        key = f"{feature}_breakpoints"
        return self.router.get("fuzzy_controller", {}).get(key, [])
    
    # Cascade-specific accessors
    
    def get_cascade_service_order(self) -> list:
        """Get the cascade escalation order (service names)."""
        return self.cascade.get("service_order", ["local/4bit", "local/8bit", "local/16bit"])
    
    def get_judger_threshold(self) -> float:
        """Get the generation judger stop threshold."""
        return self.cascade.get("judger", {}).get("threshold", 0.5)
    
    def set_judger_threshold(self, threshold: float) -> None:
        """Set the judger threshold (e.g., after optimization)."""
        if "cascade" not in self._config_data:
            self._config_data["cascade"] = {}
        if "judger" not in self._config_data["cascade"]:
            self._config_data["cascade"]["judger"] = {}
        self._config_data["cascade"]["judger"]["threshold"] = threshold
    
    # Benchmark-specific accessors
    
    def get_energy_output_dir(self) -> str:
        """Get energy tracking output directory."""
        path = self.benchmark.get("energy_tracker", {}).get("output_dir", "results/energy_logs")
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_eval_dataset_path(self) -> str:
        """Get evaluation dataset path."""
        return self.benchmark.get("eval_dataset", {}).get("path", "data/eval_prompts.jsonl")
    
    def get_benchmark_tasks(self) -> list:
        """Get list of lm-eval benchmark tasks."""
        return self.benchmark.get("accuracy_eval", {}).get("tasks", ["mmlu"])
    
    def get_figure_output_dir(self) -> str:
        """Get output directory for plots."""
        path = self.benchmark.get("tradeoff_plotter", {}).get("output_dir", "results/figures")
        os.makedirs(path, exist_ok=True)
        return path
    
    # Pipeline-specific accessors
    
    def is_dry_run(self) -> bool:
        """Check if dry-run mode is enabled."""
        return self.pipeline.get("dry_run", False)
    
    def get_dry_run_count(self) -> int:
        """Get number of prompts to process in dry-run mode."""
        return self.pipeline.get("dry_run_prompts", 10)
    
    def get_log_output_dir(self) -> str:
        """Get pipeline log output directory."""
        path = self.pipeline.get("log_output_dir", "results/pipeline_logs")
        os.makedirs(path, exist_ok=True)
        return path
    
    def should_warmup(self) -> bool:
        """Check if model warmup is enabled."""
        return self.pipeline.get("warmup_enabled", True)
    
    def get_gpu_device(self) -> int:
        """Get GPU device index."""
        return self.hardware.get("gpu_device", 0)
    
    def get_seed(self) -> int:
        """Get random seed."""
        return self.hardware.get("seed", 42)


def get_config() -> Config:
    """
    Convenience function to get the singleton Config instance.
    """
    return Config()
