"""
Tests for config.py — loads backend/src/green_weight/config.yaml (the real
file, not a fixture copy) via the Config singleton and checks the
convenience accessors against its actual current contents.
"""

import pytest

from config import get_config, Config


def test_get_config_loads_without_error():
    config = get_config()
    assert config is not None
    assert isinstance(config, Config)


def test_get_config_is_a_singleton():
    assert get_config() is get_config()


def test_get_model_id():
    assert get_config().get_model_id() == "meta-llama/Llama-3.2-1B"


def test_is_lazy_16bit():
    assert get_config().is_lazy_16bit() is False


def test_get_benchmark_tasks():
    assert get_config().get_benchmark_tasks() == ["mmlu", "hellaswag"]


def test_get_fuzzy_tier_thresholds():
    thresholds = get_config().get_fuzzy_tier_thresholds()
    assert thresholds == {"4bit_upper": 33, "8bit_upper": 66, "16bit_lower": 67}


def test_get_routing_zone_boundaries():
    zones = get_config().get_routing_zone_boundaries()
    assert zones == {"mid_zone_lower": pytest.approx(0.33), "mid_zone_upper": pytest.approx(0.66)}


@pytest.mark.parametrize("feature,expected", [
    # Trimmed to 2 values each 2026-08-22: _trimf_low_mid_high() only ever
    # reads bps[0]/bps[1] (MEDIUM's peak is their auto-computed midpoint) —
    # a 3rd "HIGH transition" value was silently inert since that function
    # was written, so config.yaml's lists were cut to match actual
    # behavior instead of carrying a misleading dead number. See
    # CREDIBILITY_REPORT.md / NEW.md Phase 4.
    ("flesch_kincaid", [6, 12]),
    ("token_length", [0.2, 0.5]),
    ("entropy", [3.8, 4.1]),
    ("syntax_depth", [4, 5]),
])
def test_get_fuzzy_membership_breakpoints(feature, expected):
    assert get_config().get_fuzzy_membership_breakpoints(feature) == expected


def test_get_fuzzy_membership_breakpoints_unknown_feature_returns_empty():
    assert get_config().get_fuzzy_membership_breakpoints("nonexistent_feature") == []


def test_get_dot_notation_lookup():
    config = get_config()
    assert config.get("model.base_model_id") == "meta-llama/Llama-3.2-1B"
    assert config.get("nonexistent.key", "default") == "default"


def test_get_quantization_config_4bit():
    quant = get_config().get_quantization_config("4bit")
    assert quant.enabled is True
    assert quant.bnb_4bit_quant_type == "nf4"


def test_get_cascade_service_order():
    assert get_config().get_cascade_service_order() == ["local/4bit", "local/8bit", "local/16bit"]
