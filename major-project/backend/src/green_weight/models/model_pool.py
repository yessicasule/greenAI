"""
Model Pool
==========

Purpose: Load and cache all three quantized model tiers at startup.

What it does:
- Reads model config from config.yaml (base_model_id, quantization settings)
- Loads 4-bit model using BitsAndBytes NF4 quantization
- Loads 8-bit model using BitsAndBytes int8 quantization
- Loads 16-bit model in bfloat16 (optionally lazy if lazy_16bit=true in config)
- Registers each loaded model with LocalLLMforAll via register_model()
- Exposes load_pool() and warmup() called by run_pipeline.py
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available")

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("peft not available — adapters will not be loaded")

from config import get_config
from models.local_llm_adapter import register_model


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment."""
    import os
    return os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")


def _load_tokenizer(model_id: str, hf_token: Optional[str]):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_4bit(model_id: str, hf_token: Optional[str], adapter_path: Optional[str] = None):
    """Load model in 4-bit NF4 quantization."""
    logger.info(f"Loading 4-bit model: {model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )

    if adapter_path and PEFT_AVAILABLE:
        logger.info(f"  Loading 4-bit adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    logger.info("[OK] 4-bit model loaded")
    return model


def _load_8bit(model_id: str, hf_token: Optional[str], adapter_path: Optional[str] = None):
    """Load model in 8-bit int8 quantization."""
    logger.info(f"Loading 8-bit model: {model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )

    if adapter_path and PEFT_AVAILABLE:
        logger.info(f"  Loading 8-bit adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    logger.info("[OK] 8-bit model loaded")
    return model


def _load_16bit(model_id: str, hf_token: Optional[str], adapter_path: Optional[str] = None):
    """Load model in bfloat16 full precision."""
    logger.info(f"Loading 16-bit model: {model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )

    if adapter_path and PEFT_AVAILABLE:
        logger.info(f"  Loading 16-bit adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    logger.info("[OK] 16-bit model loaded")
    return model


def load_pool(lazy_16bit: bool = False) -> None:
    """
    Load all three quantized model tiers and register them with LocalLLMforAll.

    Args:
        lazy_16bit: If True, skip loading the 16-bit model at startup.
                    It will be loaded on first use. Useful for 16GB GPUs.

    Called by run_pipeline.py Phase 1.
    """
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        logger.error("torch/transformers not available. Cannot load models.")
        return

    if not torch.cuda.is_available():
        logger.error(
            "No CUDA GPU detected. Model loading requires a GPU. "
            "Use --routing-only flag to run without models."
        )
        return

    config = get_config()
    model_id = config.get_model_id()
    hf_token = _get_hf_token()

    if not hf_token:
        logger.warning(
            "No HuggingFace token found. Set HUGGING_FACE_HUB_TOKEN env var. "
            "Gated models (e.g. Llama-2) will fail without it."
        )

    from pathlib import Path
    # major-project/ is 5 levels up from this file
    # (models/model_pool.py -> green_weight/ -> src/ -> backend/ -> major-project/);
    # adapters/ lives at major-project/adapters/, not inside backend/src/green_weight/
    # (confirmed 2026-08-22 — the old `parent.parent` + adapter_4bit/8bit/16bit naming
    # pointed at a path that never existed, so load_pool() was silently never loading
    # any QAT adapter here). Directory names match the tier->adapter mapping used by
    # training/scripts/kaggle_routing_experiment.py and kaggle_accuracy_eval.py.
    project_root = Path(__file__).resolve().parents[4]
    adapter_4bit = project_root / "adapters" / "adapter_simple"
    adapter_8bit = project_root / "adapters" / "adapter_medium"
    adapter_16bit = project_root / "adapters" / "adapter_complex"

    adapter_4bit  = str(adapter_4bit)  if adapter_4bit.exists()  else None
    adapter_8bit  = str(adapter_8bit)  if adapter_8bit.exists()  else None
    adapter_16bit = str(adapter_16bit) if adapter_16bit.exists() else None

    tokenizer = _load_tokenizer(model_id, hf_token)

    try:
        model_4bit = _load_4bit(model_id, hf_token, adapter_4bit)
        register_model("local/4bit", model_4bit, tokenizer)
    except Exception as e:
        logger.error(f"Failed to load 4-bit model: {e}")

    try:
        model_8bit = _load_8bit(model_id, hf_token, adapter_8bit)
        register_model("local/8bit", model_8bit, tokenizer)
    except Exception as e:
        logger.error(f"Failed to load 8-bit model: {e}")

    if not lazy_16bit:
        try:
            model_16bit = _load_16bit(model_id, hf_token, adapter_16bit)
            register_model("local/16bit", model_16bit, tokenizer)
        except Exception as e:
            logger.error(f"Failed to load 16-bit model: {e}")
    else:
        logger.info("16-bit model skipped (lazy_16bit=true). Will load on first use.")


def warmup(num_prompts: int = 5) -> None:
    """
    Run a few dummy prompts through each loaded model to warm up CUDA kernels.

    Args:
        num_prompts: Number of warmup prompts per tier (default: 5)
    """
    from models.local_llm_adapter import LocalLLMforAll

    llm = LocalLLMforAll()
    services = llm.list_services()

    if not services:
        logger.warning("No models loaded for warmup. Skipping.")
        return

    warmup_prompt = "Hello, how are you?"
    logger.info(f"Warming up {len(services)} models with {num_prompts} prompts each...")

    for service in services:
        for i in range(num_prompts):
            try:
                llm.get_completion(warmup_prompt, service_name=service, genparams={"max_tokens": 16})
            except Exception as e:
                logger.warning(f"Warmup failed for {service}: {e}")
                break

    logger.info("[OK] Warmup complete")


def infer(prompt: str, tier: str = "8bit", max_new_tokens: int = 256) -> str:
    """
    Convenience function: run inference on a single prompt at a given tier.

    Args:
        prompt: Input text
        tier: "4bit", "8bit", or "16bit"
        max_new_tokens: Max tokens to generate

    Returns:
        Generated response string
    """
    from models.local_llm_adapter import LocalLLMforAll
    llm = LocalLLMforAll()
    return llm.get_completion(
        prompt,
        service_name=f"local/{tier}",
        genparams={"max_tokens": max_new_tokens}
    )