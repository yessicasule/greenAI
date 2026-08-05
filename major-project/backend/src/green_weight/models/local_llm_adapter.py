"""
Local LLM Adapter
=================

Purpose: Provides a unified interface between the three quantized model tiers
and the FrugalGPT cascade / energy tracker.

LocalLLMforAll is the single class that:
- Holds references to loaded models (set by model_pool after loading)
- Maps service names like "local/4bit" to the correct model + tokenizer
- Exposes get_completion(prompt, service_name, genparams) used by frugal_cascade
- Exposes list_services() used by frugal_cascade to know escalation order
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Service order matches cascade config
SERVICE_ORDER = ["local/4bit", "local/8bit", "local/16bit"]

# Module-level registry: populated by model_pool.load_pool()
_model_registry: Dict[str, Any] = {}   # service_name -> {"model": ..., "tokenizer": ...}


def register_model(service_name: str, model, tokenizer) -> None:
    """Called by model_pool to register a loaded model."""
    _model_registry[service_name] = {"model": model, "tokenizer": tokenizer}
    logger.info(f"[OK] Registered model for service: {service_name}")


class LocalLLMforAll:
    """
    Unified adapter over all three quantized model tiers.

    Used by:
    - frugal_cascade.py  -> get_completion(), list_services()
    - energy_tracker.py  -> via frugal_cascade
    """

    def list_services(self) -> list:
        """Return available service names in cascade order."""
        available = [s for s in SERVICE_ORDER if s in _model_registry]
        if not available:
            logger.warning("No models registered yet. Call model_pool.load_pool() first.")
        return available

    def get_completion(
        self,
        prompt: str,
        service_name: str = "local/8bit",
        genparams: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a completion using the specified service/tier.

        Args:
            prompt: Input text
            service_name: One of "local/4bit", "local/8bit", "local/16bit"
            genparams: Generation parameters dict, e.g. {"max_tokens": 256}

        Returns:
            Generated response string
        """
        genparams = genparams or {}
        max_tokens = genparams.get("max_tokens", 256)

        # If no models are loaded (routing-only mode), return placeholder
        if service_name not in _model_registry:
            logger.warning(
                f"Model '{service_name}' not loaded. "
                "Running in routing-only mode — returning placeholder response."
            )
            return f"[routing-only mode — no inference for {service_name}]"

        entry = _model_registry[service_name]
        model = entry["model"]
        tokenizer = entry["tokenizer"]

        try:
            import torch

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=genparams.get("temperature", 0.7),
                    top_p=genparams.get("top_p", 0.95),
                    do_sample=genparams.get("do_sample", True),
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens (skip the prompt)
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            logger.debug(f"[{service_name}] Generated {len(generated_ids)} tokens")
            return response.strip()

        except Exception as e:
            logger.error(f"Inference failed for {service_name}: {e}")
            return f"[inference error: {e}]"

    def get_next_service(self, current_service: str) -> Optional[str]:
        """
        Return the next tier up from current_service for cascade escalation.

        Returns None if already at the top tier.
        """
        available = self.list_services()
        try:
            idx = available.index(current_service)
            if idx + 1 < len(available):
                return available[idx + 1]
        except ValueError:
            pass
        return None