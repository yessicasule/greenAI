"""Quantization-Aware Training (QAT) for bit-resilient models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import nn


@dataclass
class QATConfig:
    """Configuration for QAT training."""

    bits: int
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    quantize_every_n_steps: int = 10


class FakeQuantizer:
    """
    Fake quantization for QAT.

    Simulates low-precision arithmetic during training while
    keeping weights in high precision for gradient updates.
    """

    def __init__(self, bits: int):
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to tensor."""
        # Find scale factor
        x_min = x.min()
        x_max = x.max()
        scale = (x_max - x_min) / (self.qmax - self.qmin)

        if scale == 0:
            return x

        # Quantize and dequantize (fake quantization)
        x_quant = torch.clamp(
            torch.round(x / scale),
            self.qmin,
            self.qmax
        )
        return x_quant * scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization with straight-through estimator."""
        if not self.training:
            return self.quantize(x)

        # Straight-through estimator: forward uses quantized, backward uses full precision
        return x + (self.quantize(x) - x).detach()


class BitResilientTrainer:
    """
    Trainer for bit-resilient models using QAT.

    Trains the model to be robust to different quantization levels
    by cycling through bit-widths during training.
    """

    def __init__(
        self,
        model: nn.Module,
        bit_widths: tuple[int, ...] = (4, 8, 16),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.bit_widths = bit_widths
        self.device = device
        self.quantizers = {bits: FakeQuantizer(bits) for bits in bit_widths}
        self.current_bit_idx = 0

    def _get_current_bit_width(self) -> int:
        """Get current training bit-width (cycles through)."""
        return self.bit_widths[self.current_bit_idx % len(self.bit_widths)]

    def _apply_quantization(self, module: nn.Module, bits: int) -> None:
        """Apply fake quantization to module weights."""
        quantizer = self.quantizers[bits]
        for param in module.parameters():
            if param.requires_grad:
                param.data = quantizer(param.data)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> dict[str, float]:
        """
        Single training step with QAT.

        Args:
            batch: Training batch
            optimizer: Optimizer instance
            step: Global step number

        Returns:
            Dict with loss and metrics
        """
        self.model.train()

        # Cycle bit-width every N steps
        current_bits = self._get_current_bit_width()
        if step > 0 and step % 10 == 0:
            self.current_bit_idx += 1

        # Apply fake quantization
        self._apply_quantization(self.model, current_bits)

        # Forward pass
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)

        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            "bit_width": float(current_bits),
            "step": float(step),
        }

    def save_adapter(self, path: Path, bit_width: int) -> None:
        """Save LoRA adapter for specific bit-width."""
        adapter_path = path / f"adapter_{bit_width}bit"
        adapter_path.mkdir(parents=True, exist_ok=True)

        # Save only trainable parameters (LoRA)
        state_dict = {
            k: v for k, v in self.model.state_dict().items()
            if "lora" in k.lower()
        }
        torch.save(state_dict, adapter_path / "adapter_model.pt")

        print(f"Saved {bit_width}-bit adapter to {adapter_path}")

    def train_epoch(
        self,
        dataloader: Iterator[dict],
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> dict[str, float]:
        """Train for one epoch."""
        total_loss = 0.0
        num_steps = 0

        for batch in dataloader:
            metrics = self.training_step(batch, optimizer, num_steps)
            total_loss += metrics["loss"]
            num_steps += 1

        return {
            "epoch": epoch,
            "avg_loss": total_loss / max(num_steps, 1),
            "steps": num_steps,
        }
