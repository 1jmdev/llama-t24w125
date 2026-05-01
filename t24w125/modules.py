from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .quant import QuantConfig, pack_t24, quantize_t24_ste, unpack_t24


class T24LinearSTE(nn.Module):
    """Training-time fake-quantized linear layer.

    Keeps a trainable dense master weight and uses T2:4 ternary quantization in the
    forward pass with STE gradients. Export replaces this with packed weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        cfg: QuantConfig | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cfg = cfg or QuantConfig()
        self.register_buffer("quant_alpha", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear: nn.Linear, cfg: QuantConfig) -> "T24LinearSTE":
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            cfg=cfg,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        mod.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and mod.bias is not None:
            mod.bias.data.copy_(linear.bias.data)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = quantize_t24_ste(self.weight, self.cfg)
        alpha = self.quant_alpha.to(device=self.weight.device, dtype=self.weight.dtype)
        qw = self.weight + alpha * (q - self.weight)
        return F.linear(x, qw, self.bias)

    @torch.no_grad()
    def pack(self) -> dict:
        payload = pack_t24(self.weight, self.cfg)
        if self.bias is not None:
            payload["bias"] = self.bias.detach().cpu()
        return payload


class PackedT24Linear(nn.Module):
    """Reference inference module using packed T24 weights.

    This is intentionally simple. It verifies correctness and gives a clean loading
    path; production serving should replace the dense dequant step with a fused CUDA
    or Triton kernel.
    """

    def __init__(self, packed: dict, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.packed = packed
        self.in_features = int(packed["orig_in_features"])
        self.out_features = int(packed["orig_out_features"])
        if bias is None and "bias" in packed:
            bias = packed["bias"]
        self.register_buffer("bias", bias.float() if bias is not None else None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = unpack_t24(self.packed, device=x.device).to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.linear(x, w, b)


def _should_replace(name: str, module: nn.Module, skip_names: Iterable[str]) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if module.bias is not None:
        return False
    if any(s in name for s in skip_names):
        return False
    return True


def replace_linear_with_t24(
    model: nn.Module,
    cfg: QuantConfig,
    skip_names: Iterable[str] = ("lm_head", "embed_tokens"),
) -> int:
    """Replace eligible nn.Linear modules in-place with T24LinearSTE."""
    replaced = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            fq_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if _should_replace(fq_name, child, skip_names):
                setattr(parent, child_name, T24LinearSTE.from_linear(child, cfg))
                replaced += 1
    return replaced


def iter_t24_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, T24LinearSTE):
            yield name, module


@torch.no_grad()
def set_t24_alpha(model: nn.Module, alpha: float) -> None:
    alpha = max(0.0, min(1.0, float(alpha)))
    for _, module in iter_t24_modules(model):
        module.quant_alpha.fill_(alpha)
