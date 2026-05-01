from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer


@torch.no_grad()
def zeropower_via_newtonschulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate matrix sign / orthogonalized update used by Muon.

    This compact implementation follows the common Newton-Schulz form used in
    public Muon examples. It works on 2D matrices; callers flatten higher dims.
    """
    if g.ndim != 2:
        raise ValueError("zeropower_via_newtonschulz expects a 2D tensor")
    dtype = g.dtype
    x = g.float()
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    else:
        transposed = False
    x = x / (x.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=dtype)


class Muon(Optimizer):
    """Minimal Muon optimizer for matrix-like hidden-layer parameters.

    Use AdamW for vectors, embeddings, norms, and output heads. This class is for
    2D hidden weights only; 1D tensors are skipped.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
    ) -> None:
        defaults: dict[str, Any] = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None or p.ndim < 2:
                    continue
                g = p.grad.detach()
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                if update.ndim > 2:
                    update = update.flatten(1)
                    original_shape = p.shape
                else:
                    original_shape = None
                update = zeropower_via_newtonschulz(update, steps=ns_steps)
                if original_shape is not None:
                    update = update.view(original_shape)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr * math.sqrt(max(1, p.size(0)) / max(1, p.size(1))))
        return loss


def split_muon_adamw_params(model: torch.nn.Module):
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lowered = name.lower()
        if p.ndim >= 2 and all(k not in lowered for k in ("embed", "lm_head", "scales")):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params
