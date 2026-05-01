from __future__ import annotations

import torch
from torch import nn

from t24w125.modules import PackedT24Linear, T24LinearSTE
from t24w125.quant import QuantConfig


def test_t24_linear_forward_backward_and_pack():
    torch.manual_seed(0)
    cfg = QuantConfig(scale_group_size=64)
    src = nn.Linear(65, 9, bias=False)
    mod = T24LinearSTE.from_linear(src, cfg)
    x = torch.randn(4, 3, 65)
    y = mod(x).sum()
    y.backward()
    assert mod.weight.grad is not None
    packed = mod.pack()
    ref = PackedT24Linear(packed)
    out = ref(x)
    assert out.shape == (4, 3, 9)
