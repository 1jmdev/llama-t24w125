#!/usr/bin/env python3
from __future__ import annotations

import json

import torch

from t24w125.modules import T24LinearSTE
from t24w125.quant import QuantConfig, pack_t24, quantize_t24_dense, theoretical_bits_per_weight, unpack_t24
from t24w125.muon import Muon


def main() -> None:
    torch.manual_seed(0)
    cfg = QuantConfig(scale_group_size=128, scale_dtype="float16")
    w = torch.randn(17, 257)
    q = quantize_t24_dense(w, cfg)
    packed = pack_t24(w, cfg)
    u = unpack_t24(packed)
    assert q.shape == w.shape
    assert u.shape == w.shape
    assert torch.allclose(q, u, atol=1e-3, rtol=1e-3)

    lin = T24LinearSTE(257, 17, cfg=cfg)
    x = torch.randn(2, 8, 257)
    y = lin(x).sum()
    y.backward()
    opt = Muon([lin.weight], lr=1e-3)
    opt.step()

    print(json.dumps({
        "ok": True,
        "format": "T24W125",
        "bits_per_weight_raw": 1.25,
        "bits_per_weight_with_fp16_scales": theoretical_bits_per_weight(128),
        "packed_codes_u8_shape": list(packed["codes_u8"].shape),
        "packed_5bit_bytes": int(packed["codes_5bit"].numel()),
    }, indent=2))


if __name__ == "__main__":
    main()
