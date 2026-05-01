from __future__ import annotations

import torch

from t24w125.quant import QuantConfig, pack_5bit_codes, pack_t24, quantize_t24_dense, theoretical_bits_per_weight, unpack_5bit_codes, unpack_t24


def test_5bit_roundtrip():
    codes = torch.arange(256, dtype=torch.uint8) & 0x1F
    bs = pack_5bit_codes(codes)
    out = unpack_5bit_codes(bs, codes.numel())
    assert torch.equal(codes, out)


def test_t24_pack_roundtrip_shape_and_values():
    torch.manual_seed(42)
    cfg = QuantConfig(scale_group_size=128)
    w = torch.randn(31, 257)
    dense = quantize_t24_dense(w, cfg)
    packed = pack_t24(w, cfg)
    unpacked = unpack_t24(packed)
    assert unpacked.shape == w.shape
    assert torch.allclose(dense, unpacked, atol=1e-3, rtol=1e-3)


def test_exact_two_nonzeros_per_four():
    torch.manual_seed(7)
    cfg = QuantConfig(scale_group_size=64)
    w = torch.randn(8, 64)
    dense = quantize_t24_dense(w, cfg).view(8, 16, 4)
    nnz = dense.ne(0).sum(dim=-1)
    assert torch.equal(nnz, torch.full_like(nnz, 2))


def test_bits_per_weight():
    assert theoretical_bits_per_weight(128) == 1.375
