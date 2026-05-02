from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


PAIR_TO_CODE = {
    (0, 1): 0,
    (0, 2): 1,
    (0, 3): 2,
    (1, 2): 3,
    (1, 3): 4,
    (2, 3): 5,
}
CODE_TO_PAIR = torch.tensor(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=torch.long
)


@dataclass(frozen=True)
class QuantConfig:
    scale_group_size: int = 128
    eps: float = 1e-8
    scale_dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.scale_group_size < 4 or self.scale_group_size % 4 != 0:
            raise ValueError("scale_group_size must be a multiple of 4 and >= 4")
        if self.scale_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("scale_dtype must be float16, bfloat16, or float32")

    @property
    def torch_scale_dtype(self) -> torch.dtype:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            self.scale_dtype
        ]


def _pad_in_features(weight: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int]:
    pad = (-weight.shape[1]) % multiple
    if pad:
        weight = F.pad(weight, (0, pad))
    return weight, pad


def quantize_t24_ste(weight: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """Fake-quantize to T2:4 ternary using a straight-through estimator.

    Forward uses quantized values. Backward passes gradients to the original weight.
    """
    q = quantize_t24_dense(weight, cfg)
    return weight + (q - weight).detach()


@torch.no_grad()
def quantize_t24_dense(weight: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """Return dense dequantized T2:4 tensor with the same shape as weight."""
    orig_shape = weight.shape
    w, pad = _pad_in_features(weight.detach().float(), cfg.scale_group_size)
    out_features, in_padded = w.shape
    groups = in_padded // cfg.scale_group_size
    blocks_per_group = cfg.scale_group_size // 4
    wb = w.view(out_features, groups, blocks_per_group, 4)

    idx = wb.abs().topk(k=2, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(wb, dtype=torch.bool).scatter_(-1, idx, True)
    signed = torch.where(mask, wb.sign(), torch.zeros_like(wb))

    numerator = (wb * signed).sum(dim=(-1, -2), keepdim=True)
    denominator = signed.abs().sum(dim=(-1, -2), keepdim=True).clamp_min(cfg.eps)
    scale = numerator / denominator
    scale = scale.clamp_min(cfg.eps)
    deq = (signed * scale).view(out_features, in_padded)
    if pad:
        deq = deq[:, :-pad]
    return deq.to(dtype=weight.dtype).view(orig_shape)


@torch.no_grad()
def pack_t24(
    weight: torch.Tensor,
    cfg: QuantConfig,
    scale_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int | str]:
    """Pack a dense weight into T2:4 codes.

    Each four-weight block is encoded as one 5-bit logical code:
      bits 0..2: nonzero pair index among six possible pairs
      bits 3..4: signs for the two nonzero values, 1 means negative

    The function returns both byte-per-code and true 5-bit bitstream forms.
    """
    if weight.ndim != 2:
        raise ValueError("pack_t24 expects a 2D weight tensor")

    orig_out, orig_in = weight.shape
    w, pad = _pad_in_features(weight.detach().float(), cfg.scale_group_size)
    out_features, in_padded = w.shape
    scale_groups = in_padded // cfg.scale_group_size
    blocks_per_scale_group = cfg.scale_group_size // 4
    total_blocks_per_row = in_padded // 4

    wb = w.view(out_features, scale_groups, blocks_per_scale_group, 4)
    top2 = wb.abs().topk(k=2, dim=-1, largest=True, sorted=True).indices
    top2_sorted = top2.sort(dim=-1).values

    pair_code = torch.empty(top2_sorted.shape[:-1], dtype=torch.uint8, device=weight.device)
    for pair, code in PAIR_TO_CODE.items():
        m = (top2_sorted[..., 0] == pair[0]) & (top2_sorted[..., 1] == pair[1])
        pair_code[m] = code

    mask = torch.zeros_like(wb, dtype=torch.bool).scatter_(-1, top2_sorted, True)
    signed = torch.where(mask, wb.sign(), torch.zeros_like(wb))
    if scale_override is None:
        numerator = (wb * signed).sum(dim=(-1, -2))
        denominator = signed.abs().sum(dim=(-1, -2)).clamp_min(cfg.eps)
        scale = (numerator / denominator).clamp_min(cfg.eps)
    else:
        scale = scale_override.detach().float().clamp_min(cfg.eps)

    scale = scale.to(cfg.torch_scale_dtype).contiguous()

    sign0 = signed.gather(-1, top2_sorted[..., 0:1]).squeeze(-1).lt(0).to(torch.uint8)
    sign1 = signed.gather(-1, top2_sorted[..., 1:2]).squeeze(-1).lt(0).to(torch.uint8)
    codes = (pair_code | (sign0 << 3) | (sign1 << 4)).contiguous()
    codes = codes.view(out_features, total_blocks_per_row).contiguous()
    bitstream = pack_5bit_codes(codes.flatten()).cpu()
    codes = codes.cpu()

    return {
        "codes_u8": codes,
        "codes_5bit": bitstream,
        "scales": scale.cpu(),
        "orig_out_features": orig_out,
        "orig_in_features": orig_in,
        "padded_in_features": in_padded,
        "scale_group_size": cfg.scale_group_size,
        "scale_dtype": cfg.scale_dtype,
        "format": "T24W125",
    }


def unpack_t24(packed: dict[str, torch.Tensor | int | str], device: torch.device | str = "cpu") -> torch.Tensor:
    codes = packed.get("codes_u8")
    if codes is None:
        codes_5bit = packed["codes_5bit"]
        out_features = int(packed["orig_out_features"])
        padded_in = int(packed["padded_in_features"])
        n_codes = out_features * (padded_in // 4)
        codes = unpack_5bit_codes(codes_5bit, n_codes).view(out_features, padded_in // 4)
    codes = codes.to(device=device, dtype=torch.uint8)
    scales = packed["scales"].to(device=device).float()
    orig_in = int(packed["orig_in_features"])
    padded_in = int(packed["padded_in_features"])
    scale_group_size = int(packed["scale_group_size"])
    out_features = int(packed["orig_out_features"])
    blocks_per_scale_group = scale_group_size // 4
    scale_groups = padded_in // scale_group_size

    flat = torch.zeros((out_features, padded_in // 4, 4), dtype=torch.float32, device=device)
    pair_code = (codes & 0b111).long()
    sign0 = torch.where((codes & 0b01000) != 0, -1.0, 1.0)
    sign1 = torch.where((codes & 0b10000) != 0, -1.0, 1.0)
    pairs = CODE_TO_PAIR.to(device=device)[pair_code.clamp_max(5)]

    row_idx = torch.arange(out_features, device=device)[:, None].expand_as(pair_code)
    block_idx = torch.arange(padded_in // 4, device=device)[None, :].expand_as(pair_code)
    flat[row_idx, block_idx, pairs[..., 0]] = sign0
    flat[row_idx, block_idx, pairs[..., 1]] = sign1

    flat = flat.view(out_features, scale_groups, blocks_per_scale_group, 4)
    dense = flat * scales[:, :, None, None]
    dense = dense.view(out_features, padded_in)
    return dense[:, :orig_in]


def pack_5bit_codes(codes: torch.Tensor) -> torch.Tensor:
    c = codes.to(dtype=torch.uint8).flatten() & 0x1F
    n = c.numel()
    pad = (-n) % 8

    if pad:
        c = torch.cat([c, torch.zeros(pad, dtype=torch.uint8, device=c.device)])

    c64 = c.view(-1, 8).to(torch.int64)

    v = (
        c64[:, 0]
        | (c64[:, 1] << 5)
        | (c64[:, 2] << 10)
        | (c64[:, 3] << 15)
        | (c64[:, 4] << 20)
        | (c64[:, 5] << 25)
        | (c64[:, 6] << 30)
        | (c64[:, 7] << 35)
    )

    out = torch.empty((c64.shape[0], 5), dtype=torch.uint8, device=c.device)
    out[:, 0] = (v >> 0).to(torch.uint8)
    out[:, 1] = (v >> 8).to(torch.uint8)
    out[:, 2] = (v >> 16).to(torch.uint8)
    out[:, 3] = (v >> 24).to(torch.uint8)
    out[:, 4] = (v >> 32).to(torch.uint8)

    return out.flatten()[: (n * 5 + 7) // 8].contiguous()


def unpack_5bit_codes(bitstream: torch.Tensor, n_codes: int) -> torch.Tensor:
    bs = bitstream.to(dtype=torch.uint8).flatten()
    n_groups = (n_codes + 7) // 8
    need_bytes = n_groups * 5

    if bs.numel() < need_bytes:
        bs = torch.cat([bs, torch.zeros(need_bytes - bs.numel(), dtype=torch.uint8, device=bs.device)])

    b = bs[:need_bytes].view(-1, 5).to(torch.int64)

    v = (
        b[:, 0]
        | (b[:, 1] << 8)
        | (b[:, 2] << 16)
        | (b[:, 3] << 24)
        | (b[:, 4] << 32)
    )

    out = torch.empty((n_groups, 8), dtype=torch.uint8, device=bs.device)
    out[:, 0] = ((v >> 0) & 0x1F).to(torch.uint8)
    out[:, 1] = ((v >> 5) & 0x1F).to(torch.uint8)
    out[:, 2] = ((v >> 10) & 0x1F).to(torch.uint8)
    out[:, 3] = ((v >> 15) & 0x1F).to(torch.uint8)
    out[:, 4] = ((v >> 20) & 0x1F).to(torch.uint8)
    out[:, 5] = ((v >> 25) & 0x1F).to(torch.uint8)
    out[:, 6] = ((v >> 30) & 0x1F).to(torch.uint8)
    out[:, 7] = ((v >> 35) & 0x1F).to(torch.uint8)

    return out.flatten()[:n_codes].contiguous()


def theoretical_bits_per_weight(scale_group_size: int, scale_bits: int = 16) -> float:
    if scale_group_size % 4 != 0:
        raise ValueError("scale_group_size must be divisible by 4")
    return 1.25 + scale_bits / scale_group_size
