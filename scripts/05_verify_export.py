#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from t24w125.modules import iter_t24_modules, replace_linear_with_t24
from t24w125.quant import QuantConfig, quantize_t24_dense, unpack_t24, unpack_5bit_codes
from t24w125.utils import get_amp_dtype, load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/llama32_1b_t24_qat.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--packed-dir", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-layers", type=int, default=8)
    p.add_argument("--atol", type=float, default=1e-3)
    return p.parse_args()


def clean_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model or cfg["model"]["name_or_path"]
    packed_dir = Path(args.packed_dir or cfg["export"]["packed_dir"])
    metadata = json.loads((packed_dir / "t24w125_metadata.json").read_text())
    tensors = load_file(packed_dir / "model_t24w125.safetensors", device="cpu")
    qcfg = QuantConfig(
        scale_group_size=cfg["quant"].get("scale_group_size", 128),
        scale_dtype=cfg["quant"].get("scale_dtype", "float16"),
    )
    dtype = get_amp_dtype(cfg["model"].get("dtype", "bf16"))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    replace_linear_with_t24(model, qcfg, skip_names=cfg["quant"].get("skip_names", []))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"] if "model" in ckpt else ckpt), strict=True)
    model.to(device).eval()

    checked = []
    for i, (name, module) in enumerate(iter_t24_modules(model)):
        if i >= args.max_layers:
            break
        info = metadata["modules"][name]
        codes_key = info.get("codes_u8_key")
        if codes_key and codes_key in tensors:
            codes_u8 = tensors[codes_key]
        else:
            n_codes = int(info["orig_out_features"]) * (int(info["padded_in_features"]) // 4)
            codes_u8 = unpack_5bit_codes(tensors[info["codes_5bit_key"]], n_codes).view(
                int(info["orig_out_features"]), int(info["padded_in_features"]) // 4
            )
        packed = {
            "codes_u8": codes_u8,
            "scales": tensors[info["scales_key"]],
            "orig_out_features": int(info["orig_out_features"]),
            "orig_in_features": int(info["orig_in_features"]),
            "padded_in_features": int(info["padded_in_features"]),
            "scale_group_size": int(info["scale_group_size"]),
            "scale_dtype": info["scale_dtype"],
            "format": "T24W125",
        }
        dense_from_export = unpack_t24(packed, device=device).float()
        dense_ref = quantize_t24_dense(module.weight, qcfg).float()
        max_abs = (dense_from_export - dense_ref).abs().max().item()
        ok = max_abs <= args.atol
        checked.append({"name": name, "max_abs_diff": max_abs, "ok": ok})
        if not ok:
            raise AssertionError(f"{name} max_abs_diff={max_abs} > {args.atol}")
    print(json.dumps({"checked_layers": checked, "ok": True}, indent=2))


if __name__ == "__main__":
    main()
