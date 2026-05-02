#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from t24w125.modules import T24LinearSTE, iter_t24_modules, replace_linear_with_t24
from t24w125.quant import QuantConfig, theoretical_bits_per_weight
from t24w125.utils import get_amp_dtype, load_config, save_json, sha256_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/llama32_1b_t24_qat.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-codes-u8", action="store_true")
    return p.parse_args()


def clean_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    qcfg = QuantConfig(
        scale_group_size=cfg["quant"].get("scale_group_size", 128),
        scale_dtype=cfg["quant"].get("scale_dtype", "float16"),
    )
    model_name = args.model or cfg["model"]["name_or_path"]
    output_dir = Path(args.output_dir or cfg["export"]["packed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = get_amp_dtype(cfg["model"].get("dtype", "bf16"))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    replace_linear_with_t24(model, qcfg, skip_names=cfg["quant"].get("skip_names", []))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = clean_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    tensors: dict[str, torch.Tensor] = {}
    metadata = {
        "format": "T24W125",
        "base_model": model_name,
        "scale_group_size": qcfg.scale_group_size,
        "scale_dtype": qcfg.scale_dtype,
        "bits_per_weight_raw": 1.25,
        "bits_per_weight_including_fp16_scales": theoretical_bits_per_weight(qcfg.scale_group_size),
        "modules": {},
        "non_quantized_tensors": [],
    }

    t24_names = {name for name, _ in iter_t24_modules(model)}
    for name, module in tqdm(list(iter_t24_modules(model)), desc="packing T24 modules"):
        assert isinstance(module, T24LinearSTE)
        payload = module.pack()
        prefix = f"t24.{name}"
        tensors[f"{prefix}.codes_5bit"] = payload["codes_5bit"].contiguous()
        if args.save_codes_u8:
            tensors[f"{prefix}.codes_u8"] = payload["codes_u8"].contiguous()
        tensors[f"{prefix}.scales"] = payload["scales"].contiguous()
        metadata["modules"][name] = {
            "orig_out_features": int(payload["orig_out_features"]),
            "orig_in_features": int(payload["orig_in_features"]),
            "padded_in_features": int(payload["padded_in_features"]),
            "scale_group_size": int(payload["scale_group_size"]),
            "scale_dtype": str(payload["scale_dtype"]),
            "codes_5bit_key": f"{prefix}.codes_5bit",
            "codes_u8_key": f"{prefix}.codes_u8" if args.save_codes_u8 else None,
            "scales_key": f"{prefix}.scales",
        }

    for name, tensor in model.state_dict().items():
        if name.removeprefix("_orig_mod.").endswith(".weight"):
            base_name = name.removeprefix("_orig_mod.")[: -len(".weight")]
            if base_name in t24_names:
                continue
        key = f"dense.{name.removeprefix('_orig_mod.')}"
        tensors[key] = tensor.detach().cpu().contiguous()
        metadata["non_quantized_tensors"].append(key)

    packed_path = output_dir / "model_t24w125.safetensors"
    safe_tensors = {}
    seen_ptrs = set()

    for name, tensor in tensors.items():
        tensor = tensor.detach().cpu().contiguous()
        ptr = tensor.untyped_storage().data_ptr()

        if ptr in seen_ptrs:
            tensor = tensor.clone()

        seen_ptrs.add(ptr)
        safe_tensors[name] = tensor

    save_file(safe_tensors, packed_path, metadata={"format": "T24W125"})

    AutoConfig.from_pretrained(model_name).save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(model_name, use_fast=True).save_pretrained(output_dir)
    metadata["safetensors_sha256"] = sha256_file(packed_path)
    save_json(output_dir / "t24w125_metadata.json", metadata)
    print(json.dumps(metadata, indent=2)[:6000])
    print(f"wrote {packed_path}")


if __name__ == "__main__":
    main()
