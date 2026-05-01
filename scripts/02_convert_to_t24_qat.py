#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from t24w125.modules import iter_t24_modules, replace_linear_with_t24
from t24w125.quant import QuantConfig, theoretical_bits_per_weight
from t24w125.utils import get_amp_dtype, load_config, save_json, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/llama32_1b_t24_qat.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--output-dir", default="outputs/llama32_1b_t24_qat_init")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["train"].get("seed", 1337))
    qcfg = QuantConfig(
        scale_group_size=cfg["quant"].get("scale_group_size", 128),
        scale_dtype=cfg["quant"].get("scale_dtype", "float16"),
    )
    model_name = args.model or cfg["model"]["name_or_path"]
    dtype = get_amp_dtype(cfg["model"].get("dtype", "bf16"))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    ).to(device)
    n_replaced = replace_linear_with_t24(model, qcfg, skip_names=cfg["quant"].get("skip_names", []))
    n_t24 = sum(1 for _ in iter_t24_modules(model))
    report = {
        "model": model_name,
        "format": "T24W125 fake-quant QAT init",
        "replaced_linear_layers": n_replaced,
        "t24_modules": n_t24,
        "scale_group_size": qcfg.scale_group_size,
        "bits_per_weight_including_scales": theoretical_bits_per_weight(qcfg.scale_group_size),
    }
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_name, use_fast=True).save_pretrained(out)
    save_json(out / "t24_report.json", report)


if __name__ == "__main__":
    main()
