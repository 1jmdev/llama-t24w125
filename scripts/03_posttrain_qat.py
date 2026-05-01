#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from t24w125.data import TokenBlockDataset
from t24w125.modules import replace_linear_with_t24, set_t24_alpha
from t24w125.muon import Muon, split_muon_adamw_params
from t24w125.quant import QuantConfig, theoretical_bits_per_weight
from t24w125.utils import get_amp_dtype, load_config, save_json, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/llama32_1b_t24_qat.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--token-file", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume-state", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def save_checkpoint(path: Path, model, optimizers, schedulers, step: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": [o.state_dict() for o in optimizers],
            "schedulers": [s.state_dict() for s in schedulers],
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def evaluate(model, loader, device, dtype, max_batches: int) -> dict:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                out = model(**batch)
            losses.append(float(out.loss.detach().cpu()))
    model.train()
    loss = sum(losses) / max(1, len(losses))
    return {"loss": loss, "ppl": math.exp(min(20, loss))}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["train"]
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    qcfg = QuantConfig(
        scale_group_size=cfg["quant"].get("scale_group_size", 128),
        scale_dtype=cfg["quant"].get("scale_dtype", "float16"),
    )
    seed_everything(int(tcfg.get("seed", 1337)))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = get_amp_dtype(mcfg.get("dtype", "bf16"))
    model_name = args.model or mcfg["name_or_path"]
    token_file = Path(args.token_file or Path(dcfg["output_dir"]) / "tokens.bin")
    output_dir = Path(args.output_dir or tcfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=mcfg.get("trust_remote_code", False),
        attn_implementation=mcfg.get("attn_implementation", "sdpa"),
    )
    replaced = replace_linear_with_t24(model, qcfg, skip_names=cfg["quant"].get("skip_names", []))
    set_t24_alpha(model, 0.0)

    if mcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model.to(device)

    if args.compile or mcfg.get("torch_compile", False):
        model = torch.compile(model)

    seq_len = int(dcfg.get("seq_len", 2048))
    dataset = TokenBlockDataset(token_file, seq_len=seq_len)
    eval_len = min(max(128, len(dataset) // 100), max(1, len(dataset) // 10)) if len(dataset) else 0
    train_len = len(dataset) - eval_len
    if train_len <= 0:
        raise RuntimeError("Token dataset is too small. Run 01_prepare_ultrafineweb.py first.")

    train_ds, eval_ds = random_split(
        dataset,
        [train_len, eval_len],
        generator=torch.Generator().manual_seed(int(tcfg.get("seed", 1337))),
    )

    batch_size = int(args.batch_size or tcfg.get("batch_size", 1))
    grad_accum = int(args.grad_accum_steps or tcfg.get("grad_accum_steps", 16))
    max_steps = int(args.max_steps or tcfg.get("max_steps", 10000))
    optimizer_steps = max(1, math.ceil(max_steps / grad_accum))
    warmup = min(int(tcfg.get("warmup_steps", 20)), max(1, optimizer_steps // 10))

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(tcfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    baseline = evaluate(model, eval_loader, device, dtype, 8)
    print(json.dumps({"dense_baseline_before_qat": baseline}), flush=True)

    muon_params, adamw_params = split_muon_adamw_params(model)
    opt_muon = Muon(
        muon_params,
        lr=float(tcfg.get("learning_rate_muon", 0.006)),
        momentum=float(tcfg.get("muon_momentum", 0.95)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        ns_steps=int(tcfg.get("muon_ns_steps", 5)),
    )
    opt_adamw = torch.optim.AdamW(
        adamw_params,
        lr=float(tcfg.get("learning_rate_adamw", 2e-5)),
        betas=(0.9, 0.95),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
    )

    sched_muon = get_cosine_schedule_with_warmup(opt_muon, warmup, optimizer_steps)
    sched_adamw = get_cosine_schedule_with_warmup(opt_adamw, warmup, optimizer_steps)

    start_step = 0
    if args.resume_state:
        ckpt = torch.load(args.resume_state, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        opt_muon.load_state_dict(ckpt["optimizers"][0])
        opt_adamw.load_state_dict(ckpt["optimizers"][1])
        sched_muon.load_state_dict(ckpt["schedulers"][0])
        sched_adamw.load_state_dict(ckpt["schedulers"][1])
        start_step = int(ckpt.get("step", 0))

    report = {
        "model": model_name,
        "token_file": str(token_file),
        "dataset_blocks": len(dataset),
        "replaced_linear_layers": replaced,
        "format": "T24W125 QAT",
        "scale_group_size": qcfg.scale_group_size,
        "bits_per_weight_including_fp16_scales": theoretical_bits_per_weight(qcfg.scale_group_size),
        "muon_params": sum(p.numel() for p in muon_params),
        "adamw_params": sum(p.numel() for p in adamw_params),
        "grad_accum_steps": grad_accum,
        "micro_steps": max_steps,
        "optimizer_steps": optimizer_steps,
        "warmup_optimizer_steps": warmup,
        "quant_warmup_steps": int(tcfg.get("quant_warmup_steps", 4000)),
        "quant_start_alpha": float(tcfg.get("quant_start_alpha", 0.0)),
        "quant_end_alpha": float(tcfg.get("quant_end_alpha", 1.0)),
    }
    save_json(output_dir / "train_report.json", report)
    print(json.dumps(report, indent=2))

    log_every = int(tcfg.get("log_every", 10))
    eval_every = int(tcfg.get("eval_every", 250))
    save_every = int(tcfg.get("save_every", 500))
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
    quant_warmup_steps = int(tcfg.get("quant_warmup_steps", 4000))
    quant_start_alpha = float(tcfg.get("quant_start_alpha", 0.0))
    quant_end_alpha = float(tcfg.get("quant_end_alpha", 1.0))

    step = start_step
    accum_loss = 0.0
    t0 = time.time()
    model.train()
    opt_muon.zero_grad(set_to_none=True)
    opt_adamw.zero_grad(set_to_none=True)

    pbar = tqdm(total=max_steps, initial=start_step, desc="QAT")
    while step < max_steps:
        for batch in loader:
            if quant_warmup_steps > 0:
                t = min(1.0, step / quant_warmup_steps)
                t = t * t
                a = quant_start_alpha + (quant_end_alpha - quant_start_alpha) * t
            else:
                a = quant_end_alpha
            set_t24_alpha(model, a)

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                out = model(**batch)
                loss = out.loss / grad_accum

            loss.backward()
            accum_loss += float(loss.detach().cpu())

            if (step + 1) % grad_accum == 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt_muon.step()
                opt_adamw.step()
                sched_muon.step()
                sched_adamw.step()
                opt_muon.zero_grad(set_to_none=True)
                opt_adamw.zero_grad(set_to_none=True)

            step += 1
            pbar.update(1)

            if step % log_every == 0:
                toks = step * batch_size * seq_len
                elapsed = max(1e-6, time.time() - t0)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": accum_loss / log_every * grad_accum,
                            "quant_alpha": a,
                            "tok_s": toks / elapsed,
                        }
                    )
                )
                accum_loss = 0.0

            if eval_every and step % eval_every == 0:
                metrics = evaluate(model, eval_loader, device, dtype, int(tcfg.get("eval_batches", 64)))
                save_json(output_dir / f"eval_step_{step}.json", metrics)
                print(json.dumps({"eval_step": step, **metrics}))

            if save_every and step % save_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_step_{step}.pt",
                    model,
                    [opt_muon, opt_adamw],
                    [sched_muon, sched_adamw],
                    step,
                    {"loss_recent": accum_loss, "quant_alpha": a},
                )

            if step >= max_steps:
                break

    pbar.close()
    save_checkpoint(output_dir / "checkpoint_final.pt", model, [opt_muon, opt_adamw], [sched_muon, sched_adamw], step, {})
    AutoTokenizer.from_pretrained(model_name, use_fast=True).save_pretrained(output_dir / "tokenizer")


if __name__ == "__main__":
    main()
