#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from t24w125.utils import load_config, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/llama32_1b_t24_qat.yaml")
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--text-column", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-raw-gb", type=float, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--token-dtype", default=None, choices=["uint32", "int64"])
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--no-streaming", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    dataset_name = args.dataset_name or dcfg["dataset_name"]
    dataset_config = args.dataset_config if args.dataset_config is not None else dcfg.get("dataset_config")
    split = args.split or dcfg.get("split", "train")
    text_column = args.text_column or dcfg.get("text_column", "text")
    model_name = args.model or mcfg["name_or_path"]
    output_dir = Path(args.output_dir or dcfg["output_dir"])
    max_raw_gb = float(args.max_raw_gb if args.max_raw_gb is not None else dcfg.get("max_raw_gb", 10))
    seq_len = int(args.seq_len or dcfg.get("seq_len", 2048))
    token_dtype = np.dtype(args.token_dtype or dcfg.get("token_dtype", "uint32"))
    streaming = dcfg.get("streaming", True)
    if args.streaming:
        streaming = True
    if args.no_streaming:
        streaming = False

    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / "tokens.bin"
    meta_path = output_dir / "tokens.bin.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(path=dataset_name, split=split, streaming=streaming)
    if dataset_config:
        kwargs["name"] = dataset_config
    ds = load_dataset(**kwargs)

    raw_limit = int(max_raw_gb * (1024**3))
    raw_seen = 0
    token_count = 0
    doc_count = 0

    with open(token_path, "wb") as f:
        pbar = tqdm(total=raw_limit, unit="B", unit_scale=True, desc="raw text")
        for row in ds:
            text = row.get(text_column)
            if not text:
                continue
            encoded_raw = text.encode("utf-8", errors="ignore")
            raw_seen += len(encoded_raw)
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids.append(tokenizer.eos_token_id)
            arr = np.asarray(ids, dtype=token_dtype)
            arr.tofile(f)
            token_count += int(arr.size)
            doc_count += 1
            pbar.update(min(len(encoded_raw), max(0, raw_limit - pbar.n)))
            if raw_seen >= raw_limit:
                break
        pbar.close()

    meta = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "text_column": text_column,
        "model_name": model_name,
        "raw_bytes_seen": raw_seen,
        "max_raw_gb": max_raw_gb,
        "token_count": token_count,
        "doc_count": doc_count,
        "seq_len": seq_len,
        "dtype": str(token_dtype),
        "num_blocks": max(0, (token_count - 1) // seq_len),
    }
    save_json(meta_path, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
