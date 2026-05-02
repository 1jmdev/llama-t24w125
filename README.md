# llama-t24w125

Python reference repo for converting `meta-llama/Llama-3.2-1B` from FP16/BF16 weights into a GPU-oriented **T24W125** 1.25-bit-like format, then post-training it with QAT on an Ultra-FineWeb slice.

`T24W125` means:

```text
Ternary weights: {-scale, 0, +scale}
2:4 structure: exactly two nonzero weights per group of four
Packed code: 5 bits per 4 weights = 1.25 raw bits/weight
Scales: configurable per output row and per input block, default 1 FP16 scale per 128 weights
Default effective storage: 1.25 + 16/128 = 1.375 bits/weight before tensor headers
```

This repo intentionally does **not** expand the final model to INT4/FP16 for storage. The export writes true 5-bit packed code streams plus scale tensors. The included inference module is a correctness reference; production serving should replace `PackedT24Linear` with fused CUDA/Triton kernels.

## What is included

```text
configs/llama32_1b_t24_qat.yaml     default config
scripts/00_smoke_test.py            local quantization + Muon sanity check
scripts/01_prepare_ultrafineweb.py   stream/tokenize ~10 GB Ultra-FineWeb
scripts/02_convert_to_t24_qat.py     replace Llama linear layers with T24 fake-quant modules
scripts/03_posttrain_qat.py          Muon + AdamW QAT post-training
scripts/04_export_packed_t24.py      export true packed 1.25-bit weights
scripts/05_verify_export.py          compare exported packed weights with QAT fake-quant reference
t24w125/quant.py                 T24 quantization, 5-bit packing, unpacking
t24w125/modules.py               T24LinearSTE and reference packed linear
t24w125/muon.py                  minimal Muon optimizer implementation
tests/                              unit tests
```

## Hardware target

For Llama 3.2 1B QAT, use a single strong GPU with at least 24 GB VRAM. For faster runs, use 48–80 GB VRAM, BF16, gradient checkpointing, and longer sequence/batch settings.

For future 100B–1T models, this Python repo is a format/training prototype. A real 1T runtime needs fused custom kernels and no dense master weights during inference.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

On cloud images that already have PyTorch/CUDA installed:

```bash
python -m pip install -e .
```

Log in to Hugging Face and accept the Llama 3.2 license before loading Meta weights:

```bash
hf login
```

## Verify local code

```bash
python scripts/00_smoke_test.py
pytest -q
```

Expected result: all tests pass and the smoke script prints `"ok": true`.

## Prepare ~10 GB Ultra-FineWeb slice

Default config uses `openbmb/Ultra-FineWeb`. Some mirrors/configs may require accepting Hugging Face dataset terms.

```bash
python scripts/01_prepare_ultrafineweb.py \
  --config configs/llama32_1b_t24_qat.yaml \
  --max-raw-gb 10 \
  --output-dir data/fineweb_10gb
```

This creates:

```text
data/fineweb_10gb/tokens.bin
data/fineweb_10gb/tokens.bin.json
```

If the official repo layout changes, override the dataset fields:

```bash
python scripts/01_prepare_ultrafineweb.py \
  --dataset-name openbmb/Ultra-FineWeb \
  --dataset-config en \
  --split train \
  --text-column text \
  --max-raw-gb 10
```

## Optional dry conversion check

```bash
python scripts/02_convert_to_t24_qat.py \
  --config configs/llama32_1b_t24_qat.yaml \
  --dry-run
```

This loads Llama 3.2 1B, replaces eligible `nn.Linear` layers with `T24LinearSTE`, and prints the number of converted layers.

## Post-train with Muon QAT

```bash
python scripts/03_posttrain_qat.py \
  --config configs/llama32_1b_t24_qat.yaml \
  --token-file data/fineweb_10gb/tokens.bin \
  --output-dir outputs/llama32_1b_t24_qat
```

Useful cloud overrides:

```bash
python scripts/03_posttrain_qat.py \
  --config configs/llama32_1b_t24_qat.yaml \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --max-steps 4000 \
  --compile
```

Checkpoints are written as:

```text
outputs/llama32_1b_t24_qat/checkpoint_step_*.pt
outputs/llama32_1b_t24_qat/checkpoint_final.pt
```

## Export packed 1.25-bit model

```bash
python scripts/04_export_packed_t24.py \
  --config configs/llama32_1b_t24_qat.yaml \
  --checkpoint outputs/llama32_1b_t24_qat/checkpoint_final.pt \
  --output-dir outputs/llama32_1b_t24_packed
```

Output:

```text
outputs/llama32_1b_t24_packed/model_t24w125.safetensors
outputs/llama32_1b_t24_packed/t24w125_metadata.json
outputs/llama32_1b_t24_packed/config.json
outputs/llama32_1b_t24_packed/tokenizer files
```

## Verify export

```bash
python scripts/05_verify_export.py \
  --config configs/llama32_1b_t24_qat.yaml \
  --checkpoint outputs/llama32_1b_t24_qat/checkpoint_final.pt \
  --packed-dir outputs/llama32_1b_t24_packed \
  --max-layers 16
```

This reconstructs selected packed layers and checks them against the dense fake-quant reference.

## Training notes

Muon is applied only to matrix-like hidden parameters. AdamW is used for vectors, norms, embeddings, and output-head-like parameters. This follows the common Muon usage pattern: matrix hidden layers get orthogonalized updates, while scalar/vector/special tensors stay on AdamW.

The QAT layer keeps a dense trainable master weight and uses T24 fake quantization in the forward pass with straight-through gradients. That is correct for post-training Llama 3.2 1B. It is not the final 1T inference design; final inference must run packed weights through fused kernels.

## Format details

Each four-weight block stores one logical 5-bit code:

```text
bits 0..2: which pair of positions is nonzero, one of six pairs
bit 3: sign of first nonzero
bit 4: sign of second nonzero
```

The repo stores:

```text
codes_5bit: true bit-packed byte stream
scales: scale tensor [out_features, padded_in_features / scale_group_size]
dense.*: non-quantized tensors such as norms, embeddings, lm_head, and config-critical weights
```

For debugging, add `--save-codes-u8` during export to also store one byte per logical 5-bit code.

## Arch Linux note

On Arch, install CUDA/PyTorch using your normal CUDA stack or a cloud image. The repo itself has no Hyprland-specific requirements.

## Known limitations

- This is a full Python reference implementation, not a fused CUDA runtime.
- Training-time QAT stores dense master weights, so it is not memory-equivalent to final inference.
- `PackedT24Linear` dequantizes for reference correctness and should not be used as the final high-performance serving path.
- The 1.25-bit code stream is real; the effective bits/weight are higher after scale overhead.
