from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class TokenBlockDataset(Dataset):
    def __init__(self, token_file: str | Path, seq_len: int) -> None:
        self.token_file = Path(token_file)
        self.seq_len = seq_len
        meta_path = self.token_file.with_suffix(self.token_file.suffix + ".json")
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text())
            dtype = np.dtype(self.meta.get("dtype", "uint32"))
        else:
            self.meta = {}
            dtype = np.uint32
        self.tokens = np.memmap(self.token_file, dtype=dtype, mode="r")
        self.n = max(0, len(self.tokens) // seq_len)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.seq_len
        arr = np.asarray(self.tokens[start : start + self.seq_len], dtype=np.int64)
        x = torch.from_numpy(arr.copy())
        return {"input_ids": x, "labels": x.clone()}


class StreamingTokenBlockDataset(IterableDataset):
    def __init__(self, token_file: str | Path, seq_len: int, shuffle_buffer: int = 0) -> None:
        self.base = TokenBlockDataset(token_file, seq_len)
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        n = len(self.base)
        if worker_info is None:
            start, step = 0, 1
        else:
            start, step = worker_info.id, worker_info.num_workers
        indices = range(start, n, step)
        if self.shuffle_buffer <= 0:
            for i in indices:
                yield self.base[i]
        else:
            rng = np.random.default_rng()
            buf = []
            for i in indices:
                buf.append(i)
                if len(buf) >= self.shuffle_buffer:
                    j = int(rng.integers(len(buf)))
                    yield self.base[buf.pop(j)]
            while buf:
                j = int(rng.integers(len(buf)))
                yield self.base[buf.pop(j)]
