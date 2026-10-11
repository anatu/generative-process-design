"""Datasets and batching for string-to-string tasks (BLUEPRINT WS5)."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from torch import Tensor

from gpd.canonical.tokenizer import Vocab
from gpd.corpus.pairs import read_jsonl


def load_pairs(
    path: Path, *, src_field: str = "pfd", tgt_field: str = "pid"
) -> list[tuple[str, str]]:
    return [(getattr(r, src_field), getattr(r, tgt_field)) for r in read_jsonl(path)]


class PairDataset:
    """Token-id pairs. Pairs longer than ``max_len`` (including BOS/EOS) are dropped and counted."""

    def __init__(self, pairs: Sequence[tuple[str, str]], vocab: Vocab, *, max_len: int) -> None:
        self.items: list[tuple[list[int], list[int]]] = []
        self.dropped = 0
        for s, t in pairs:
            si, ti = vocab.encode(s), vocab.encode(t)
            if len(si) > max_len or len(ti) > max_len:
                self.dropped += 1
                continue
            self.items.append((si, ti))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[list[int], list[int]]:
        return self.items[i]


def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int) -> Tensor:
    width = max(len(s) for s in seqs)
    out = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def collate(batch: Sequence[tuple[list[int], list[int]]], pad_id: int) -> dict[str, Tensor]:
    src = pad_batch([s for s, _ in batch], pad_id)
    tgt = pad_batch([t for _, t in batch], pad_id)
    return {"src": src, "tgt_in": tgt[:, :-1], "tgt_out": tgt[:, 1:]}


def bucket_batches(
    dataset: PairDataset, *, batch_size: int, shuffle: bool, seed: int, bucket_mult: int = 50
) -> Iterator[list[tuple[list[int], list[int]]]]:
    """Length-bucketed batching: shuffle, sort within chunks of ``bucket_mult`` batches, re-shuffle batches."""
    idx = list(range(len(dataset)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(idx)
    chunk = batch_size * bucket_mult
    batches: list[list[int]] = []
    for start in range(0, len(idx), chunk):
        part = sorted(idx[start : start + chunk], key=lambda i: len(dataset[i][1]))
        batches += [part[j : j + batch_size] for j in range(0, len(part), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    for b in batches:
        yield [dataset[i] for i in b]


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
