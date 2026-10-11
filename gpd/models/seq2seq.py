"""Encoder-decoder transformer for SFILES -> SFILES tasks (BLUEPRINT WS5.3 / WS5.4)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Seq2SeqConfig:
    vocab_size: int
    pad_id: int
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 320

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Seq2SeqConfig:
        return cls(**d)


class Seq2Seq(nn.Module):
    def __init__(self, cfg: Seq2SeqConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.scale = cfg.d_model**0.5
        # Output projection tied to the input embedding.
        self.out_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.drop = nn.Dropout(cfg.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Tied in/out embeddings need std d_model**-0.5 so initial logits are O(1)."""
        nn.init.normal_(self.tok_emb.weight, std=self.cfg.d_model**-0.5)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[self.cfg.pad_id].zero_()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _embed(self, ids: Tensor) -> Tensor:
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        out: Tensor = self.drop(self.tok_emb(ids) * self.scale + self.pos_emb(pos))
        return out

    def encode(self, src: Tensor) -> tuple[Tensor, Tensor]:
        src_pad = src.eq(self.cfg.pad_id)
        memory = self.transformer.encoder(self._embed(src), src_key_padding_mask=src_pad)
        return memory, src_pad

    def decode(self, tgt_in: Tensor, memory: Tensor, src_pad: Tensor) -> Tensor:
        length = tgt_in.size(1)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tgt_in.device), diagonal=1
        )
        tgt_pad = tgt_in.eq(self.cfg.pad_id)
        h = self.transformer.decoder(
            self._embed(tgt_in),
            memory,
            tgt_mask=causal,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        h = self.out_norm(h)
        logits: Tensor = h @ self.tok_emb.weight.T + self.out_bias
        return logits

    def forward(self, src: Tensor, tgt_in: Tensor) -> Tensor:
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad)

    # ----------------------------------------------------------- persistence
    def save(self, path: Path, *, extra: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": asdict(self.cfg), "state_dict": self.state_dict(), "extra": extra or {}},
            path,
        )

    @classmethod
    def load(
        cls, path: Path, *, device: torch.device | str = "cpu"
    ) -> tuple[Seq2Seq, dict[str, Any]]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(Seq2SeqConfig.from_dict(ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        extra: dict[str, Any] = ckpt.get("extra", {})
        return model, extra
