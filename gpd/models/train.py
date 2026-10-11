"""Trainer for seq2seq tasks (BLUEPRINT WS5).

    uv run python -m gpd.models.train \
        --train data/interim/hirtreiter/standard/train_10k.jsonl \
        --val   data/interim/hirtreiter/standard/val.jsonl \
        --out   runs/pfd2pid_10k

Writes ``vocab.json``, ``config.json``, ``train_log.jsonl``, ``best.pt`` (by val
exact match), ``last.pt``, and ``summary.json``.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import typer
from torch import nn

from gpd.canonical.tokenizer import Vocab
from gpd.models.data import PairDataset, bucket_batches, collate, load_pairs, pick_device
from gpd.models.decode import greedy
from gpd.models.seq2seq import Seq2Seq, Seq2SeqConfig


@dataclass
class TrainConfig:
    train: Path
    val: Path
    out: Path
    epochs: int = 30
    batch_size: int = 64
    lr: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    d_model: int = 256
    nhead: int = 4
    layers: int = 4
    ff: int = 1024
    dropout: float = 0.1
    max_len: int = 320
    max_val: int = 500
    max_steps: int | None = None
    patience: int = 6
    seed: int = 0
    device: str = "auto"
    log_every: int = 50


def lr_at(step: int, cfg: TrainConfig, total_steps: int) -> float:
    """Linear warmup, then cosine decay to 10% of peak."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps))
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def exact_match(preds: list[str], targets: list[str]) -> float:
    return sum(p == t for p, t in zip(preds, targets, strict=True)) / max(1, len(targets))


def run(cfg: TrainConfig) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = pick_device(cfg.device)
    cfg.out.mkdir(parents=True, exist_ok=True)

    train_pairs = load_pairs(cfg.train)
    val_pairs = load_pairs(cfg.val)[: cfg.max_val]
    vocab = Vocab.build(s for pair in train_pairs for s in pair)
    vocab.save(cfg.out / "vocab.json")

    train_ds = PairDataset(train_pairs, vocab, max_len=cfg.max_len)
    val_ds = PairDataset(val_pairs, vocab, max_len=cfg.max_len)
    val_srcs = [s for s, _ in val_pairs]
    val_tgts = [t for _, t in val_pairs]

    mcfg = Seq2SeqConfig(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.layers,
        num_decoder_layers=cfg.layers,
        dim_feedforward=cfg.ff,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
    )
    model = Seq2Seq(mcfg).to(device)
    (cfg.out / "config.json").write_text(
        json.dumps(
            {
                "train": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                "model": asdict(mcfg),
            },
            indent=2,
        )
    )

    steps_per_epoch = math.ceil(len(train_ds) / cfg.batch_size)
    total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=cfg.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, label_smoothing=cfg.label_smoothing)

    typer.echo(
        f"device={device} params={model.num_parameters() / 1e6:.1f}M vocab={len(vocab)} "
        f"train={len(train_ds)} (dropped {train_ds.dropped}) val={len(val_ds)} steps={total_steps}"
    )

    log = (cfg.out / "train_log.jsonl").open("w")
    best_em, best_epoch, bad_epochs, step = -1.0, -1, 0, 0
    t0 = time.time()
    history: list[dict[str, Any]] = []
    stop = False

    for epoch in range(cfg.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for batch in bucket_batches(
            train_ds, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed + epoch
        ):
            tensors = {k: v.to(device) for k, v in collate(batch, vocab.pad_id).items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg, total_steps)
            logits = model(tensors["src"], tensors["tgt_in"])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tensors["tgt_out"].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.detach().item()
            n_batches += 1
            step += 1
            if step % cfg.log_every == 0:
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": running / n_batches,
                    "lr": lr_at(step, cfg, total_steps),
                    "t": time.time() - t0,
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                typer.echo(
                    f"step {step} epoch {epoch} loss {rec['loss']:.4f} lr {rec['lr']:.2e} {rec['t']:.0f}s"
                )
            if cfg.max_steps and step >= cfg.max_steps:
                stop = True
                break

        # ---- validation
        model.eval()
        with torch.no_grad():
            val_loss, n = 0.0, 0
            for batch in bucket_batches(val_ds, batch_size=cfg.batch_size, shuffle=False, seed=0):
                tensors = {k: v.to(device) for k, v in collate(batch, vocab.pad_id).items()}
                logits = model(tensors["src"], tensors["tgt_in"])
                val_loss += float(
                    loss_fn(logits.reshape(-1, logits.size(-1)), tensors["tgt_out"].reshape(-1))
                )
                n += 1
            preds = greedy(model, vocab, val_srcs, device=device, max_len=cfg.max_len)
            em = exact_match(preds, val_tgts)
        rec = {
            "epoch": epoch,
            "step": step,
            "train_loss": running / max(1, n_batches),
            "val_loss": val_loss / max(1, n),
            "val_exact_greedy": em,
            "t": time.time() - t0,
        }
        history.append(rec)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        typer.echo(
            f"== epoch {epoch} train {rec['train_loss']:.4f} val {rec['val_loss']:.4f} exact@greedy {em:.3f} ({rec['t']:.0f}s)"
        )

        model.save(
            cfg.out / "last.pt", extra={"epoch": epoch, "step": step, "val_exact_greedy": em}
        )
        if em > best_em:
            best_em, best_epoch, bad_epochs = em, epoch, 0
            model.save(
                cfg.out / "best.pt", extra={"epoch": epoch, "step": step, "val_exact_greedy": em}
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                typer.echo(f"early stop: no val improvement for {cfg.patience} epochs")
                break
        if stop:
            break

    log.close()
    summary = {
        "best_val_exact_greedy": best_em,
        "best_epoch": best_epoch,
        "steps": step,
        "params": model.num_parameters(),
        "vocab_size": len(vocab),
        "train_pairs": len(train_ds),
        "train_dropped": train_ds.dropped,
        "elapsed_s": time.time() - t0,
        "device": str(device),
        "history": history,
    }
    (cfg.out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    train: Path = typer.Option(...),
    val: Path = typer.Option(...),
    out: Path = typer.Option(...),
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 5e-4,
    warmup_steps: int = 500,
    d_model: int = 256,
    nhead: int = 4,
    layers: int = 4,
    ff: int = 1024,
    dropout: float = 0.1,
    max_len: int = 320,
    max_val: int = 500,
    max_steps: int | None = None,
    patience: int = 6,
    seed: int = 0,
    device: str = "auto",
    log_every: int = 50,
) -> None:
    cfg = TrainConfig(
        train=train, val=val, out=out, epochs=epochs, batch_size=batch_size, lr=lr,
        warmup_steps=warmup_steps, d_model=d_model, nhead=nhead, layers=layers, ff=ff,
        dropout=dropout, max_len=max_len, max_val=max_val, max_steps=max_steps,
        patience=patience, seed=seed, device=device, log_every=log_every,
    )  # fmt: skip
    summary = run(cfg)
    typer.echo(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    sys.exit(app())
