"""Benchmarks and the regression gate (BLUEPRINT WS5.6).

    uv run python -m gpd.eval.benchmarks pfd2pid \
        --checkpoint runs/pfd2pid_10k/best.pt \
        --test data/interim/hirtreiter/standard/test.jsonl

Writes ``eval/runs/<timestamp>_pfd2pid.json`` and compares against
``gpd/eval/baselines.json``.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from gpd.canonical import sfiles
from gpd.canonical.tokenizer import Vocab
from gpd.models.data import load_pairs, pick_device
from gpd.models.decode import beam_search
from gpd.models.seq2seq import Seq2Seq

ROOT = Path(__file__).resolve().parents[2]
BASELINES = Path(__file__).with_name("baselines.json")
RUNS_DIR = ROOT / "eval" / "runs"


def load_baselines(path: Path = BASELINES) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _canonical_or_none(s: str) -> str | None:
    try:
        return sfiles.canonical_sfiles(s)
    except sfiles.CanonicalizationError:
        return None


def score_pfd2pid(hyps: list[list[tuple[str, float]]], targets: list[str]) -> dict[str, float]:
    n = len(targets)
    top1 = top5 = parses = canonical_match = idempotent = 0
    for hy, tgt in zip(hyps, targets, strict=True):
        texts = [h[0] for h in hy]
        best = texts[0] if texts else ""
        top1 += best == tgt
        top5 += tgt in texts
        c_best = _canonical_or_none(best)
        parses += c_best is not None
        idempotent += c_best == best
        if c_best is not None:
            c_tgt = _canonical_or_none(tgt)
            canonical_match += c_tgt is not None and c_best == c_tgt
    return {
        "n": n,
        "top1_exact": top1 / n,
        "top5_exact": top5 / n,
        "top1_parses": parses / n,
        "top1_is_canonical": idempotent / n,
        "top1_canonical_match": canonical_match / n,
    }


def evaluate_pfd2pid(
    checkpoint: Path,
    test: Path,
    *,
    beam: int = 5,
    limit: int | None = None,
    device: str = "auto",
    batch_size: int = 32,
) -> dict[str, Any]:
    dev = pick_device(device)
    model, extra = Seq2Seq.load(checkpoint, device=dev)
    vocab = Vocab.load(checkpoint.parent / "vocab.json")
    pairs = load_pairs(test)[:limit]
    srcs = [s for s, _ in pairs]
    tgts = [t for _, t in pairs]
    t0 = time.time()
    hyps = beam_search(model, vocab, srcs, device=dev, beam=beam, batch_size=batch_size)
    elapsed = time.time() - t0
    metrics = score_pfd2pid(hyps, tgts)
    return {
        "task": "pfd2pid",
        "checkpoint": str(checkpoint),
        "checkpoint_extra": extra,
        "test": str(test),
        "beam": beam,
        "device": str(dev),
        "elapsed_s": elapsed,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": metrics,
        "samples": [
            {"pfd": s, "target": t, "top1": h[0][0] if h else "", "ok": bool(h and h[0][0] == t)}
            for s, t, h in list(zip(srcs, tgts, hyps, strict=True))[:5]
        ],
    }


def compare(metrics: dict[str, float], baselines: dict[str, Any], task: str) -> list[str]:
    """Return human-readable pass/fail lines for the task's pass bars and published numbers."""
    lines: list[str] = []
    entry = baselines.get(task, {})
    for key, bar in entry.get("pass", {}).items():
        val = metrics.get(key)
        ok = val is not None and val >= bar
        lines.append(
            f"{'PASS' if ok else 'FAIL'} {key}: {val if val is None else f'{val:.3f}'} (bar {bar:.3f})"
        )
    for key, pub in entry.get("published", {}).get("metrics", {}).items():
        val = metrics.get(key)
        if val is not None:
            lines.append(
                f"ref  {key}: ours {val:.3f} vs published {pub:.3f} ({(val - pub) * 100:+.1f} pts)"
            )
    return lines


app = typer.Typer(add_completion=False, help=__doc__)


@app.callback()
def _root() -> None:
    """Benchmarks. Pick a task subcommand (keeps typer in subcommand mode with one command)."""


@app.command()
def pfd2pid(
    checkpoint: Path = typer.Option(...),
    test: Path = typer.Option(...),
    beam: int = 5,
    limit: int | None = None,
    device: str = "auto",
    batch_size: int = 32,
    out_dir: Path = RUNS_DIR,
) -> None:
    result = evaluate_pfd2pid(
        checkpoint, test, beam=beam, limit=limit, device=device, batch_size=batch_size
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"{stamp}_pfd2pid.json"
    out.write_text(json.dumps(result, indent=2))
    typer.echo(json.dumps(result["metrics"], indent=2))
    for line in compare(result["metrics"], load_baselines(), "pfd2pid"):
        typer.echo(line)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(app())
