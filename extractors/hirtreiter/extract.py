"""Hirtreiter PFD->P&ID ingest (BLUEPRINT WS2.1).

Reads the Zenodo files under ``data/raw/hirtreiter/Training Data Public Upload``
and writes ``PairRecord`` JSONL per split, keeping upstream splits verbatim:

    train_data_<size>.json       -> train_<size>.jsonl
    eval_data_1k.json            -> val.jsonl
    test_data_1k.json            -> test.jsonl
    train_data_augm_<size>.json  -> train_augm_<size>.jsonl   (optional)

All records are tagged ``is_real=False`` (the ReadMe states the data is synthetic).

    uv run python extractors/hirtreiter/extract.py --train-size 10k
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from gpd.corpus.pairs import PairRecord, Variant, make_pair, read_raw_pairs, write_jsonl
from gpd.corpus.record import Split

SOURCE = "hirtreiter"
LICENSE = "CC-BY-4.0"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data" / "raw" / SOURCE / "Training Data Public Upload"
DEFAULT_OUT = ROOT / "data" / "interim" / SOURCE

SPLIT_FILES: dict[str, tuple[str, Split]] = {
    "eval_data_1k": ("val", "val"),
    "test_data_1k": ("test", "test"),
}


def _suffix(variant: Variant) -> str:
    return "_wo_valves" if variant == "wo_valves" else ""


def convert_file(
    path: Path, *, split: Split, variant: Variant, out_name: str, canonicalize: bool
) -> list[PairRecord]:
    records: list[PairRecord] = []
    for i, (pfd, pid) in enumerate(read_raw_pairs(path)):
        records.append(
            make_pair(
                id=f"{SOURCE}:{out_name}:{i}",
                source=SOURCE,
                license=LICENSE,
                is_real=False,
                variant=variant,
                split=split,
                pfd=pfd,
                pid=pid,
                canonicalize=canonicalize,
            )
        )
    return records


def stats_for(records: list[PairRecord]) -> dict[str, float | int]:
    n = len(records)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "pfd_parses": sum(r.pfd_parses for r in records) / n,
        "pid_parses": sum(r.pid_parses for r in records) / n,
        "pfd_is_canonical": sum(r.pfd_is_canonical for r in records) / n,
        "pid_is_canonical": sum(r.pid_is_canonical for r in records) / n,
        "pfd_tokens_max": max(r.pfd_tokens for r in records),
        "pid_tokens_max": max(r.pid_tokens for r in records),
        "pid_tokens_p99": sorted(r.pid_tokens for r in records)[int(0.99 * (n - 1))],
    }


def run(
    *,
    raw_dir: Path,
    out_dir: Path,
    train_size: str,
    variant: Variant,
    include_augmented: bool,
    canonicalize: bool,
) -> dict[str, dict[str, float | int]]:
    sfx = _suffix(variant)
    out_dir = out_dir / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, Split, str]] = [
        (raw_dir / f"train_data_{train_size}{sfx}.json", "train", f"train_{train_size}"),
        (raw_dir / f"eval_data_1k{sfx}.json", "val", "val"),
        (raw_dir / f"test_data_1k{sfx}.json", "test", "test"),
    ]
    if include_augmented:
        jobs.append(
            (
                raw_dir / f"train_data_augm_{train_size}{sfx}.json",
                "train",
                f"train_augm_{train_size}",
            )
        )
    stats: dict[str, dict[str, float | int]] = {}
    for path, split, out_name in jobs:
        if not path.exists():
            typer.echo(f"skip (missing): {path.name}", err=True)
            continue
        records = convert_file(
            path, split=split, variant=variant, out_name=out_name, canonicalize=canonicalize
        )
        n = write_jsonl(records, out_dir / f"{out_name}.jsonl")
        stats[out_name] = stats_for(records)
        typer.echo(f"{out_name}: {n} pairs -> {out_dir / (out_name + '.jsonl')}")
    stats_path = out_dir / "stats.json"
    existing = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    existing.update(stats)
    stats_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return stats


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    raw_dir: Path = typer.Option(DEFAULT_RAW),
    out_dir: Path = typer.Option(DEFAULT_OUT),
    train_size: str = typer.Option("10k", help="100 | 1k | 10k | 100k"),
    variant: str = typer.Option("standard", help="standard | wo_valves"),
    include_augmented: bool = typer.Option(False),
    canonicalize: bool = typer.Option(True, help="compute canonical forms and grammar flags"),
) -> None:
    if variant not in ("standard", "wo_valves"):
        raise typer.BadParameter("variant must be standard or wo_valves")
    stats = run(
        raw_dir=raw_dir,
        out_dir=out_dir,
        train_size=train_size,
        variant=variant,  # type: ignore[arg-type]
        include_augmented=include_augmented,
        canonicalize=canonicalize,
    )
    typer.echo(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(app())
