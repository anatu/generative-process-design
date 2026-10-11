"""Download raw sources with checksum verification (BLUEPRINT §2.2, WS0).

    uv run python scripts/fetch.py --source hirtreiter
    uv run python scripts/fetch.py --source all --max-size-mb 100
    uv run python scripts/fetch.py --list

Every fetched artifact's sha256 (or git commit) is recorded in
``scripts/sources.lock.json`` and verified on subsequent runs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import typer
import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
SOURCES_YAML = ROOT / "scripts" / "sources.yaml"
LOCK_JSON = ROOT / "scripts" / "sources.lock.json"
RAW_DIR = ROOT / "data" / "raw"

Downloader = Callable[[str, Path], None]


class FetchError(RuntimeError):
    pass


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["http", "git", "manual"]
    url: str
    filename: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    extract: Literal["zip"] | None = None
    ref: str | None = None
    license: str
    tier: str
    size_mb: float | None = None
    expected: str | None = None
    notes: str | None = None


def load_sources(path: Path = SOURCES_YAML) -> dict[str, Source]:
    raw = yaml.safe_load(path.read_text())
    return {name: Source.model_validate(cfg) for name, cfg in raw["sources"].items()}


def load_lock(path: Path = LOCK_JSON) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data: dict[str, dict[str, Any]] = json.loads(path.read_text())
    return data


def save_lock(lock: dict[str, dict[str, Any]], path: Path = LOCK_JSON) -> None:
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def http_download(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for block in resp.iter_bytes(1 << 20):
                fh.write(block)


def safe_extract_zip(archive: Path, target: Path) -> int:
    """Extract, refusing members that would land outside ``target`` (zip-slip)."""
    target = target.resolve()
    n = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            dest = (target / member.filename).resolve()
            if not dest.is_relative_to(target):
                raise FetchError(f"zip member escapes target dir: {member.filename!r}")
        for member in zf.infolist():
            zf.extract(member, target)
            n += 1
    return n


def _expected_sha(name: str, src: Source, lock: dict[str, dict[str, Any]]) -> str | None:
    if src.sha256:
        return src.sha256
    entry = lock.get(name)
    if entry and isinstance(entry.get("sha256"), str):
        return str(entry["sha256"])
    return None


def fetch_http(
    name: str,
    src: Source,
    *,
    raw_dir: Path,
    lock: dict[str, dict[str, Any]],
    downloader: Downloader = http_download,
    force: bool = False,
) -> Path:
    out_dir = raw_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = src.filename or src.url.rsplit("/", 1)[-1].split("?", 1)[0]
    final = out_dir / filename
    expected = _expected_sha(name, src, lock)

    if final.exists() and not force:
        actual = sha256_file(final)
        if expected and actual != expected:
            raise FetchError(f"{name}: existing file checksum mismatch; rerun with --force")
        typer.echo(f"{name}: already present ({actual[:12]})")
    else:
        part = final.with_suffix(final.suffix + ".part")
        part.unlink(missing_ok=True)
        typer.echo(f"{name}: downloading {src.url}")
        downloader(src.url, part)
        actual = sha256_file(part)
        if expected and actual != expected:
            part.unlink(missing_ok=True)
            raise FetchError(
                f"{name}: checksum mismatch\n  expected {expected}\n  actual   {actual}"
            )
        part.replace(final)
        if not expected:
            typer.echo(f"{name}: no checksum on record; recorded sha256={actual}")

    extracted = 0
    if src.extract == "zip":
        extracted = safe_extract_zip(final, out_dir)
        typer.echo(f"{name}: extracted {extracted} members")

    lock[name] = {
        "kind": "http",
        "sha256": actual,
        "filename": filename,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "extracted_members": extracted,
    }
    return final


def fetch_git(
    name: str,
    src: Source,
    *,
    raw_dir: Path,
    lock: dict[str, dict[str, Any]],
    force: bool = False,
) -> Path:
    repo = raw_dir / name / "repo"
    if repo.exists() and force:
        shutil.rmtree(repo)
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1"]
        if src.ref:
            cmd += ["--branch", src.ref]
        cmd += [src.url, str(repo)]
        typer.echo(f"{name}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    locked = lock.get(name, {}).get("commit")
    if locked and locked != head and not force:
        typer.echo(f"{name}: WARNING checked-out commit {head[:12]} != locked {locked[:12]}")
    lock[name] = {
        "kind": "git",
        "commit": head,
        "ref": src.ref,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return repo


def fetch_one(
    name: str,
    *,
    sources: dict[str, Source] | None = None,
    raw_dir: Path = RAW_DIR,
    lock_path: Path = LOCK_JSON,
    downloader: Downloader = http_download,
    force: bool = False,
) -> Path | None:
    sources = sources or load_sources()
    if name not in sources:
        raise FetchError(f"unknown source {name!r}; known: {sorted(sources)}")
    src = sources[name]
    lock = load_lock(lock_path)
    if src.kind == "manual":
        typer.echo(f"{name}: manual source - {src.notes or src.url}")
        return None
    if src.kind == "http":
        result = fetch_http(
            name, src, raw_dir=raw_dir, lock=lock, downloader=downloader, force=force
        )
    else:
        result = fetch_git(name, src, raw_dir=raw_dir, lock=lock, force=force)
    save_lock(lock, lock_path)
    return result


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    source: str = typer.Option("all", help="source name from sources.yaml, or 'all'"),
    force: bool = typer.Option(False, help="re-download / re-clone"),
    max_size_mb: float = typer.Option(500, help="with --source all: skip sources larger than this"),
    list_: bool = typer.Option(False, "--list", help="list sources and exit"),
) -> None:
    sources = load_sources()
    if list_:
        lock = load_lock()
        for name, src in sources.items():
            state = "fetched" if name in lock else "-"
            size = f"{src.size_mb:g} MB" if src.size_mb else "?"
            typer.echo(
                f"{name:28s} {src.kind:6s} {src.tier}  {src.license:14s} {size:>10s}  {state}"
            )
        raise typer.Exit()
    names = list(sources) if source == "all" else [source]
    failures: list[str] = []
    for name in names:
        src = sources[name]
        if source == "all" and src.size_mb and src.size_mb > max_size_mb:
            typer.echo(f"{name}: skipped ({src.size_mb:g} MB > {max_size_mb:g} MB)")
            continue
        try:
            fetch_one(name, sources=sources, force=force)
        except (FetchError, httpx.HTTPError, subprocess.CalledProcessError) as exc:
            failures.append(name)
            typer.echo(f"{name}: FAILED - {exc}", err=True)
    if failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    sys.exit(app())
