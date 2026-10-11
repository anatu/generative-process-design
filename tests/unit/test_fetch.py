from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fetch", ROOT / "scripts" / "fetch.py")
assert spec and spec.loader
fetch = importlib.util.module_from_spec(spec)
sys.modules["fetch"] = fetch
spec.loader.exec_module(fetch)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _fake_downloader(payload: bytes):  # type: ignore[no-untyped-def]
    def dl(url: str, dest: Path) -> None:
        dest.write_bytes(payload)

    return dl


def _src(sha: str | None, extract: str | None = "zip") -> fetch.Source:
    return fetch.Source(
        kind="http",
        url="https://x/y.zip",
        filename="y.zip",
        sha256=sha,
        extract=extract,
        license="MIT",
        tier="A",
    )


def test_sources_yaml_loads_and_hirtreiter_is_pinned() -> None:
    sources = fetch.load_sources()
    assert (
        sources["hirtreiter"].sha256
        == "f3216d96c04286813f8c55df249fb6d376d58944c11e85fa9bc581f403256cc7"
    )
    assert sources["hirtreiter"].kind == "http"
    assert all(s.license for s in sources.values())


def test_download_verify_extract_and_lock(tmp_path: Path) -> None:
    payload = _zip_bytes({"d/a.json": b'{"PFD":"(raw)(prod)"}\n'})
    sha = hashlib.sha256(payload).hexdigest()
    lock_path = tmp_path / "lock.json"
    out = fetch.fetch_one(
        "s",
        sources={"s": _src(sha)},
        raw_dir=tmp_path / "raw",
        lock_path=lock_path,
        downloader=_fake_downloader(payload),
    )
    assert out == tmp_path / "raw" / "s" / "y.zip"
    assert (tmp_path / "raw" / "s" / "d" / "a.json").read_text().startswith('{"PFD"')
    lock = json.loads(lock_path.read_text())
    assert lock["s"]["sha256"] == sha and lock["s"]["extracted_members"] == 1


def test_checksum_mismatch_leaves_no_file(tmp_path: Path) -> None:
    payload = _zip_bytes({"a": b"1"})
    with pytest.raises(fetch.FetchError, match="checksum mismatch"):
        fetch.fetch_one(
            "s",
            sources={"s": _src("0" * 64)},
            raw_dir=tmp_path / "raw",
            lock_path=tmp_path / "lock.json",
            downloader=_fake_downloader(payload),
        )
    assert not list((tmp_path / "raw" / "s").glob("y.zip*"))
    assert not (tmp_path / "lock.json").exists()


def test_unpinned_source_records_then_verifies(tmp_path: Path) -> None:
    payload = _zip_bytes({"a": b"1"})
    lock_path = tmp_path / "lock.json"
    kwargs = dict(sources={"s": _src(None)}, raw_dir=tmp_path / "raw", lock_path=lock_path)
    fetch.fetch_one("s", downloader=_fake_downloader(payload), **kwargs)  # type: ignore[arg-type]
    recorded = json.loads(lock_path.read_text())["s"]["sha256"]
    assert recorded == hashlib.sha256(payload).hexdigest()
    # Second fetch with different bytes must fail against the recorded hash.
    with pytest.raises(fetch.FetchError, match="checksum mismatch"):
        fetch.fetch_one("s", downloader=_fake_downloader(payload + b"x"), force=True, **kwargs)  # type: ignore[arg-type]


def test_zip_slip_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", b"x")
    with pytest.raises(fetch.FetchError, match="escapes"):
        fetch.safe_extract_zip(archive, tmp_path / "target")
    assert not (tmp_path / "escape.txt").exists()


def test_manual_source_is_noop(tmp_path: Path) -> None:
    src = fetch.Source(kind="manual", url="https://x", license="pending", tier="A", notes="scrape")
    assert (
        fetch.fetch_one("m", sources={"m": src}, raw_dir=tmp_path, lock_path=tmp_path / "l.json")
        is None
    )


def test_unknown_source() -> None:
    with pytest.raises(fetch.FetchError, match="unknown source"):
        fetch.fetch_one("nope", sources={})
