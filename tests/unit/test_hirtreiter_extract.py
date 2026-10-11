from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from gpd.corpus.pairs import read_jsonl, read_raw_pairs

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "hirtreiter"

spec = importlib.util.spec_from_file_location(
    "hirtreiter_extract", ROOT / "extractors" / "hirtreiter" / "extract.py"
)
assert spec and spec.loader
ext = importlib.util.module_from_spec(spec)
sys.modules["hirtreiter_extract"] = ext
spec.loader.exec_module(ext)


def test_read_raw_pairs_handles_both_layouts() -> None:
    jsonl = list(read_raw_pairs(FIX / "train_data_100.json"))
    wrapped = list(read_raw_pairs(FIX / "train_data_augm_100.json"))
    assert len(jsonl) == 3 and len(wrapped) == 2
    assert jsonl[0] == wrapped[0]
    assert all(p.startswith("(raw)") and "(C)" in q for p, q in jsonl)


def test_run_writes_splits_and_stats(tmp_path: Path) -> None:
    stats = ext.run(
        raw_dir=FIX,
        out_dir=tmp_path,
        train_size="100",
        variant="standard",
        include_augmented=True,
        canonicalize=True,
    )
    out = tmp_path / "standard"
    assert {p.name for p in out.glob("*.jsonl")} == {
        "train_100.jsonl",
        "val.jsonl",
        "test.jsonl",
        "train_augm_100.jsonl",
    }
    train = list(read_jsonl(out / "train_100.jsonl"))
    assert len(train) == 3
    assert all(r.split == "train" and not r.is_real and r.license == "CC-BY-4.0" for r in train)
    assert all(r.pfd_parses and r.pid_parses for r in train)
    assert {r.split for r in read_jsonl(out / "val.jsonl")} == {"val"}
    assert {r.split for r in read_jsonl(out / "test.jsonl")} == {"test"}
    saved = json.loads((out / "stats.json").read_text())
    assert saved["train_100"]["n"] == 3 and stats["val"]["n"] == 2
    assert saved["test"]["pid_tokens_max"] >= saved["test"]["pid_tokens_p99"]


def test_missing_variant_files_are_skipped(tmp_path: Path) -> None:
    stats = ext.run(
        raw_dir=FIX,
        out_dir=tmp_path,
        train_size="10k",
        variant="wo_valves",
        include_augmented=False,
        canonicalize=False,
    )
    # only train_data_10k_wo_valves.json exists in the fixture set
    assert set(stats) == {"train_10k"}
    rec = next(read_jsonl(tmp_path / "wo_valves" / "train_10k.jsonl"))
    assert rec.variant == "wo_valves" and rec.pfd_canonical is None and not rec.pfd_parses
