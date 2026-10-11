"""Paired-string records for seq2seq tasks (BLUEPRINT WS2.1, WS5.3).

String-native corpora (Hirtreiter) do not pass through the IR. They land here:
the published strings verbatim for comparability, plus our canonical forms and
grammar flags for stats and filtering.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpd.canonical import sfiles
from gpd.corpus.record import Split

Variant = Literal["standard", "wo_valves"]


class PairRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    license: str
    is_real: bool
    variant: Variant = "standard"
    split: Split
    split_origin: Literal["upstream", "ours"] = "upstream"
    pfd: str = Field(min_length=1)  # as published
    pid: str = Field(min_length=1)  # as published
    pfd_canonical: str | None = None  # None when SFILES2 could not parse it
    pid_canonical: str | None = None
    pfd_parses: bool  # G1
    pid_parses: bool
    pfd_is_canonical: bool  # G2: published string == its canonical form
    pid_is_canonical: bool
    pfd_tokens: int = Field(ge=1)
    pid_tokens: int = Field(ge=1)


def _canon(s: str) -> tuple[str | None, bool, bool]:
    """(canonical string, parses, is_canonical)."""
    try:
        c = sfiles.canonical_sfiles(s)
    except sfiles.CanonicalizationError:
        return None, False, False
    return c, True, c == s


def make_pair(
    *,
    id: str,
    source: str,
    license: str,
    is_real: bool,
    variant: Variant,
    split: Split,
    pfd: str,
    pid: str,
    canonicalize: bool = True,
) -> PairRecord:
    pfd_c, pfd_p, pfd_ok = _canon(pfd) if canonicalize else (None, False, False)
    pid_c, pid_p, pid_ok = _canon(pid) if canonicalize else (None, False, False)
    return PairRecord(
        id=id,
        source=source,
        license=license,
        is_real=is_real,
        variant=variant,
        split=split,
        pfd=pfd,
        pid=pid,
        pfd_canonical=pfd_c,
        pid_canonical=pid_c,
        pfd_parses=pfd_p,
        pid_parses=pid_p,
        pfd_is_canonical=pfd_ok,
        pid_is_canonical=pid_ok,
        pfd_tokens=len(sfiles.tokenize(pfd)),
        pid_tokens=len(sfiles.tokenize(pid)),
    )


def write_jsonl(records: Iterable[PairRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[PairRecord]:
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield PairRecord.model_validate_json(line)


def read_raw_pairs(path: Path) -> Iterator[tuple[str, str]]:
    """Hirtreiter files: JSON Lines of {"PFD","PID"}, or one object {"data": [...]}."""
    text = path.read_text()
    stripped = text.lstrip()
    if stripped.startswith('{"data"'):
        for row in json.loads(text)["data"]:
            yield row["PFD"], row["PID"]
        return
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            yield row["PFD"], row["PID"]
