from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given

from gpd.canonical import sfiles
from gpd.canonical.tokenizer import SPECIALS, Vocab
from tests.strategies import flowsheet_graphs

CORPUS = [
    "(raw)(hex){1}(dist)[{tout}(prod)]{bout}(prod)n|(raw)(hex){1}(prod)",
    "(raw)(C){FC}_1(v)<_1(prod)",
    "(raw)(pp)<1(splt)[(hex)<&|(raw)&|(prod)]1",
    "(raw)(mix)<&|(raw)(C){FC}_5(v)&<_5|(pp)(prod)",
]


def test_build_sorted_and_specials_first() -> None:
    v = Vocab.build(CORPUS)
    assert v.itos[: len(SPECIALS)] == list(SPECIALS)
    rest = v.itos[len(SPECIALS) :]
    assert rest == sorted(rest)
    assert v.pad_id == 0 and v.bos_id == 1 and v.eos_id == 2 and v.unk_id == 3


@pytest.mark.parametrize("s", CORPUS)
def test_roundtrip_exact(s: str) -> None:
    v = Vocab.build(CORPUS)
    ids = v.encode(s)
    assert ids[0] == v.bos_id and ids[-1] == v.eos_id
    assert v.decode(ids) == s
    assert v.decode(v.encode(s, add_special=False)) == s
    assert not v.unknown_tokens(s)


def test_unknown_maps_to_unk_and_is_reported() -> None:
    v = Vocab.build(CORPUS[:1])
    s = "(raw)(turb)(prod)"
    assert v.unknown_tokens(s) == ["(turb)"]
    ids = v.encode(s, add_special=False)
    assert ids[1] == v.unk_id


def test_decode_stops_at_eos_and_skips_pad() -> None:
    v = Vocab.build(CORPUS)
    ids = [v.bos_id, v.stoi["(raw)"], v.stoi["(prod)"], v.eos_id, v.stoi["(hex)"], v.pad_id]
    assert v.decode(ids) == "(raw)(prod)"
    assert v.decode(ids, stop_at_eos=False) == "(raw)(prod)(hex)"


def test_save_load(tmp_path: Path) -> None:
    v = Vocab.build(CORPUS)
    v.save(tmp_path / "vocab.json")
    w = Vocab.load(tmp_path / "vocab.json")
    assert w.itos == v.itos
    with pytest.raises(ValueError, match="version"):
        (tmp_path / "bad.json").write_text('{"version": 0, "tokens": []}')
        Vocab.load(tmp_path / "bad.json")


def test_min_freq() -> None:
    v = Vocab.build(["(raw)(prod)", "(raw)(hex)(prod)"], min_freq=2)
    assert "(hex)" not in v and "(raw)" in v


@given(flowsheet_graphs())
def test_roundtrip_on_generated_flowsheets(g) -> None:  # type: ignore[no-untyped-def]
    s = sfiles.graph_to_sfiles(g)
    v = Vocab.build([s])
    assert v.decode(v.encode(s)) == s
