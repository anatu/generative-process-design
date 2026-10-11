"""vocab.py must mirror the pinned SFILES2 submodule exactly (BLUEPRINT WS0)."""

from __future__ import annotations

import importlib.metadata
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from gpd.canonical import vocab

ROOT = Path(__file__).resolve().parents[2]
SUBMODULE = ROOT / "third_party" / "SFILES2"


def test_mapping_matches_upstream() -> None:
    from Flowsheet_Class.OntoCape_SFILES_mapping import OntoCape_SFILES_map

    assert dict(vocab.ONTOCAPE_TO_SFILES) == OntoCape_SFILES_map, (
        "upstream SFILES2 mapping changed; update gpd/canonical/vocab.py deliberately"
    )


def test_installed_version_is_pinned() -> None:
    assert importlib.metadata.version("SFILES2") == vocab.SFILES2_PINNED_VERSION


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_submodule_commit_is_pinned() -> None:
    if not (SUBMODULE / ".git").exists():
        pytest.skip("submodule not checked out")
    head = subprocess.run(
        ["git", "-C", str(SUBMODULE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == vocab.SFILES2_PINNED_COMMIT


def test_class_names_are_plain_letters() -> None:
    for cls in vocab.UNIT_CLASSES:
        assert re.fullmatch(r"[A-Za-z]+", cls), cls


def test_degree_constraints_cover_vocab_exactly() -> None:
    assert set(vocab.DEGREE_CONSTRAINTS) == set(vocab.UNIT_CLASSES)


def test_special_classes_in_vocab() -> None:
    for cls in (vocab.RAW, vocab.PROD, vocab.CONTROL, vocab.UNK):
        assert cls in vocab.UNIT_CLASSES
    assert vocab.CONTROL not in vocab.MATERIAL_CLASSES
    assert vocab.TERMINAL_CLASSES <= vocab.MATERIAL_CLASSES


@pytest.mark.parametrize(
    ("n_in", "n_out", "spec", "ok"),
    [
        (1, 1, vocab.DegreeSpec(1, 1, 1, 1), True),
        (0, 1, vocab.DegreeSpec(1, 1, 1, 1), False),
        (1, 2, vocab.DegreeSpec(1, 1, 1, 1), False),
        (3, 1, vocab.DegreeSpec(2, None, 1, 1), True),
        (1, 5, vocab.DegreeSpec(1, None, 2, None), True),
        (1, 1, vocab.DegreeSpec(1, None, 2, None), False),
        (0, 0, vocab.DegreeSpec(0, 0, 0, 0), True),
    ],
)
def test_degree_spec_check(n_in: int, n_out: int, spec: vocab.DegreeSpec, ok: bool) -> None:
    assert spec.check(n_in, n_out) is ok


def test_terminal_degree_specs() -> None:
    assert vocab.DEGREE_CONSTRAINTS["raw"].check(0, 1)
    assert not vocab.DEGREE_CONSTRAINTS["raw"].check(1, 1)
    assert vocab.DEGREE_CONSTRAINTS["prod"].check(1, 0)
    assert not vocab.DEGREE_CONSTRAINTS["prod"].check(1, 1)
    assert vocab.DEGREE_CONSTRAINTS["C"].check(0, 0)
    assert not vocab.DEGREE_CONSTRAINTS["C"].check(1, 0)


@pytest.mark.parametrize(
    ("node_id", "expected"),
    [
        ("hex-1", vocab.NodeId("hex", 1)),
        ("hex-12/2", vocab.NodeId("hex", 12, 2)),
        ("C-3", vocab.NodeId("C", 3)),
        ("raw-0", vocab.NodeId("raw", 0)),
    ],
)
def test_parse_node_id(node_id: str, expected: vocab.NodeId) -> None:
    parsed = vocab.parse_node_id(node_id)
    assert parsed == expected
    assert str(parsed) == node_id


@pytest.mark.parametrize("bad", ["hex", "hex1", "-1", "hex-", "hex-1/", "hex-1/2/3", "he x-1", ""])
def test_parse_node_id_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match="malformed"):
        vocab.parse_node_id(bad)


def test_node_class_and_membership() -> None:
    assert vocab.node_class("dist-4") == "dist"
    assert vocab.is_known_class("dist")
    assert not vocab.is_known_class("unk")
    assert not vocab.is_known_class("turbine")
