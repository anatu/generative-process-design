"""Scaffold sanity: package imports, layout matches BLUEPRINT §2.2."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SUBPACKAGES = [
    "gpd.canonical",
    "gpd.validity",
    "gpd.validity.rules",
    "gpd.validity.sim",
    "gpd.corpus",
    "gpd.synth",
    "gpd.extract.text",
    "gpd.extract.image",
    "gpd.models",
    "gpd.eval",
    "gpd.serve",
    "gpd.util",
]


def test_version() -> None:
    import gpd

    assert gpd.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_importable(name: str) -> None:
    importlib.import_module(name)


def test_data_dirs_exist_and_are_ignored() -> None:
    for sub in ("raw", "interim", "corpus"):
        assert (ROOT / "data" / sub / ".gitkeep").exists()
    ignore = (ROOT / ".gitignore").read_text()
    assert "data/raw/*" in ignore
    assert "data/corpus/*" in ignore
