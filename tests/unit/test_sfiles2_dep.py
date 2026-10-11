"""The vendored SFILES2 dependency behaves as the blueprint assumes (BLUEPRINT WS0).

Covers the week-1 verification items that touch SFILES2:
- the random flowsheet generator is absent upstream (WS2.2 fallback stands);
- upstream demo strings round-trip;
- the 9 real DWSIM graphs shipped in the submodule convert via ``create_from_nx``.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[2]
SUBMODULE = ROOT / "third_party" / "SFILES2"
PICKLES = SUBMODULE / "Real_flowsheets" / "DWSim_pickle_files"
SFILES_FIXTURES = ROOT / "tests" / "fixtures" / "sfiles"

_TAG_RE = re.compile(r"\{.*?\}")
_LEGACY_NODE_RE = re.compile(r"([a-zA-Z]+)([0-9]+)")


def test_random_generator_is_absent_upstream() -> None:
    """BLUEPRINT decision log: generator absent -> own motif generator in WS2.2.

    If this fails, upstream started shipping ``PID_generation`` and the WS2.2
    fallback should be revisited.
    """
    from Flowsheet_Class.flowsheet import PID_generator

    assert PID_generator is False


def _fixture_strings() -> list[pytest.param]:
    return [
        pytest.param(p.read_text().strip(), id=p.stem)
        for p in sorted(SFILES_FIXTURES.glob("*.sfiles"))
    ]


@pytest.mark.parametrize("sfiles", _fixture_strings())
def test_demo_strings_roundtrip_exactly(sfiles: str) -> None:
    from Flowsheet_Class.flowsheet import Flowsheet

    fs = Flowsheet()
    fs.create_from_sfiles(sfiles)
    fs.sfiles = ""
    fs.convert_to_sfiles(version="v2")
    assert fs.sfiles == sfiles


def _pickle_paths() -> list[pytest.param]:
    if not PICKLES.exists():
        return []
    return [pytest.param(p, id=p.stem) for p in sorted(PICKLES.glob("*.pkl"))]


@pytest.mark.parametrize("path", _pickle_paths())
def test_real_dwsim_graphs_convert(path: Path) -> None:
    """Mirror of upstream ``Real_flowsheets/test_and_load_new_data.py``.

    Loads a pickled DiGraph, relabels ``hex1`` -> ``hex-1``, converts to SFILES
    v2, parses it back, and checks the tag-stripped string is stable.
    """
    from Flowsheet_Class.flowsheet import Flowsheet

    with path.open("rb") as fh:
        g = nx.MultiDiGraph(pickle.load(fh))  # data from the pinned MIT submodule

    relabel: dict[str, str] = {}
    for n in g.nodes:
        m = _LEGACY_NODE_RE.match(n)
        assert m is not None, n
        relabel[n] = f"{m.group(1)}-{m.group(2)}"
    g = nx.relabel_nodes(g, relabel)

    fs = Flowsheet(OntoCapeConformity=True)
    fs.create_from_nx(g)
    n_edges_before = fs.state.number_of_edges()
    fs.convert_to_sfiles(version="v2", remove_hex_tags=True)
    sfiles_1 = fs.sfiles
    assert sfiles_1

    fs.create_from_sfiles(overwrite_nx=True, merge_HI_nodes=False)
    assert fs.state.number_of_edges() == n_edges_before
    fs.convert_to_sfiles(version="v2", remove_hex_tags=True)
    sfiles_2 = fs.sfiles

    assert len(sfiles_1) == len(sfiles_2)
    assert _TAG_RE.sub("", sfiles_1) == _TAG_RE.sub("", sfiles_2)


def test_nine_real_graphs_present() -> None:
    assert len(list(PICKLES.glob("*.pkl"))) == 9
