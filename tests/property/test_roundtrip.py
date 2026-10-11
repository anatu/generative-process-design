"""Graph <-> SFILES round trips (BLUEPRINT WS0 acceptance)."""

from __future__ import annotations

import networkx as nx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from gpd.canonical import sfiles, vocab
from tests.strategies import flowsheet_graphs, relabel_and_shuffle

pytestmark = pytest.mark.property


def _material_degrees(g: nx.MultiDiGraph, n: str) -> tuple[int, int]:
    n_in = sum(1 for _, _, d in g.in_edges(n, data=True) if not d["tags"]["signal"])
    n_out = sum(1 for _, _, d in g.out_edges(n, data=True) if not d["tags"]["signal"])
    return n_in, n_out


@given(flowsheet_graphs())
def test_strategy_respects_degree_constraints(g: nx.MultiDiGraph) -> None:
    for n in g.nodes:
        cls = vocab.node_class(n)
        n_in, n_out = _material_degrees(g, n)
        assert vocab.DEGREE_CONSTRAINTS[cls].check(n_in, n_out), (n, n_in, n_out)


@given(flowsheet_graphs())
def test_graph_sfiles_graph_is_isomorphic(g: nx.MultiDiGraph) -> None:
    s = sfiles.graph_to_sfiles(g)
    h = sfiles.sfiles_to_graph(s)
    assert h.number_of_nodes() == g.number_of_nodes()
    assert h.number_of_edges() == g.number_of_edges()
    assert sfiles.is_isomorphic(g, h)


@given(flowsheet_graphs())
def test_canonical_string_is_idempotent(g: nx.MultiDiGraph) -> None:
    s = sfiles.graph_to_sfiles(g)
    assert sfiles.canonical_sfiles(s) == s


@given(flowsheet_graphs(), st.integers(0, 2**32 - 1))
def test_canonical_string_invariant_under_relabel_and_reorder(
    g: nx.MultiDiGraph, seed: int
) -> None:
    h = relabel_and_shuffle(g, seed)
    assert sfiles.graph_to_sfiles(h) == sfiles.graph_to_sfiles(g)


@given(flowsheet_graphs())
def test_tokenize_covers_whole_string(g: nx.MultiDiGraph) -> None:
    s = sfiles.graph_to_sfiles(g)
    assert "".join(sfiles.tokenize(s)) == s


@given(flowsheet_graphs())
def test_canonicalize_returns_parse_of_string(g: nx.MultiDiGraph) -> None:
    s, norm = sfiles.canonicalize(g)
    assert sfiles.graph_to_sfiles(norm) == s
