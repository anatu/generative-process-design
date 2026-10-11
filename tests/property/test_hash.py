"""Dedup keys are invariants of the flowsheet, not of its encoding (BLUEPRINT §3.4)."""

from __future__ import annotations

import networkx as nx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from gpd.canonical import hashing, sfiles
from gpd.canonical.to_graph import empty_tags
from tests.strategies import flowsheet_graphs, relabel_and_shuffle

pytestmark = pytest.mark.property


@given(flowsheet_graphs(), st.integers(0, 2**32 - 1))
def test_hashes_invariant_under_relabel_and_reorder(g: nx.MultiDiGraph, seed: int) -> None:
    h = relabel_and_shuffle(g, seed)
    assert hashing.canonical_hash(sfiles.graph_to_sfiles(g)) == hashing.canonical_hash(
        sfiles.graph_to_sfiles(h)
    )
    assert hashing.wl_hash(g) == hashing.wl_hash(h)


@given(flowsheet_graphs())
def test_wl_hash_stable_across_canonical_roundtrip(g: nx.MultiDiGraph) -> None:
    _, norm = sfiles.canonicalize(g)
    assert hashing.wl_hash(norm) == hashing.wl_hash(g)


@given(flowsheet_graphs())
def test_adding_a_unit_changes_both_hashes(g: nx.MultiDiGraph) -> None:
    h = g.copy()
    prod = next(n for n in h.nodes if n.startswith("prod-"))
    (feeder,) = [u for u, _ in h.in_edges(prod)]
    key = next(iter(h[feeder][prod]))
    tags = h[feeder][prod][key]["tags"]
    h.remove_edge(feeder, prod, key)
    n_v = 1 + sum(1 for n in h.nodes if n.startswith("v-"))
    valve = f"v-{n_v}"
    h.add_node(valve)
    h.add_edge(feeder, valve, tags=tags)
    h.add_edge(valve, prod, tags=empty_tags())
    assert hashing.canonical_hash(sfiles.graph_to_sfiles(g)) != hashing.canonical_hash(
        sfiles.graph_to_sfiles(h)
    )
    assert hashing.wl_hash(g) != hashing.wl_hash(h)
