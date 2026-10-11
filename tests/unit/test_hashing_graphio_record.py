from __future__ import annotations

import networkx as nx
import pytest

from gpd.canonical import hashing, sfiles
from gpd.canonical.graphio import graph_from_json, graph_stats, graph_to_json
from gpd.canonical.ir import IRFlowsheet
from gpd.corpus.record import Provenance, Record, Validity

SIMPLE = "(raw)(hex)(dist)[{bout}(prod)]{tout}(prod)"
RECYCLE = (
    "(raw)(flash)[{tout}(prod)]{bout}(splt)[(prod)](r)<&|(raw)(flash){tout}&{bout}(prod)|(prod)"
)
CTRL = "(raw)(C){FC}_1(v)<_1(prod)"


@pytest.mark.parametrize(
    ("node_id", "expected"),
    [("hex-1", "hex"), ("hex-1/2", "hex"), ("C-3/FC", "C/FC"), ("C-3", "C"), ("dist-12", "dist")],
)
def test_node_type(node_id: str, expected: str) -> None:
    assert hashing.node_type(node_id) == expected
    assert hashing.node_class(node_id) == expected.split("/")[0]


def test_canonical_hash_is_sha256_of_string() -> None:
    h = hashing.canonical_hash(SIMPLE)
    assert len(h) == 64 and h != hashing.canonical_hash(SIMPLE + " ")


def test_wl_hash_differs_for_different_flowsheets() -> None:
    assert hashing.wl_hash(sfiles.sfiles_to_graph(SIMPLE)) != hashing.wl_hash(
        sfiles.sfiles_to_graph(RECYCLE)
    )


def test_wl_hash_distinguishes_controller_function() -> None:
    a = hashing.wl_hash(sfiles.sfiles_to_graph(CTRL))
    b = hashing.wl_hash(sfiles.sfiles_to_graph(CTRL.replace("{FC}", "{LC}")))
    assert a != b


def test_graph_json_roundtrip_preserves_everything() -> None:
    g = sfiles.sfiles_to_graph(RECYCLE)
    g.nodes["r-1"]["src_class"] = "Reactor_CSTR"
    h = graph_from_json(graph_to_json(g))
    assert sorted(h.nodes(data=True)) == sorted(g.nodes(data=True))
    assert sorted(h.edges(keys=True, data=True), key=str) == sorted(
        g.edges(keys=True, data=True), key=str
    )


def test_graph_json_version_guard() -> None:
    with pytest.raises(ValueError, match="version"):
        graph_from_json({"v": 99, "nodes": [], "edges": []})


@pytest.mark.parametrize(
    ("s", "n_units", "n_streams", "recycle", "control"),
    [
        (SIMPLE, 2, 4, False, False),
        # `<&|...&|` is a second feed joining the reactor, not a recycle.
        (RECYCLE, 4, 9, False, False),
        (CTRL, 1, 3, False, True),  # controller is not a unit; signal edge is not a stream
    ],
)
def test_graph_stats(s: str, n_units: int, n_streams: int, recycle: bool, control: bool) -> None:
    st = graph_stats(sfiles.sfiles_to_graph(s))
    assert (st.n_units, st.n_streams, st.has_recycle, st.has_control) == (
        n_units,
        n_streams,
        recycle,
        control,
    )


def test_graph_stats_detects_true_recycle() -> None:
    from gpd.canonical.to_graph import empty_tags

    g = nx.MultiDiGraph()
    for u, v in [("raw-1", "r-1"), ("r-1", "splt-1"), ("splt-1", "prod-1"), ("splt-1", "r-1")]:
        g.add_edge(u, v, tags=empty_tags())
    assert graph_stats(g).has_recycle
    s = sfiles.graph_to_sfiles(g)
    assert graph_stats(sfiles.sfiles_to_graph(s)).has_recycle


def test_record_schema_and_graph_restore() -> None:
    ir = IRFlowsheet(
        source="test",
        source_uri="u",
        source_file="f",
        source_sha256="b" * 64,
        license="MIT",
        tier="A",
        domain="chemical",
        extraction_method="manual",
        nodes=[],
        edges=[],
    )
    g = sfiles.sfiles_to_graph(SIMPLE)
    stats = graph_stats(g)
    rec = Record(
        id=Record.make_id("test", "b" * 64, 0),
        sfiles=SIMPLE,
        graph=graph_to_json(g),
        canonical_hash=hashing.canonical_hash(SIMPLE),
        wl_hash=hashing.wl_hash(g),
        n_units=stats.n_units,
        n_streams=stats.n_streams,
        has_recycle=stats.has_recycle,
        has_control=stats.has_control,
        provenance=Provenance.from_ir(ir),
        validity=Validity(grammar_ok=True),
        is_real=True,
    )
    assert rec.id == "test:bbbbbbbbbbbb:0"
    again = Record.model_validate_json(rec.model_dump_json())
    assert sfiles.graph_to_sfiles(again.to_graph()) == SIMPLE
    assert again.split is None and again.duplicates == []
