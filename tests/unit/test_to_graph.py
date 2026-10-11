from __future__ import annotations

import pytest

from gpd.canonical import sfiles
from gpd.canonical.ir import IREdge, IRFlowsheet, IRNode
from gpd.canonical.to_graph import ir_to_graph
from gpd.canonical.typemap import TypeMap

SHA = "a" * 64

TM = TypeMap(
    source="test",
    classes={"Heater": "hex", "Pump": "pp", "Column": "dist", "Mixer": "mix", "Vessel": "flash"},
    ports={"Distillate": "tout", "Bottoms": "bout", "HotIn": "hot_in", "ColdOut": "cold_out"},
)


def _ir(nodes: list[IRNode], edges: list[IREdge]) -> IRFlowsheet:
    return IRFlowsheet(
        source="test",
        source_uri="u",
        source_file="f",
        source_sha256=SHA,
        license="MIT",
        tier="A",
        domain="chemical",
        extraction_method="manual",
        nodes=nodes,
        edges=edges,
    )


def _tags(g, u, v):  # type: ignore[no-untyped-def]
    return [d["tags"] for d in g.get_edge_data(u, v).values()]


def test_boundary_streams_become_raw_and_prod() -> None:
    ir = _ir(
        [IRNode(id="p", source_class="ns.Pump")],
        [IREdge(src=None, dst="p"), IREdge(src="p", dst=None)],
    )
    res = ir_to_graph(ir, TM)
    assert set(res.graph.nodes) == {"raw-1", "pp-1", "prod-1"}
    assert res.graph.has_edge("raw-1", "pp-1") and res.graph.has_edge("pp-1", "prod-1")
    assert not res.unk_nodes
    assert sfiles.graph_to_sfiles(res.graph) == "(raw)(pp)(prod)"


def test_numbering_per_class_in_ir_order() -> None:
    ir = _ir(
        [
            IRNode(id="a", source_class="Heater"),
            IRNode(id="b", source_class="Pump"),
            IRNode(id="c", source_class="Heater"),
        ],
        [
            IREdge(src=None, dst="a"),
            IREdge(src="a", dst="b"),
            IREdge(src="b", dst="c"),
            IREdge(src="c", dst=None),
        ],
    )
    g = ir_to_graph(ir, TM).graph
    assert g.nodes["hex-1"]["src_id"] == "a"
    assert g.nodes["hex-2"]["src_id"] == "c"
    assert g.nodes["pp-1"]["src_class"] == "Pump"


def test_column_ports_become_col_tags() -> None:
    ir = _ir(
        [IRNode(id="d", source_class="Column")],
        [
            IREdge(src=None, dst="d"),
            IREdge(src="d", dst=None, src_port="Distillate"),
            IREdge(src="d", dst=None, src_port="Bottoms"),
        ],
    )
    g = ir_to_graph(ir, TM).graph
    tags = sorted(t["col"][0] for v in ("prod-1", "prod-2") for t in _tags(g, "dist-1", v))
    assert tags == ["bout", "tout"]
    assert sfiles.graph_to_sfiles(g) == "(raw)(dist)[{bout}(prod)]{tout}(prod)"


def test_hex_side_ports_become_numeric_he_tags() -> None:
    ir = _ir(
        [IRNode(id="h", source_class="Heater")],
        [
            IREdge(src=None, dst="h", dst_port="HotIn"),
            IREdge(src="h", dst=None, src_port="ColdOut"),
        ],
    )
    g = ir_to_graph(ir, TM).graph
    assert _tags(g, "raw-1", "hex-1")[0]["he"] == ["1_in"]
    assert _tags(g, "hex-1", "prod-1")[0]["he"] == ["2_out"]


def test_energy_edges_dropped_and_counted() -> None:
    ir = _ir(
        [IRNode(id="h", source_class="Heater"), IRNode(id="u", source_class="Utility")],
        [
            IREdge(src=None, dst="h"),
            IREdge(src="h", dst=None),
            IREdge(src="u", dst="h", kind="energy"),
        ],
    )
    res = ir_to_graph(ir, TM)
    assert res.dropped_energy_edges == 1
    assert res.graph.in_degree("hex-1") == 1
    assert any("energy" in n for n in res.notes)


def test_unmapped_class_becomes_x_and_is_flagged() -> None:
    ir = _ir(
        [IRNode(id="z", source_class="ns.Widget")],
        [IREdge(src=None, dst="z"), IREdge(src="z", dst=None)],
    )
    res = ir_to_graph(ir, TM)
    assert "X-1" in res.graph.nodes
    assert res.unk_nodes == ["z"]
    assert res.has_unk_nodes
    assert sfiles.graph_to_sfiles(res.graph) == "(raw)(X)(prod)"


def test_controller_inline_with_next_unitop_signal() -> None:
    ir = _ir(
        [IRNode(id="c", source_class="PID", ctrl_fn="FC"), IRNode(id="v", source_class="Valve")],
        [
            IREdge(src=None, dst="c"),
            IREdge(src="c", dst="v"),
            IREdge(src="c", dst="v", kind="signal"),
            IREdge(src="v", dst=None),
        ],
    )
    tm = TypeMap(source="t", classes={"Valve": "v"})
    g = ir_to_graph(ir, tm).graph
    assert "C-1/FC" in g.nodes
    signals = [t["signal"] for t in _tags(g, "C-1/FC", "v-1") if t["signal"]]
    assert signals == [["next_unitop"]]
    s = sfiles.graph_to_sfiles(g)
    assert s == "(raw)(C){FC}_1(v)<_1(prod)"
    assert sfiles.is_isomorphic(g, sfiles.sfiles_to_graph(s))


def test_controller_remote_signal_is_not_next_unitop() -> None:
    ir = _ir(
        [
            IRNode(id="t", source_class="Tank"),
            IRNode(id="c", source_class="PID", ctrl_fn="LC"),
            IRNode(id="v", source_class="Valve"),
        ],
        [
            IREdge(src=None, dst="t"),
            IREdge(src="t", dst="v"),
            IREdge(src="t", dst="c"),
            IREdge(src="c", dst=None),
            IREdge(src="c", dst="v", kind="signal"),
            IREdge(src="v", dst=None),
        ],
    )
    tm = TypeMap(source="t", classes={"Valve": "v", "Tank": "tank"})
    g = ir_to_graph(ir, tm).graph
    assert _tags(g, "C-1/LC", "v-1") == [{"he": [], "col": [], "signal": ["not_next_unitop"]}]
    s = sfiles.graph_to_sfiles(g)
    assert sfiles.is_isomorphic(g, sfiles.sfiles_to_graph(s))


@pytest.mark.parametrize("cls", ["mix", "flash"])
def test_mix_and_flash_roundtrip(cls: str) -> None:
    src = {"mix": "Mixer", "flash": "Vessel"}[cls]
    if cls == "mix":
        edges = [IREdge(src=None, dst="u"), IREdge(src=None, dst="u"), IREdge(src="u", dst=None)]
    else:
        edges = [IREdge(src=None, dst="u"), IREdge(src="u", dst=None), IREdge(src="u", dst=None)]
    g = ir_to_graph(_ir([IRNode(id="u", source_class=src)], edges), TM).graph
    s = sfiles.graph_to_sfiles(g)
    assert sfiles.is_isomorphic(g, sfiles.sfiles_to_graph(s))
