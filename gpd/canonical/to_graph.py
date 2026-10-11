"""IR -> networkx MultiDiGraph in SFILES2 conventions (BLUEPRINT §3.2).

Conventions produced here and consumed by ``sfiles.py``:
- node id ``<class>-<n>`` numbered per class in IR order; controllers ``C-<n>/<FN>``;
- every edge carries ``tags={"he": [...], "col": [...], "signal": [...]}``;
- boundary material streams become ``raw-<n>`` / ``prod-<n>`` nodes;
- energy edges are dropped and counted;
- a signal edge is tagged ``next_unitop`` when a material edge joins the same pair.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from gpd.canonical import vocab
from gpd.canonical.ir import IREdge, IRFlowsheet
from gpd.canonical.typemap import TypeMap

# SFILES2's merge/split logic wants numeric sides on multi-pass exchangers.
_HE_SIDE_TO_NUM = {"hot": "1", "cold": "2"}


def empty_tags() -> dict[str, list[str]]:
    return {"he": [], "col": [], "signal": []}


@dataclass
class ConversionResult:
    graph: nx.MultiDiGraph
    unk_nodes: list[str] = field(default_factory=list)  # IR node ids mapped to X
    dropped_energy_edges: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def has_unk_nodes(self) -> bool:
        return bool(self.unk_nodes)


def _he_tag(port_tag: str, direction: str) -> str:
    side = port_tag.split("_", 1)[0]
    return f"{_HE_SIDE_TO_NUM[side]}_{direction}"


def _port_tags(typemap: TypeMap, edge: IREdge) -> dict[str, list[str]]:
    """Translate IR port names on one edge into SFILES2 edge tags."""
    tags = empty_tags()
    for port, direction in ((edge.src_port, "out"), (edge.dst_port, "in")):
        tag = typemap.map_port(port)
        if tag is None:
            continue
        if tag in vocab.COLUMN_TAGS:
            tags["col"].append(tag)
        else:
            tags["he"].append(_he_tag(tag, direction))
    return tags


def ir_to_graph(ir: IRFlowsheet, typemap: TypeMap) -> ConversionResult:
    g = nx.MultiDiGraph()
    result = ConversionResult(graph=g)
    counters: Counter[str] = Counter()

    def new_name(cls: str, suffix: str | None = None) -> str:
        counters[cls] += 1
        base = f"{cls}-{counters[cls]}"
        return base if suffix is None else f"{base}/{suffix}"

    id_map: dict[str, str] = {}
    for node in ir.nodes:
        attrs: dict[str, Any] = {"src_id": node.id, "src_class": node.source_class}
        if node.name:
            attrs["label"] = node.name
        if node.ctrl_fn:
            name = new_name(vocab.CONTROL, node.ctrl_fn)
        else:
            cls, known = typemap.map_class(node.source_class)
            if not known:
                result.unk_nodes.append(node.id)
                result.notes.append(f"unmapped class {node.source_class!r} for node {node.id!r}")
            name = new_name(cls)
        id_map[node.id] = name
        g.add_node(name, **attrs)

    def endpoint(ir_id: str | None, boundary_cls: str) -> str:
        if ir_id is not None:
            return id_map[ir_id]
        name = new_name(boundary_cls)
        g.add_node(name, src_id=None, src_class=boundary_cls)
        return name

    for edge in ir.edges:
        if edge.kind == "energy":
            result.dropped_energy_edges += 1
            continue
        if edge.kind == "signal":
            continue  # second pass: needs the material edges in place
        u = endpoint(edge.src, vocab.RAW)
        v = endpoint(edge.dst, vocab.PROD)
        attrs = {"tags": _port_tags(typemap, edge)}
        if edge.name:
            attrs["label"] = edge.name
        g.add_edge(u, v, **attrs)

    for edge in ir.edges:
        if edge.kind != "signal":
            continue
        assert edge.src is not None and edge.dst is not None  # enforced by IREdge
        u, v = id_map[edge.src], id_map[edge.dst]
        has_material = any(
            not d["tags"]["signal"] for d in g.get_edge_data(u, v, default={}).values()
        )
        tags = empty_tags()
        tags["signal"].append("next_unitop" if has_material else "not_next_unitop")
        g.add_edge(u, v, tags=tags)

    if result.dropped_energy_edges:
        result.notes.append(f"dropped {result.dropped_energy_edges} energy edge(s)")
    return result
