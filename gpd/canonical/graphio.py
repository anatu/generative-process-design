"""Graph <-> JSON and summary statistics (BLUEPRINT §3.5).

Own JSON shape instead of ``nx.node_link_data`` so networkx key renames never
touch stored records.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from pydantic import BaseModel

from gpd.canonical.hashing import is_control_node

GRAPH_JSON_VERSION = 1


def graph_to_json(g: nx.MultiDiGraph) -> dict[str, Any]:
    return {
        "v": GRAPH_JSON_VERSION,
        "nodes": [{"id": n, **attrs} for n, attrs in g.nodes(data=True)],
        "edges": [
            {"source": u, "target": v, "key": k, **attrs}
            for u, v, k, attrs in g.edges(keys=True, data=True)
        ],
    }


def graph_from_json(d: dict[str, Any]) -> nx.MultiDiGraph:
    if d.get("v") != GRAPH_JSON_VERSION:
        raise ValueError(f"unsupported graph json version {d.get('v')!r}")
    g = nx.MultiDiGraph()
    for node in d["nodes"]:
        attrs = dict(node)
        g.add_node(attrs.pop("id"), **attrs)
    for edge in d["edges"]:
        attrs = dict(edge)
        g.add_edge(attrs.pop("source"), attrs.pop("target"), key=attrs.pop("key"), **attrs)
    return g


class GraphStats(BaseModel):
    n_units: int  # nodes excluding raw/prod/controllers
    n_streams: int  # material edges
    n_controllers: int
    has_recycle: bool
    has_control: bool


def _is_material(attrs: dict[str, Any]) -> bool:
    tags = attrs.get("tags") or {}
    return not tags.get("signal")


def graph_stats(g: nx.MultiDiGraph) -> GraphStats:
    controllers = [n for n in g.nodes if is_control_node(n)]
    terminals = {n for n in g.nodes if n.split("-", 1)[0] in ("raw", "prod")}
    material = nx.DiGraph()
    material.add_nodes_from(g.nodes)
    for u, v, attrs in g.edges(data=True):
        if _is_material(attrs):
            material.add_edge(u, v)
    return GraphStats(
        n_units=g.number_of_nodes() - len(terminals) - len(controllers),
        n_streams=material.number_of_edges(),
        n_controllers=len(controllers),
        has_recycle=not nx.is_directed_acyclic_graph(material),
        has_control=bool(controllers),
    )
