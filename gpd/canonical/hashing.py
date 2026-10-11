"""Dedup keys (BLUEPRINT §3.4).

``canonical_hash`` is the primary key: sha256 of the canonical SFILES string.
``wl_hash`` is the secondary key: a Weisfeiler-Lehman hash over the split-HI
graph with class labels, degrees, and col/signal edge tags. Two records are
duplicates if the canonical hashes match, or the WL hashes match and
``sfiles.is_isomorphic`` confirms.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import defaultdict
from typing import Any

import networkx as nx

from gpd.canonical import vocab
from gpd.canonical.sfiles import split_hi


def canonical_hash(sfiles: str) -> str:
    return hashlib.sha256(sfiles.encode("utf-8")).hexdigest()


def node_type(node_id: str) -> str:
    """``hex-1/2`` -> ``hex``; ``C-3/FC`` -> ``C/FC``; ``dist-1`` -> ``dist``."""
    cls, _, suffix = node_id.partition("/")
    cls = cls.split("-", 1)[0]
    if suffix and not suffix.isnumeric():
        return f"{cls}/{suffix}"
    return cls


def node_class(node_id: str) -> str:
    return node_type(node_id).split("/", 1)[0]


def _edge_label(tags: dict[str, Any] | None) -> str:
    if not tags:
        return ""
    parts = sorted(tags.get("col", [])) + sorted(tags.get("signal", []))
    return "+".join(parts)


def wl_hash(g: nx.MultiDiGraph, *, iterations: int = 4) -> str:
    h = split_hi(g)
    d = nx.DiGraph()
    for n in h.nodes:
        d.add_node(n, label=f"{node_type(n)}|i{h.in_degree(n)}|o{h.out_degree(n)}")
    labels: dict[tuple[str, str], list[str]] = defaultdict(list)
    for u, v, data in h.edges(data=True):
        labels[(u, v)].append(_edge_label(data.get("tags")))
    for (u, v), parts in labels.items():
        d.add_edge(u, v, label="|".join(sorted(parts)))
    with warnings.catch_warnings():
        # networkx >= 3.5 warns once per call that directed WL hashes changed. We pin >= 3.5.
        warnings.simplefilter("ignore", UserWarning)
        out: str = nx.weisfeiler_lehman_graph_hash(
            d, node_attr="label", edge_attr="label", iterations=iterations
        )
    return out


def is_control_node(node_id: str) -> bool:
    return node_class(node_id) == vocab.CONTROL
