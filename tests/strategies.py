"""Hypothesis strategies for valid flowsheet graphs (BLUEPRINT §5.4).

``flowsheet_graphs`` grows a plant from open outlets so every graph satisfies
the G6 degree table in ``vocab.DEGREE_CONSTRAINTS`` by construction. Heat
integration (two-pass exchangers) and controllers are not generated here yet.
"""

from __future__ import annotations

import random
from collections import Counter

import networkx as nx
from hypothesis import strategies as st

from gpd.canonical.to_graph import empty_tags

# Classes the growth strategy knows how to wire. Outlets: 1 by default.
ONE_IN_ONE_OUT = ("hex", "pp", "comp", "v", "r", "tank")
TWO_OUT_TAGGED = ("flash", "dist")  # outlets tagged tout / bout
SPLITTER = "splt"
MIXER = "mix"
RECYCLE_TARGETS = ("r", "tank")  # accept >= 1 inlet per DEGREE_CONSTRAINTS

GROWTH_CLASSES = (*ONE_IN_ONE_OUT, *TWO_OUT_TAGGED, SPLITTER, MIXER)


def _edge(g: nx.MultiDiGraph, u: str, v: str, col: str | None = None) -> None:
    tags = empty_tags()
    if col:
        tags["col"].append(col)
    g.add_edge(u, v, tags=tags)


@st.composite
def flowsheet_graphs(
    draw: st.DrawFn, *, min_units: int = 1, max_units: int = 8, allow_recycle: bool = True
) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    counters: Counter[str] = Counter()

    def new(cls: str) -> str:
        counters[cls] += 1
        name = f"{cls}-{counters[cls]}"
        g.add_node(name)
        return name

    open_outlets: list[tuple[str, str | None]] = [(new("raw"), None)]
    n_units = draw(st.integers(min_units, max_units))

    for _ in range(n_units):
        cls = draw(st.sampled_from(GROWTH_CLASSES))
        if cls == MIXER:
            if len(open_outlets) < 2:
                open_outlets.append((new("raw"), None))
            u1, t1 = open_outlets.pop(draw(st.integers(0, len(open_outlets) - 1)))
            u2, t2 = open_outlets.pop(draw(st.integers(0, len(open_outlets) - 1)))
            m = new(MIXER)
            _edge(g, u1, m, t1)
            _edge(g, u2, m, t2)
            open_outlets.append((m, None))
            continue

        u, t = open_outlets.pop(draw(st.integers(0, len(open_outlets) - 1)))
        n = new(cls)
        _edge(g, u, n, t)
        if cls in TWO_OUT_TAGGED:
            open_outlets += [(n, "tout"), (n, "bout")]
        elif cls == SPLITTER:
            open_outlets += [(n, None), (n, None)]
        else:
            open_outlets.append((n, None))

    if allow_recycle and len(open_outlets) > 1 and draw(st.booleans()):
        targets = [n for n in g.nodes if n.split("-", 1)[0] in RECYCLE_TARGETS]
        if targets:
            i = draw(st.integers(0, len(open_outlets) - 1))
            u, t = open_outlets[i]
            v = draw(st.sampled_from(targets))
            if u != v and not g.has_edge(u, v):
                open_outlets.pop(i)
                _edge(g, u, v, t)

    for u, t in open_outlets:
        _edge(g, u, new("prod"), t)
    return g


def relabel_and_shuffle(g: nx.MultiDiGraph, seed: int) -> nx.MultiDiGraph:
    """Same flowsheet, different node numbering and edge insertion order."""
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = {}
    for n in g.nodes:
        by_class.setdefault(n.split("-", 1)[0], []).append(n)
    mapping: dict[str, str] = {}
    for cls, names in by_class.items():
        nums = list(range(1, len(names) + 1))
        rng.shuffle(nums)
        for name, num in zip(names, nums, strict=True):
            mapping[name] = f"{cls}-{num}"
    edges = [(mapping[u], mapping[v], dict(d)) for u, v, d in g.edges(data=True)]
    rng.shuffle(edges)
    nodes = [mapping[n] for n in g.nodes]
    rng.shuffle(nodes)
    h = nx.MultiDiGraph()
    h.add_nodes_from(nodes)
    for u, v, d in edges:
        h.add_edge(u, v, **d)
    return h
