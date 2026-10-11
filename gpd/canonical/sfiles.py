"""Thin, defensive wrapper over the SFILES2 ``Flowsheet`` class (BLUEPRINT §3.3).

Canonical string = ``convert_to_sfiles(version="v2", remove_hex_tags=True,
canonical=True)``. ``remove_hex_tags=True`` is required: SFILES2 cannot
re-parse strings carrying ``hot_in``/``cold_out`` side labels. Heat-integration
pairing ``{n}`` tags survive; only the hot/cold side assignment is lost.

Upstream mutates graphs in place and warns on type conversion. Every entry
point deep-copies its input and silences those warnings.
"""

from __future__ import annotations

import copy
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

import networkx as nx
from Flowsheet_Class.flowsheet import Flowsheet
from Flowsheet_Class.utils_isomorphism import check_isomorphism

# Upstream SFILES_parser regex, verbatim (flowsheet.py). Assumes < 100 cycles.
TOKEN_RE: Final = re.compile(
    r"(\(.+?\)|\{.+?\}|[<%_]+\d+|\]|\[|\<\&\||(?<!<)&\||n\||(?<!&)(?<!n)\||&(?!\|)|\d)"
)


class CanonicalizationError(RuntimeError):
    """SFILES2 failed or produced an empty result. Message carries the cause."""


@contextmanager
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _tag_key(attrs: dict[str, Any]) -> str:
    tags = attrs.get("tags") or {}
    return "|".join(
        f"{k}:{','.join(sorted(map(str, tags.get(k, []))))}" for k in ("col", "he", "signal")
    )


def _normalize(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Rebuild ``g`` with deterministic insertion order.

    SFILES2 breaks ties by networkx iteration order. Parallel edges between the
    same pair (e.g. a flash whose top and bottom both feed one mixer) would
    otherwise flip which one is written as the back-edge. Sorting parallel
    edges by their tag label removes the dependence on construction history.
    """
    h = nx.MultiDiGraph()
    h.graph.update(g.graph)
    for n in sorted(g.nodes):
        h.add_node(n, **g.nodes[n])
    edges = sorted(
        ((u, v, copy.deepcopy(d)) for u, v, d in g.edges(data=True)),
        key=lambda e: (e[0], e[1], _tag_key(e[2])),
    )
    for u, v, d in edges:
        h.add_edge(u, v, **d)
    return h


def _flowsheet_from_graph(g: nx.MultiDiGraph, *, ontocape_names: bool = False) -> Any:
    fs = Flowsheet(OntoCapeConformity=ontocape_names)
    with _quiet():
        fs.create_from_nx(_normalize(g))
    return fs


def graph_to_sfiles(g: nx.MultiDiGraph, *, ontocape_names: bool = False) -> str:
    """Canonical SFILES 2.0 string for a graph in SFILES2 node conventions.

    Set ``ontocape_names=True`` when nodes are named ``HeatExchanger-1`` etc.
    """
    try:
        fs = _flowsheet_from_graph(g, ontocape_names=ontocape_names)
        with _quiet():
            fs.convert_to_sfiles(version="v2", remove_hex_tags=True, canonical=True)
    except Exception as exc:  # upstream raises bare AssertionError/KeyError/IndexError
        raise CanonicalizationError(f"graph -> SFILES failed: {exc!r}") from exc
    s = fs.sfiles
    if not isinstance(s, str) or not s:
        raise CanonicalizationError("graph -> SFILES produced an empty string")
    return s


def sfiles_to_graph(s: str) -> nx.MultiDiGraph:
    """Parse an SFILES 2.0 string. Heat-integrated exchangers come back merged."""
    if not s or not s.strip():
        raise CanonicalizationError("empty SFILES string")
    try:
        fs = Flowsheet()
        with _quiet():
            fs.create_from_sfiles(s)
    except Exception as exc:
        raise CanonicalizationError(f"SFILES -> graph failed for {s!r}: {exc!r}") from exc
    g = fs.state
    if g.number_of_nodes() == 0:
        raise CanonicalizationError(f"SFILES -> graph produced no nodes for {s!r}")
    return g


def canonical_sfiles(s: str) -> str:
    return graph_to_sfiles(sfiles_to_graph(s))


def canonicalize(
    g: nx.MultiDiGraph, *, ontocape_names: bool = False
) -> tuple[str, nx.MultiDiGraph]:
    """Return (canonical string, normalized graph). The graph is the parse of the string."""
    s = graph_to_sfiles(g, ontocape_names=ontocape_names)
    return s, sfiles_to_graph(s)


def split_hi(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return a copy with multi-pass exchangers split into ``hex-N/k`` nodes."""
    try:
        fs = _flowsheet_from_graph(g)
        with _quiet():
            fs.split_HI_nodes()
    except Exception as exc:
        raise CanonicalizationError(f"split_HI_nodes failed: {exc!r}") from exc
    return fs.state


def is_isomorphic(a: nx.MultiDiGraph, b: nx.MultiDiGraph) -> bool:
    """SFILES2 isomorphism: same unit classes (+ controller function), same col/signal tags."""
    fa = _flowsheet_from_graph(a)
    fb = _flowsheet_from_graph(b)
    with _quiet():
        return bool(check_isomorphism(fa, fb, check_HI_nodes=False))


def tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s)
