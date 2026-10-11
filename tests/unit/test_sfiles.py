from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from gpd.canonical import sfiles
from gpd.canonical.to_graph import empty_tags

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sfiles"


def _fixture_strings() -> list[pytest.param]:
    return [
        pytest.param(p.read_text().strip(), id=p.stem) for p in sorted(FIXTURES.glob("*.sfiles"))
    ]


@pytest.mark.parametrize("s", _fixture_strings())
def test_fixtures_are_canonical(s: str) -> None:
    assert sfiles.canonical_sfiles(s) == s


@pytest.mark.parametrize("s", _fixture_strings())
def test_fixture_graphs_roundtrip_isomorphic(s: str) -> None:
    g = sfiles.sfiles_to_graph(s)
    assert sfiles.is_isomorphic(g, sfiles.sfiles_to_graph(sfiles.graph_to_sfiles(g)))


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_string_rejected(bad: str) -> None:
    with pytest.raises(sfiles.CanonicalizationError, match="empty"):
        sfiles.sfiles_to_graph(bad)


@pytest.mark.parametrize("bad", ["xyz", "(((", "{tout}"])
def test_tokenless_string_raises(bad: str) -> None:
    with pytest.raises(sfiles.CanonicalizationError, match="no nodes"):
        sfiles.sfiles_to_graph(bad)


@pytest.mark.parametrize(
    "lenient", ["(raw)[(hex)", "(raw)(hex)]", "(raw)<1(hex)", "[{tout}(prod)]"]
)
def test_upstream_parses_malformed_strings_leniently(lenient: str) -> None:
    """SFILES2 does not validate grammar: unbalanced brackets and dangling recycle
    markers parse to a smaller graph. Canonical idempotence (WS4.1 G2) is what
    rejects them, so it must stay in the validity layer."""
    sfiles.sfiles_to_graph(lenient)  # does not raise
    assert sfiles.canonical_sfiles(lenient) != lenient


def test_empty_graph_raises() -> None:
    with pytest.raises(sfiles.CanonicalizationError):
        sfiles.graph_to_sfiles(nx.MultiDiGraph())


def test_input_graph_is_not_mutated() -> None:
    g = nx.MultiDiGraph()
    g.add_edge("raw-1", "hex-1", tags=empty_tags())
    g.add_edge("hex-1", "prod-1", tags=empty_tags())
    before = (list(g.nodes), list(g.edges(keys=True, data=True)))
    sfiles.graph_to_sfiles(g)
    sfiles.split_hi(g)
    sfiles.is_isomorphic(g, g)
    assert (list(g.nodes), list(g.edges(keys=True, data=True))) == before


def test_ontocape_named_graph() -> None:
    g = nx.MultiDiGraph()
    g.add_edge("RawMaterial-1", "HeatExchanger-1", tags=empty_tags())
    g.add_edge("HeatExchanger-1", "OutputProduct-1", tags=empty_tags())
    assert sfiles.graph_to_sfiles(g, ontocape_names=True) == "(raw)(hex)(prod)"


def test_split_hi_splits_two_pass_exchanger() -> None:
    s = "(raw)(hex){1}(prod)n|(raw)(hex){1}(prod)"
    merged = sfiles.sfiles_to_graph(s)
    assert "hex-1" in merged.nodes and merged.in_degree("hex-1") == 2
    split = sfiles.split_hi(merged)
    assert {"hex-1/1", "hex-1/2"} <= set(split.nodes)
    assert sfiles.graph_to_sfiles(merged) == s


def test_tokenize_matches_upstream_cases() -> None:
    assert sfiles.tokenize("(raw)(hex){1}(dist)[{tout}(prod)]{bout}(prod)n|(raw)(C){FC}_1(v)<_1(prod)") == [
        "(raw)", "(hex)", "{1}", "(dist)", "[", "{tout}", "(prod)", "]", "{bout}", "(prod)", "n|",
        "(raw)", "(C)", "{FC}", "_1", "(v)", "<_1", "(prod)",
    ]  # fmt: skip
    assert sfiles.tokenize("(pp)<1(splt)[(hex)<&|(raw)&|(prod)]1") == [
        "(pp)", "<1", "(splt)", "[", "(hex)", "<&|", "(raw)", "&|", "(prod)", "]", "1",
    ]  # fmt: skip
