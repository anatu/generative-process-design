"""Render SFILES 2.0 strings as flowsheet drawings (matplotlib; no graphviz needed).

    uv run python -m gpd.viz --pfd "(raw)(hex)(dist)[{tout}(prod)]{bout}(prod)" \
        --pid "(raw)(hex)(C){TC}_1(dist)[(C){LC}_2][{tout}(C){PC}_3(v)<_3(prod)]{bout}(v)<_2(prod)" \
        --out docs/img/example.png

Layout: longest-path layering over material edges (recycles cut), feeds on the
left, products on the right. Controllers sit above their measured unit.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import typer

from gpd.canonical import sfiles

UNIT_FILL, UNIT_EDGE = "#dbe9ff", "#2b5aa8"
CTRL_FILL, CTRL_EDGE = "#ffe7b3", "#b06e00"
TERM_FILL = "#eeeeee"


def _is_signal(d: dict[str, Any]) -> bool:
    return bool((d.get("tags") or {}).get("signal"))


def _layered_positions(g: nx.MultiDiGraph) -> dict[str, tuple[float, float]]:
    ctrl = {n for n in g.nodes if n.startswith("C-")}
    # Material graph with controllers contracted away. Controllers sit inline on
    # material paths, often in chains (pump -> PI -> FC -> FC -> valve); contracting
    # one at a time handles chains of any length.
    mat = nx.DiGraph()
    mat.add_nodes_from(g.nodes)
    for u, v, d in g.edges(data=True):
        if not _is_signal(d) and u != v:
            mat.add_edge(u, v)
    anchor: dict[str, str | None] = {}
    for c in sorted(ctrl):
        preds = list(mat.predecessors(c))
        succs = list(mat.successors(c))
        # Nearest non-controller material predecessor (through already-contracted chain).
        anchor[c] = next((p for p in preds if p not in ctrl), None) or next(
            (anchor[p] for p in preds if p in ctrl and anchor.get(p)), None
        )
        if anchor[c] is None:
            anchor[c] = next((s for s in succs if s not in ctrl), None)
        for u in preds:
            for v in succs:
                if u != v:
                    mat.add_edge(u, v)
        mat.remove_node(c)
    dag = mat.copy()
    while True:
        try:
            cycle = nx.find_cycle(dag)
        except nx.NetworkXNoCycle:
            break
        dag.remove_edge(*cycle[-1][:2])
    layer: dict[str, int] = {}
    for n in nx.topological_sort(dag):
        preds = list(dag.predecessors(n))
        layer[n] = 0 if not preds else 1 + max(layer[p] for p in preds)
    rows: dict[int, list[str]] = defaultdict(list)
    for n, lv in layer.items():
        rows[lv].append(n)
    pos: dict[str, tuple[float, float]] = {}
    for x, names in rows.items():
        names.sort()
        for i, n in enumerate(names):
            pos[n] = (x * 2.0, (i - (len(names) - 1) / 2) * 1.5)
    # Controllers: above the unit they measure, fanned out when several share one.
    placed: dict[str, int] = defaultdict(int)
    for c in sorted(ctrl):
        a = anchor.get(c)
        ax, ay = pos.get(a, (0.0, 0.0)) if a else (0.0, 0.0)
        k = placed[a or ""]
        placed[a or ""] += 1
        pos[c] = (ax + 0.9 + 0.6 * k, ay + 1.1 + 0.5 * k)
    return pos


def draw(g: nx.MultiDiGraph, ax: Any, title: str) -> None:
    pos = _layered_positions(g)
    ctrl = [n for n in g.nodes if n.startswith("C-")]
    terms = [n for n in g.nodes if n.split("-")[0] in ("raw", "prod")]
    units = [n for n in g.nodes if n not in ctrl and n not in terms]
    mat = [(u, v) for u, v, d in g.edges(data=True) if not _is_signal(d)]
    sig = [(u, v) for u, v, d in g.edges(data=True) if _is_signal(d)]
    nx.draw_networkx_nodes(
        g, pos, nodelist=terms, node_color=TERM_FILL, node_size=1300, edgecolors="#888", ax=ax
    )
    nx.draw_networkx_nodes(
        g, pos, nodelist=units, node_color=UNIT_FILL, node_size=1500, edgecolors=UNIT_EDGE, ax=ax
    )
    nx.draw_networkx_nodes(
        g, pos, nodelist=ctrl, node_color=CTRL_FILL, node_size=1200, edgecolors=CTRL_EDGE, ax=ax
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=mat,
        arrows=True,
        arrowsize=16,
        width=1.7,
        edge_color=UNIT_EDGE,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=sig,
        arrows=True,
        arrowsize=13,
        width=1.3,
        edge_color=CTRL_EDGE,
        style="dashed",
        connectionstyle="arc3,rad=0.3",
        ax=ax,
    )
    labels = {n: (n.split("/")[1] if n in ctrl and "/" in n else n) for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels, font_size=8, ax=ax)
    etags = {
        (u, v): ",".join(d["tags"]["col"]) for u, v, d in g.edges(data=True) if d["tags"].get("col")
    }
    nx.draw_networkx_edge_labels(g, pos, etags, font_size=7, font_color="#444", ax=ax)
    ax.set_title(title, fontsize=11, loc="left")
    ax.axis("off")


def render_pair(pfd: str, pid: str | None, out: Path, *, pid_title: str = "P&ID") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graphs = [("PFD (input)", sfiles.split_hi(sfiles.sfiles_to_graph(pfd)))]
    if pid:
        graphs.append((pid_title, sfiles.split_hi(sfiles.sfiles_to_graph(pid))))
    width = max(8, 1.6 * max(g.number_of_nodes() for _, g in graphs) ** 0.9)
    fig, axes = plt.subplots(len(graphs), 1, figsize=(min(width, 22), 4.2 * len(graphs)))
    axes = [axes] if len(graphs) == 1 else list(axes)
    for ax, (title, g) in zip(axes, graphs, strict=True):
        draw(g, ax, title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    pfd: str = typer.Option(...),
    pid: str | None = typer.Option(None),
    out: Path = typer.Option(Path("flowsheet.png")),
    pid_title: str = "P&ID",
) -> None:
    typer.echo(str(render_pair(pfd, pid, out, pid_title=pid_title)))


if __name__ == "__main__":
    sys.exit(app())
