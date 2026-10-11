from __future__ import annotations

import pytest
from pydantic import ValidationError

from gpd.canonical.ir import IREdge, IRFlowsheet, IRNode

SHA = "0" * 64


def _ir(nodes: list[IRNode], edges: list[IREdge]) -> IRFlowsheet:
    return IRFlowsheet(
        source="test",
        source_uri="file://x",
        source_file="x.xml",
        source_sha256=SHA,
        license="MIT",
        tier="A",
        domain="chemical",
        extraction_method="manual",
        nodes=nodes,
        edges=edges,
    )


def test_minimal_valid_ir() -> None:
    ir = _ir(
        [IRNode(id="h", source_class="Heater")],
        [IREdge(src=None, dst="h"), IREdge(src="h", dst=None)],
    )
    assert ir.ir_version == "1"
    assert len(ir.material_edges) == 2


def test_duplicate_node_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate node ids"):
        _ir([IRNode(id="a", source_class="X"), IRNode(id="a", source_class="Y")], [])


def test_edge_to_unknown_node_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown node"):
        _ir([IRNode(id="a", source_class="X")], [IREdge(src="a", dst="ghost")])


def test_edge_needs_an_endpoint() -> None:
    with pytest.raises(ValidationError, match="at least one endpoint"):
        IREdge(src=None, dst=None)


@pytest.mark.parametrize("kind", ["energy", "signal"])
def test_non_material_edges_cannot_cross_boundary(kind: str) -> None:
    with pytest.raises(ValidationError, match="cannot cross"):
        IREdge(src="a", dst=None, kind=kind)  # type: ignore[arg-type]


def test_ctrl_fn_pattern() -> None:
    IRNode(id="c", source_class="PID", ctrl_fn="FC")
    with pytest.raises(ValidationError):
        IRNode(id="c", source_class="PID", ctrl_fn="fc")


def test_sha_pattern_and_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        _ir([], []).model_copy(update={"source_sha256": "abc"}).model_validate(
            {**_ir([], []).model_dump(), "source_sha256": "abc"}
        )
    with pytest.raises(ValidationError):
        IRFlowsheet.model_validate({**_ir([], []).model_dump(), "extraction_confidence": 1.5})


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        IRNode.model_validate({"id": "a", "source_class": "X", "bogus": 1})


def test_json_roundtrip() -> None:
    ir = _ir(
        [IRNode(id="h", source_class="Heater", attrs={"duty_kw": 12.5, "flag": True})],
        [IREdge(src=None, dst="h", name="S1"), IREdge(src="h", dst=None)],
    )
    again = IRFlowsheet.model_validate_json(ir.model_dump_json())
    assert again == ir
