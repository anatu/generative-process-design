"""Intermediate Representation emitted by every extractor (BLUEPRINT §3.1).

Streams are edges. A plant-boundary stream (feed or product) has ``src=None``
or ``dst=None``; canonicalization turns those into ``raw``/``prod`` nodes.
Energy edges survive in the IR and are dropped at canonicalization.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IR_VERSION: Literal["1"] = "1"

EdgeKind = Literal["material", "energy", "signal"]
Tier = Literal["A", "B", "C", "D", "E"]
Domain = Literal["chemical", "power", "water", "bio", "adjacent"]
ExtractionMethod = Literal[
    "parser", "runtime", "com", "graphml", "generator", "llm", "digitizer", "manual"
]

AttrValue = str | float | int | bool


class IRNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    source_class: str = Field(min_length=1)
    name: str | None = None
    # ISA function letters for controllers (e.g. "FC", "LC", "TIR"). Forces class "C".
    ctrl_fn: str | None = Field(default=None, pattern=r"^[A-Z]{1,5}$")
    attrs: dict[str, AttrValue] = Field(default_factory=dict)


class IREdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: str | None
    dst: str | None
    kind: EdgeKind = "material"
    src_port: str | None = None
    dst_port: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _at_least_one_endpoint(self) -> IREdge:
        if self.src is None and self.dst is None:
            raise ValueError("edge needs at least one endpoint inside the plant")
        if self.kind != "material" and (self.src is None or self.dst is None):
            raise ValueError(f"{self.kind} edges cannot cross the plant boundary")
        return self


class IRFlowsheet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ir_version: Literal["1"] = IR_VERSION
    source: str = Field(min_length=1)
    source_uri: str
    source_file: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str = Field(min_length=1)  # SPDX id, or "unknown" | "pending"
    tier: Tier
    domain: Domain
    extraction_method: ExtractionMethod
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    nodes: list[IRNode]
    edges: list[IREdge]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_references(self) -> IRFlowsheet:
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate node ids: {dupes}")
        known = set(ids)
        for e in self.edges:
            for end in (e.src, e.dst):
                if end is not None and end not in known:
                    raise ValueError(f"edge references unknown node {end!r}")
        return self

    @property
    def material_edges(self) -> list[IREdge]:
        return [e for e in self.edges if e.kind == "material"]
