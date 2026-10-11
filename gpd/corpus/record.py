"""Corpus record schema (BLUEPRINT §3.5)."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from gpd.canonical.graphio import graph_from_json
from gpd.canonical.ir import Domain, ExtractionMethod, IRFlowsheet, Tier

Split = Literal["train", "val", "test"]


class Provenance(BaseModel):
    """IRFlowsheet minus nodes/edges."""

    model_config = ConfigDict(extra="forbid")

    source: str
    source_uri: str
    source_file: str
    source_sha256: str
    license: str
    tier: Tier
    domain: Domain
    extraction_method: ExtractionMethod
    extraction_confidence: float = Field(ge=0.0, le=1.0)

    @classmethod
    def from_ir(cls, ir: IRFlowsheet) -> Provenance:
        return cls(
            source=ir.source,
            source_uri=ir.source_uri,
            source_file=ir.source_file,
            source_sha256=ir.source_sha256,
            license=ir.license,
            tier=ir.tier,
            domain=ir.domain,
            extraction_method=ir.extraction_method,
            extraction_confidence=ir.extraction_confidence,
        )


class Validity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grammar_ok: bool
    rule_violations: list[str] = Field(default_factory=list)  # rule ids
    rules_ok: bool = True  # no error-severity violations
    sim_converged: bool | None = None  # None = not run
    has_unk_nodes: bool = False


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    sfiles: str  # canonical; includes control notation when present
    sfiles_pfd: str | None = None  # control-stripped variant, only when has_control
    graph: dict[str, Any]  # gpd.canonical.graphio JSON
    canonical_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    wl_hash: str
    n_units: int = Field(ge=0)
    n_streams: int = Field(ge=0)
    has_recycle: bool
    has_control: bool
    provenance: Provenance
    validity: Validity
    is_real: bool
    augmented_from: str | None = None
    duplicates: list[str] = Field(default_factory=list)  # merged record ids
    split: Split | None = None

    @staticmethod
    def make_id(source: str, source_sha256: str, idx: int) -> str:
        return f"{source}:{source_sha256[:12]}:{idx}"

    @staticmethod
    def sha256_of(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def to_graph(self) -> nx.MultiDiGraph:
        return graph_from_json(self.graph)
