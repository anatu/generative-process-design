"""Frozen SFILES 2.0 vocabulary (BLUEPRINT §3.2, §WS4.1 G6).

The class table is copied from the pinned SFILES2 submodule
(``Flowsheet_Class/OntoCape_SFILES_mapping.py``) so the core never tracks
upstream drift silently. ``tests/unit/test_vocab.py`` fails if this file and
upstream disagree, or if the submodule moves off the pinned commit.

Degree constraints are WS4 defaults. Calibrate against the real corpus at M2.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

SFILES2_PINNED_COMMIT: Final = "fe0aa9c36cbe844ebbf32034b5b7be4cf0ca6ab3"
SFILES2_PINNED_VERSION: Final = "v1.2.0"

# OntoCape class -> SFILES abbreviation. Verbatim from upstream.
ONTOCAPE_TO_SFILES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "RawMaterial": "raw",
        "OutputProduct": "prod",
        "HeatExchanger": "hex",
        "Reboiler": "reb",
        "Condenser": "cond",
        "Blower": "blwr",
        "Compressor": "comp",
        "Expander": "turb",
        "Pipe": "pipe",
        "Pump": "pp",
        "OrificePlate": "orif",
        "ChemicalReactor": "r",
        "SplittingUnit": "splt",
        "MixingUnit": "mix",
        "StorageUnit": "tank",
        "SeparationUnit": "sep",
        "DistillationSystem": "dist",
        "FlashUnit": "flash",
        "ExtractionUnit": "extr",
        "RectificationSystem": "rect",
        "StrippingSystem": "strip",
        "AbsorptionColumn": "abs",
        "LiquidFilter": "lfil",
        "GasFilter": "gfil",
        "Scrubber": "scrub",
        "CentrifugationUnit": "centr",
        "Cyclone": "cycl",
        "Hydrocyclone": "hcycl",
        "ElectricalGasCleaningUnit": "egclean",
        "Valve": "v",
        "ProcessUnit": "X",
        "Control": "C",
        "X": "X",
        "InputOutput": "X",
        "ms": "X",
    }
)

RAW: Final = "raw"
PROD: Final = "prod"
CONTROL: Final = "C"
# SFILES2's generic ProcessUnit. BLUEPRINT's "unk" (unmapped source class) renders as this.
UNK: Final = "X"

UNIT_CLASSES: Final[frozenset[str]] = frozenset(ONTOCAPE_TO_SFILES.values())
TERMINAL_CLASSES: Final[frozenset[str]] = frozenset({RAW, PROD})
# Classes whose nodes carry material streams (everything except controllers).
MATERIAL_CLASSES: Final[frozenset[str]] = UNIT_CLASSES - {CONTROL}

# Edge tag vocabulary. Regex is upstream's (flowsheet.py, create_from_sfiles).
COLUMN_TAGS: Final[frozenset[str]] = frozenset({"tout", "tin", "bout", "bin"})
SIGNAL_TAGS: Final[frozenset[str]] = frozenset({"next_unitop", "not_next_unitop"})
HE_TAG_RE: Final = re.compile(r"(hot.*|cold.*|[0-9].*|Null)")

# Node ids follow "<class>-<n>" with an optional "/<pass>" for split heat exchangers.
NODE_ID_RE: Final = re.compile(r"^(?P<cls>[A-Za-z]+)-(?P<num>\d+)(?:/(?P<pass_>\d+))?$")


@dataclass(frozen=True)
class NodeId:
    cls: str
    num: int
    pass_: int | None = None

    def __str__(self) -> str:
        base = f"{self.cls}-{self.num}"
        return base if self.pass_ is None else f"{base}/{self.pass_}"


def parse_node_id(node_id: str) -> NodeId:
    """Split an SFILES2 node id. Raises ValueError on malformed ids.

    Does not check class membership; use ``is_known_class`` for that.
    """
    m = NODE_ID_RE.match(node_id)
    if m is None:
        raise ValueError(f"malformed SFILES2 node id: {node_id!r}")
    pass_ = m.group("pass_")
    return NodeId(m.group("cls"), int(m.group("num")), int(pass_) if pass_ else None)


def node_class(node_id: str) -> str:
    return parse_node_id(node_id).cls


def is_known_class(cls: str) -> bool:
    return cls in UNIT_CLASSES


@dataclass(frozen=True)
class DegreeSpec:
    """Material in/out degree bounds for one class. ``None`` = unbounded."""

    min_in: int
    max_in: int | None
    min_out: int
    max_out: int | None

    def check(self, n_in: int, n_out: int) -> bool:
        if n_in < self.min_in or (self.max_in is not None and n_in > self.max_in):
            return False
        return not (n_out < self.min_out or (self.max_out is not None and n_out > self.max_out))


_ONE_IN_ONE_OUT = DegreeSpec(1, 1, 1, 1)
_SEPARATOR = DegreeSpec(1, None, 2, None)  # >=1 feed, >=2 outlets
_FREE = DegreeSpec(0, None, 0, None)

# Material-edge degree constraints per class (G6). Controllers carry only signal edges.
DEGREE_CONSTRAINTS: Final[Mapping[str, DegreeSpec]] = MappingProxyType(
    {
        "raw": DegreeSpec(0, 0, 1, 1),
        "prod": DegreeSpec(1, 1, 0, 0),
        # Heat exchangers: a split pass node is 1/1; a merged node is 2/2.
        "hex": DegreeSpec(1, 2, 1, 2),
        "reb": DegreeSpec(1, 2, 1, 2),
        "cond": DegreeSpec(1, 2, 1, 2),
        "blwr": _ONE_IN_ONE_OUT,
        "comp": _ONE_IN_ONE_OUT,
        "turb": _ONE_IN_ONE_OUT,
        "pipe": _ONE_IN_ONE_OUT,
        "pp": _ONE_IN_ONE_OUT,
        "orif": _ONE_IN_ONE_OUT,
        "v": _ONE_IN_ONE_OUT,
        "r": DegreeSpec(1, None, 1, None),
        "splt": DegreeSpec(1, 1, 2, None),
        "mix": DegreeSpec(2, None, 1, 1),
        "tank": DegreeSpec(1, None, 1, None),
        "sep": _SEPARATOR,
        "dist": _SEPARATOR,
        "flash": _SEPARATOR,
        "extr": _SEPARATOR,
        "rect": _SEPARATOR,
        "strip": _SEPARATOR,
        "abs": _SEPARATOR,
        "lfil": _SEPARATOR,
        "gfil": _SEPARATOR,
        "scrub": _SEPARATOR,
        "centr": _SEPARATOR,
        "cycl": _SEPARATOR,
        "hcycl": _SEPARATOR,
        "egclean": _SEPARATOR,
        "X": _FREE,
        "C": DegreeSpec(0, 0, 0, 0),
    }
)
