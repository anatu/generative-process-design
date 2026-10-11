"""Source class -> SFILES class tables (BLUEPRINT §3.2).

One YAML per source under ``gpd/canonical/typemap/``. Class lookup tries the
full source class string first, then its last dotted segment, so a namespace
change upstream does not silently turn every unit into ``X``.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gpd.canonical import vocab

# Canonical port tags an extractor may emit. Column tags land in ``tags.col``;
# heat-exchanger side tags land in ``tags.he`` after numeric conversion.
PORT_TAGS: Final[frozenset[str]] = vocab.COLUMN_TAGS | {
    "hot_in",
    "hot_out",
    "cold_in",
    "cold_out",
}


class TypeMap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    default: str = vocab.UNK
    classes: dict[str, str] = Field(default_factory=dict)
    ports: dict[str, str] = Field(default_factory=dict)

    @field_validator("default")
    @classmethod
    def _default_known(cls, v: str) -> str:
        if not vocab.is_known_class(v):
            raise ValueError(f"default class {v!r} not in vocab")
        return v

    @field_validator("classes")
    @classmethod
    def _classes_known(cls, v: dict[str, str]) -> dict[str, str]:
        bad = sorted({c for c in v.values() if not vocab.is_known_class(c)})
        if bad:
            raise ValueError(f"typemap values not in vocab: {bad}")
        return v

    @field_validator("ports")
    @classmethod
    def _ports_known(cls, v: dict[str, str]) -> dict[str, str]:
        bad = sorted({p for p in v.values() if p not in PORT_TAGS})
        if bad:
            raise ValueError(f"port tags not allowed: {bad}")
        return v

    def map_class(self, source_class: str) -> tuple[str, bool]:
        """Return (sfiles_class, known). Unknown classes map to ``default``."""
        hit = self.classes.get(source_class)
        if hit is None:
            hit = self.classes.get(source_class.rsplit(".", 1)[-1])
        if hit is None:
            return self.default, False
        return hit, True

    def map_port(self, port: str | None) -> str | None:
        if port is None:
            return None
        return self.ports.get(port)


def typemap_dir() -> Path:
    return Path(str(resources.files("gpd.canonical").joinpath("typemap")))


def load_typemap(source: str) -> TypeMap:
    path = typemap_dir() / f"{source}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no typemap for source {source!r} at {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"typemap {path} is not a mapping")
    return TypeMap.model_validate(raw)


def available_typemaps() -> list[str]:
    return sorted(p.stem for p in typemap_dir().glob("*.yaml"))
