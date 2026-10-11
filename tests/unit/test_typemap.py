from __future__ import annotations

import pytest
from pydantic import ValidationError

from gpd.canonical import vocab
from gpd.canonical.typemap import TypeMap, available_typemaps, load_typemap


def test_every_shipped_typemap_loads_and_targets_vocab() -> None:
    names = available_typemaps()
    assert "dwsim" in names
    for name in names:
        tm = load_typemap(name)
        assert tm.source == name
        assert set(tm.classes.values()) <= vocab.UNIT_CLASSES


def test_missing_typemap() -> None:
    with pytest.raises(FileNotFoundError):
        load_typemap("no_such_source")


def test_full_then_last_segment_then_default() -> None:
    tm = TypeMap(source="t", classes={"ns.Exact": "pp", "Heater": "hex"})
    assert tm.map_class("ns.Exact") == ("pp", True)
    assert tm.map_class("DWSIM.UnitOperations.UnitOperations.Heater") == ("hex", True)
    assert tm.map_class("Heater") == ("hex", True)
    assert tm.map_class("ns.Unknown") == (vocab.UNK, False)


def test_rejects_unknown_targets() -> None:
    with pytest.raises(ValidationError, match="not in vocab"):
        TypeMap(source="t", classes={"A": "turbine"})
    with pytest.raises(ValidationError, match="port tags"):
        TypeMap(source="t", ports={"P": "sideways"})
    with pytest.raises(ValidationError, match="default"):
        TypeMap(source="t", default="nope")


def test_map_port() -> None:
    tm = TypeMap(source="t", ports={"Distillate": "tout"})
    assert tm.map_port("Distillate") == "tout"
    assert tm.map_port("Other") is None
    assert tm.map_port(None) is None
