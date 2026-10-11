# Extractors

Isolation rule (BLUEPRINT D4): each directory here is its own environment. It
reads raw source files and writes `data/interim/<source>/*.ir.json` conforming
to `gpd.canonical.ir.IRFlowsheet`. Nothing else crosses the boundary.

- The core package `gpd` never imports anything from here. `.importlinter`
  forbids `idaes`, `watertap`, `biosteam`, `pyomo`, `pydexpi`, `win32com`.
- Pure-Python extractors (dwsim, aspen_bkp, pid2graph, modelica, hirtreiter) may
  depend on `gpd` for the IR schema and run in the main venv.
- Heavy or license-isolated extractors (pyomo_*, biosteam, dexpi, coco) ship
  their own `pyproject.toml` or `Dockerfile` and vendor a copy of `ir.py`
  pinned by schema version.

| dir | status | runs in |
|---|---|---|
| dwsim | WS1.1 | main venv |
| aspen_bkp | WS1.2 | main venv |
| pyomo_idaes | WS1.3 | own venv |
| pyomo_watertap | WS1.3 | own venv |
| biosteam | WS1.3 | own venv |
| dexpi | WS1.4 / WS2.3 | Docker (AGPL) |
| pid2graph | WS1.5 | main venv |
| modelica | WS1.6 | main venv |
| coco | WS1.7 | Windows CI lane |
| hirtreiter | WS2.1 | main venv |
