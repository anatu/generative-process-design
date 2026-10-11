# Generative Process Design — Technical Blueprint

Version 0.1 · 2026-10-11 · Derived from `generative-process-design-corpus.md` (2026-08-31)

This is the build document. Every workstream below has deliverables, interfaces, acceptance criteria, tests, and fallbacks. Build order is the section order.

---

## 0. Decisions assumed (override before WS0 starts)

| # | Decision | Default | Why |
|---|---|---|---|
| D1 | Scope | **Copilot first** (autocomplete, PFD→P&ID, correction). Phase 3 extraction gated. | Corpus doc §4. Public data caps real topologies at ~10³. |
| D2 | Language / runtime | Python 3.11, `uv`, PyTorch 2.x | Every upstream dependency is Python. |
| D3 | Canonical form | SFILES 2.0 string ⇄ `nx.MultiDiGraph` via SFILES2 toolkit (MIT) | Literature standard. Canonical string = dedup key. |
| D4 | Extractor isolation | Each source parser runs in its own venv/container and emits a common **IR** (JSON). Core pipeline never imports IDAES, WaterTAP, BioSTEAM, or pyDEXPI. | Dependency conflicts across Pyomo stacks. AGPL isolation for pyDEXPI. |
| D5 | License tracks | Two build profiles: `research` (everything) and `commercial` (excludes CC-BY-NC, quarantines unconfirmed). Default CI build = `commercial`. | Corpus doc §3. HUPD is NC. FOSSEE unconfirmed. |
| D6 | Dev platform | macOS dev. Linux containers for DWSIM/Mono and pyDEXPI. Windows only as optional CI lane (COCO COM). | User environment. COCO/HYSYS are Windows-only. |
| D7 | Model track | Track A: small from-scratch transformers (10–40M params). Track B fallback: LoRA fine-tune of a ≤1.5B open LLM. | Literature uses from-scratch transformers at this data scale. Cheap, reproducible. |
| D8 | Compute | Single GPU (cloud, rented per run). Mac MPS for smoke tests only. | Model sizes train in hours on one GPU. |
| D9 | Storage | Raw data gitignored, fetched by script with checksums. Corpus = JSONL + Parquet manifest. DuckDB for queries. No DVC. | Small team. Keep tooling minimal. |
| D10 | Timeline | 16 weeks, 5 milestones, hard gate at M4 for Phase 3. | Corpus doc: 3–4 months. |

---

## 1. Goal, scope, success criteria

**Goal.** A corpus + model + API that, given a partial or complete chemical process flowsheet, (a) suggests next unit operations, (b) generates a control structure (PFD→P&ID), and (c) detects and fixes topology errors.

**In scope**
- ~1,000–1,200 real semantic flowsheets from public simulator files, DEXPI XML, GraphML.
- Hirtreiter PFD→P&ID corpus (10k–100k pairs).
- 50k–100k synthetic valid-by-construction topologies.
- Validity layer: grammar + 33 engineering rules + optional simulator round-trip.
- Three trained models + eval harness + serving API + CLI.

**Out of scope (v1)**
- Operating parameters, stream conditions, sizing, economics.
- Open-ended novel process synthesis.
- Phase 3 text/image extraction beyond a gated experiment.

**Success criteria (v1 release)**

| Metric | Target | Source of bar |
|---|---|---|
| Real flowsheets in corpus (deduped, validity-passing) | ≥ 1,000 | Corpus doc §3 Phase 1 yield |
| Synthetic flowsheets | ≥ 50,000 | Corpus doc §3 Phase 2 |
| Generated-output grammar validity rate | ≥ 99% | Grammar filter makes this near-guaranteed |
| Generated-output rule-violation rate | ≤ 5% | Validity layer |
| PFD→P&ID top-1 / top-5 on Hirtreiter test split | ≥ 70% / ≥ 85% | Published 74.8% / 89.2% |
| Autocompletion next-unit top-5 on held-out real | ≥ 70% | Graph-to-SFILES published 73.2% |
| Correction fix rate / false-positive rate | ≥ 80% / ≤ 5% | Internal |
| Test coverage: `canonical/`, `validity/` | ≥ 90% | Internal |
| Test coverage: `parsers/`, `corpus/` | ≥ 85% | Internal |
| Every record has `license` ≠ `unknown` in `commercial` profile | 100% | Hard gate |

---

## 2. Architecture

### 2.1 Data flow

```
 Sources (raw, gitignored)
 ┌──────────┬──────────┬──────────┬───────────┬──────────┬───────────┐
 │ DWSIM    │ Aspen    │ Pyomo    │ DEXPI XML │ GraphML  │ Modelica  │  ...Hirtreiter JSON, generators
 │ .dwxmz   │ .bkp     │ IDAES/   │ Proteus   │ PID2Graph│ .mo       │
 │          │          │ WaterTAP/│           │          │           │
 │          │          │ BioSTEAM │           │          │           │
 └────┬─────┴────┬─────┴────┬─────┴─────┬─────┴────┬─────┴─────┬─────┘
      │          │          │           │          │           │
      ▼          ▼          ▼           ▼          ▼           ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  EXTRACTORS  (isolated envs; one per source; emit IR JSON)      │
 │  extractors/<source>/   →  data/interim/<source>/*.ir.json      │
 └──────────────────────────────┬──────────────────────────────────┘
                                ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  CANONICAL   gpd/canonical/  IR → nx.MultiDiGraph → SFILES 2.0  │
 │  typemap (source class → SFILES class), hex pairing, recycles   │
 └──────────────────────────────┬──────────────────────────────────┘
                                ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  VALIDITY    gpd/validity/   grammar → rules(33) → [sim]        │
 │  every record gets validity flags; nothing is silently dropped  │
 └──────────────────────────────┬──────────────────────────────────┘
                                ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  CORPUS      gpd/corpus/     dedup → license filter → splits →  │
 │  augmentation (train only) → manifest.parquet + shards.jsonl    │
 └──────────────────────────────┬──────────────────────────────────┘
                                ▼
 ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
 │ MODELS           │  │ EVAL             │  │ SERVE            │
 │ tokenizer        │→ │ benchmarks       │  │ FastAPI + CLI    │
 │ A: autocomplete  │  │ regression gate  │  │ validity-filtered│
 │ B: pfd2pid       │  │                  │  │ top-k            │
 │ C: correction    │  │                  │  │                  │
 └──────────────────┘  └──────────────────┘  └──────────────────┘
```

### 2.2 Repository layout

```
generative-process-design/
├── BLUEPRINT.md                      # this file
├── generative-process-design-corpus.md
├── pyproject.toml                    # core package `gpd`; deps: networkx, pydantic, duckdb, pyarrow, typer, torch
├── Makefile                          # fetch / extract / build-corpus / train / eval / smoke / test
├── gpd/
│   ├── canonical/
│   │   ├── ir.py                     # IR pydantic schema (v1)
│   │   ├── typemap/                  # YAML: source class → SFILES class, per source
│   │   ├── to_graph.py               # IR → nx.MultiDiGraph (SFILES2 conventions)
│   │   ├── sfiles.py                 # thin wrapper over SFILES2 Flowsheet; canonical string; parse
│   │   ├── hashing.py                # canonical hash + WL hash
│   │   └── tokenizer.py              # SFILES-aware tokenizer
│   ├── validity/
│   │   ├── grammar.py                # parse round-trip + structural checks
│   │   ├── rules/                    # one module per rule, R01..R33 + structural S01..Sxx
│   │   ├── engine.py                 # run rules, collect Violations, apply fixes
│   │   └── sim/                      # DWSIM round-trip client (optional lane)
│   ├── corpus/
│   │   ├── record.py                 # Record schema
│   │   ├── dedup.py
│   │   ├── license.py                # profiles: research | commercial
│   │   ├── splits.py                 # group-aware, leakage-checked
│   │   ├── augment.py                # SFILES2 branch permutation wrapper
│   │   └── build.py                  # manifest + shards
│   ├── synth/
│   │   ├── sfiles2_gen.py            # wrapper over create_random_flowsheet
│   │   ├── pydexpi_client.py         # talks to the AGPL service over files
│   │   └── rl_gen.py                 # SynGameZero / drl4procsyn wrappers (optional)
│   ├── extract/                      # Phase 3, gated
│   │   ├── text/
│   │   └── image/
│   ├── models/
│   │   ├── data.py                   # datasets, collators
│   │   ├── lm.py                     # Task A decoder-only
│   │   ├── seq2seq.py                # Task B / C encoder-decoder
│   │   ├── train.py
│   │   └── decode.py                 # grammar-constrained top-k
│   ├── eval/
│   │   ├── benchmarks.py
│   │   └── baselines.json            # frozen numbers; CI regression gate
│   ├── serve/
│   │   ├── api.py                    # FastAPI
│   │   └── cli.py                    # typer `gpd`
│   └── util/
├── extractors/                       # ISOLATED. Each has own pyproject/venv or Dockerfile.
│   ├── dwsim/                        # pure-Python XML; can live in core but kept here for uniformity
│   ├── aspen_bkp/
│   ├── pyomo_idaes/                  # venv pinned to IDAES
│   ├── pyomo_watertap/               # venv pinned to WaterTAP
│   ├── biosteam/                     # venv pinned to BioSTEAM
│   ├── dexpi/                        # Docker; pyDEXPI (AGPL); also hosts syndata generator service
│   ├── pid2graph/
│   ├── modelica/
│   ├── coco/                         # Windows-only lane
│   └── hirtreiter/
├── scripts/
│   ├── fetch.py                      # downloads with sha256 manifest → data/raw/
│   └── sources.yaml                  # URL, license, checksum, tier, expected count per source
├── data/                             # gitignored except fixtures
│   ├── raw/<source>/
│   ├── interim/<source>/*.ir.json
│   └── corpus/<profile>/{manifest.parquet, train.jsonl, val.jsonl, test.jsonl, stats.json}
├── tests/
│   ├── fixtures/<source>/            # tiny real or hand-made files (license-permitting)
│   ├── unit/
│   ├── property/
│   ├── integration/
│   └── data_quality/
└── .github/workflows/
    ├── ci.yml                        # lint, unit, property, smoke (<10 min)
    ├── corpus-nightly.yml            # full build + data-quality gates + eval regression
    └── coco-windows.yml              # optional lane
```

### 2.3 Tech stack

| Layer | Choice |
|---|---|
| Core | Python 3.11, `uv`, `ruff`, `mypy --strict` on `gpd/canonical` and `gpd/validity` |
| Graphs | `networkx` ≥ 3.2; SFILES2 vendored as git submodule pinned to a commit |
| Schema | `pydantic` v2 |
| Storage | JSONL shards, Parquet manifest (`pyarrow`), DuckDB for ad-hoc queries |
| Tests | `pytest`, `hypothesis`, `pytest-cov`, `pytest-xdist` |
| Models | PyTorch 2.x; Track B uses `transformers` + `peft` |
| Serving | FastAPI + uvicorn; `typer` CLI |
| Containers | Docker for `extractors/dexpi` and `gpd/validity/sim` |
| Tracking | W&B or MLflow (pick one at WS5 start); metrics also written to `eval/runs/*.json` |

---

## 3. Canonical data model

### 3.1 Intermediate Representation (IR) — `gpd/canonical/ir.py`

Every extractor emits this. Nothing else crosses the extractor boundary.

```python
class IRNode(BaseModel):
    id: str                       # extractor-local, stable within file
    source_class: str             # raw class string from source (e.g. "DWSIM.UnitOperations.UnitOperations.Heater")
    name: str | None              # human label if present
    attrs: dict[str, str | float | int | bool] = {}   # anything useful; never required downstream

class IREdge(BaseModel):
    src: str                      # IRNode.id
    dst: str                      # IRNode.id
    kind: Literal["material", "energy", "signal"]
    src_port: str | None = None   # e.g. "tout", "bout", "hot_out"
    dst_port: str | None = None
    name: str | None = None

class IRFlowsheet(BaseModel):
    ir_version: Literal["1"] = "1"
    source: str                   # e.g. "fossee_dwsim"
    source_uri: str               # URL or path
    source_file: str              # relative path within raw/
    source_sha256: str
    license: str                  # SPDX or "unknown" | "pending"
    tier: Literal["A","B","C","D","E"]
    domain: Literal["chemical","power","water","bio","adjacent"]
    extraction_method: Literal["parser","runtime","com","graphml","generator","llm","digitizer"]
    extraction_confidence: float = 1.0   # 1.0 for parsers; <1 for Tier C/D
    nodes: list[IRNode]
    edges: list[IREdge]
    notes: list[str] = []         # extractor warnings (unmapped classes, dropped energy streams)
```

Rules:
- Streams are **edges**, not nodes. Extractors collapse stream objects. A plant-boundary stream (feed/product) has `src=None` or `dst=None`; canonicalization creates the `raw`/`prod` node. Energy and signal edges may not cross the boundary.
- Controllers are nodes with `ctrl_fn` set (ISA letters, e.g. `FC`); the class is forced to `C` and the node id becomes `C-<n>/<FN>`.
- Energy edges are kept in IR, dropped at canonicalization (SFILES models material flow; heat exchanger pairing is captured by `hex` tags).
- Signal edges are kept; they become SFILES 2.0 control notation.
- Unmapped `source_class` → node kept, `notes` entry added, canonicalization assigns the "unk" class, which renders as SFILES2's generic `X` (ProcessUnit) so strings stay parseable upstream. Records with any `unk` node are flagged (`has_unk_nodes`), not dropped.

### 3.2 Canonical graph — `gpd/canonical/to_graph.py`

- `nx.MultiDiGraph`, node ids follow SFILES2 convention `<class>-<n>` (e.g. `hex-1`, `dist-1`, `r-1`, `mix-1`, `splt-1`, `pp-1`, `comp-1`, `v-1`, `raw-1`, `prod-1`, controllers `C-<n>`). Exact vocabulary: read from `SFILES2/Flowsheet_Class/OntoCape_SFILES_mapping.py` at WS0 and frozen into `gpd/canonical/vocab.py` with a test that fails if upstream changes.
- Edge attrs: `tags: {"he": [...], "col": [...], "signal": [...]}` as SFILES2 expects.
- `typemap/<source>.yaml`: `source_class → sfiles_class` plus port-name normalization. Every typemap has a test asserting every value is in `vocab.py`.
- Heat exchangers: sources that represent one exchanger as one unit with two passes → SFILES2 `hex-N/1`, `hex-N/2` convention. Use `Flowsheet.split_HI_nodes` / `merge_HI_nodes`.
- Feeds and products: any node with in-degree 0 (material) becomes `raw-N`; out-degree 0 becomes `prod-N`, unless the source marks it otherwise.

### 3.3 SFILES wrapper — `gpd/canonical/sfiles.py`

```python
def graph_to_sfiles(g: nx.MultiDiGraph, canonical: bool = True) -> str
def sfiles_to_graph(s: str) -> nx.MultiDiGraph
def canonical_sfiles(s: str) -> str          # parse → regenerate canonical
def is_isomorphic(a: nx.MultiDiGraph, b: nx.MultiDiGraph) -> bool   # SFILES2 check_isomorphism
```
Wraps `Flowsheet.create_from_nx`, `convert_to_sfiles(version="v2", remove_hex_tags=True, canonical=True)`, `create_from_sfiles`. All SFILES2 exceptions are caught and re-raised as `CanonicalizationError` with the offending graph serialized to `data/interim/_failures/`.

Verified behaviours that shape this wrapper (2026-10-11):
- `remove_hex_tags=True` is mandatory. SFILES2 cannot re-parse strings carrying `hot_in`/`cold_out` side labels (fails on 2 of the 9 real DWSIM graphs). Heat-integration pairing `{n}` tags survive; only the hot/cold side assignment is lost from the canonical string. The normalized graph keeps numeric `1_in/2_out` he tags.
- SFILES2 breaks ranking ties by networkx iteration order. The wrapper rebuilds every input graph with sorted nodes and parallel edges sorted by tag label before conversion; without this, a flash whose top and bottom both feed one mixer flips `{tout}`/`{bout}` depending on construction history.
- SFILES2 parses malformed strings leniently (`(raw)[(hex)` → `(raw)(hex)`). It is not a grammar checker. G2 (canonical idempotence) in WS4.1 is the grammar check.
- The canonical **graph** is the parse of the canonical string (`canonicalize()` returns both). All hashing and validity run on that normalized graph.

### 3.4 Hashing — `gpd/canonical/hashing.py`

- `canonical_hash = sha256(canonical_sfiles)` — primary dedup key.
- `wl_hash = nx.weisfeiler_lehman_graph_hash(g, node_attr="class", edge_attr="kind", iterations=4)` — secondary key. Catches cases where canonicalization is unstable.
- Two records are duplicates if `canonical_hash` matches **or** (`wl_hash` matches and `is_isomorphic` is True).

### 3.5 Record — `gpd/corpus/record.py`

```python
class Validity(BaseModel):
    grammar_ok: bool
    rule_violations: list[str]        # rule ids
    rules_ok: bool                    # no violations of severity error
    sim_converged: bool | None        # None = not run
    has_unk_nodes: bool

class Record(BaseModel):
    id: str                            # f"{source}:{sha256(source_file)[:12]}:{idx}"
    sfiles: str                        # canonical; includes control notation when present
    sfiles_pfd: str | None             # control-stripped variant, only when has_control
    graph: dict                        # gpd.canonical.graphio JSON (own shape, versioned)
    canonical_hash: str
    wl_hash: str
    n_units: int
    n_streams: int
    has_recycle: bool
    has_control: bool
    provenance: IRFlowsheet-minus-nodes/edges (source, uri, file, sha, license, tier, domain, method, confidence)
    validity: Validity
    is_real: bool                      # False for generators and augmentation
    augmented_from: str | None         # parent record id
    split: Literal["train","val","test"] | None
```

### 3.6 Splits — `gpd/corpus/splits.py`

- **Frozen real test set** created at M2 and never modified: 120–150 real records, stratified by `domain` and `n_units` tercile. Stored as `data/corpus/test_ids.frozen.txt` and committed.
- Group-aware split: records sharing `source_file` sha, or any duplicate relation, stay in one split. FOSSEE and the mirror overlap; they are grouped.
- Augmentation happens **after** splitting, train only.
- Hirtreiter's own train/test split is preserved verbatim in a separate benchmark set so published numbers are comparable.
- Leakage test: zero `canonical_hash` or `wl_hash` shared across splits. CI fails otherwise.

---

## 4. Workstreams

Each WS lists: deliverables, interface, acceptance, tests, fallbacks. "Owner" fields are left blank for assignment.

### WS0 — Foundation (week 1)

**Deliverables**
1. Repo scaffold per §2.2. `pyproject.toml`, `Makefile`, CI skeleton.
2. SFILES2 vendored as pinned submodule. `vocab.py` frozen from upstream mapping.
3. `ir.py`, `record.py`, `sfiles.py`, `hashing.py`, `to_graph.py` with a single hand-written typemap (`dwsim.yaml`, partial).
4. `scripts/sources.yaml` listing every Tier A/B source with URL, expected license, expected count. `scripts/fetch.py` downloads with sha256 verification into `data/raw/`.
5. `tests/fixtures/` seeded with: 3 hand-written SFILES strings (simple, recycle, control), 1 DEXPI TrainingTestCase XML (CC-BY), 1 hand-made GraphML, 1 hand-made `.mo`.

**Week-1 verification items** (unknowns found during blueprint research)
- [ ] Confirm `create_random_flowsheet` works out of the box — it depends on a `PID_generator`/`RandomFlowsheetGen_OC` module that may not ship in the SFILES2 repo. If absent → see WS2 fallback.
- [ ] Confirm DWSIM XML tag names for object type and connector attachment against 3 real `.dwxmz` files (tag names in §WS1.1 are to be verified).
- [ ] Confirm Aspen `.bkp` files contain the input-language `FLOWSHEET` section in ASCII.
- [ ] Confirm the 9 real DWSIM graphs shipped in SFILES2 load with `create_from_nx`.

**Acceptance**
- `make test` green. `make fetch SOURCE=hirtreiter` downloads and verifies.
- Round-trip property test passes 1,000 Hypothesis examples.

**Tests**
- `tests/property/test_roundtrip.py`: random valid graphs (built from vocab with degree constraints) → SFILES → graph is isomorphic. Hypothesis strategy in `tests/strategies.py`.
- `tests/property/test_hash.py`: canonical hash invariant under node relabeling and edge reordering.
- `tests/unit/test_vocab.py`: fails if upstream SFILES2 mapping keys differ from frozen vocab.
- `tests/unit/test_fetch.py`: checksum mismatch raises; partial download is not left in `raw/`.

**Fallbacks**
- SFILES2 canonicalization unstable on some graphs → WL hash + isomorphism check as dedup fallback (already designed in).
- SFILES2 submodule drifts → pinned commit; vocab test catches it.

---

### WS1 — Simulator harvest (weeks 1–5)

Target: ≥ 1,000 real flowsheets after dedup. Yield estimates from corpus doc.

#### WS1.1 DWSIM (`extractors/dwsim/`) — est. 450–500 unique

- Input: `.dwxmz` (zip containing one XML) and `.dwxml`. Sources: FOSSEE 275, Sashi20 mirror 194, DWSIM built-ins 186.
- Method: `zipfile` → `defusedxml` parse. Iterate simulation objects; read object type (full .NET class name) and name; read input/output connector attachments to resolve unit ↔ stream ↔ unit. Collapse `MaterialStream` objects to material edges; `EnergyStream` to energy edges. Exact XML tag names: **verify in WS0** against samples; keep tag names in a single constants block.
- Typemap: `typemap/dwsim.yaml` covering all DWSIM unit-op classes (~40). Unknown → `unk` + note.
- Dedup expectation: FOSSEE ⊃ mirror. Record both provenances on the surviving record.
- License: FOSSEE `pending` until confirmed by email (side quest, send week 1). Built-ins GPL-3.0 (data files; record license, no code copied). Mirror `unknown`.

Acceptance: parse-success ≥ 95% of files; `unk` nodes ≤ 2% of all nodes; ≥ 400 unique after dedup.

Tests: golden fixture (one small built-in sample, checked in with attribution) → exact expected IR JSON; malformed zip → `ExtractorError`, not crash; energy streams never appear in canonical graph; typemap completeness test.

Fallback: if connector tags differ across DWSIM versions, branch on `<Version>` header; if FOSSEE license is refused, exclude 275 files from `commercial` profile (yield falls to ~300; still proceed).

#### WS1.2 Aspen `.bkp` (`extractors/aspen_bkp/`) — est. ~250 unique

- Input: `.bkp` from GitHub code search (313), NREL, PSEcommunity.
- Method: `.bkp` is ASCII and embeds the Aspen input language. Parse the `FLOWSHEET` section: `BLOCK <id> IN=<s1> <s2> OUT=<s3>` lines give connectivity; `BLOCK <id> <TYPE>` declarations give class. Streams are edges by construction. Hierarchy blocks: flatten one level; nested deeper → note + skip.
- Typemap: `typemap/aspen.yaml` (RadFrac→dist, HeatX→hex, RStoic/RPlug/RCSTR→r, Mixer→mix, FSplit→splt, Pump→pp, Compr→comp, Valve→v, Flash2→flash, …).
- License: per-repo. Most absent → `unknown`. These go to `research` profile only unless a LICENSE file exists.
- Collection: `gh search code extension:bkp aspen` via API; clone shallow; store repo URL + commit in provenance.

Acceptance: ≥ 70% of files yield a graph with ≥ 3 units; ≥ 200 unique.

Tests: hand-written minimal `.bkp` fixture (synthetic text, not from any real file) with IN/OUT lines → expected IR; binary/garbage file → clean failure; hierarchy fixture.

Fallback: ASCII parse coverage < 70% → accept reduced yield; Aspen COM automation is **not** planned (needs Windows + license). If GitHub files are mostly student projects with trivial topologies, filter `n_units ≥ 3`.

#### WS1.3 Pyomo networks (`extractors/pyomo_idaes/`, `pyomo_watertap/`, `biosteam/`) — est. 80–100

- Three separate venvs (pinned). Each exposes `python -m extractor <module_glob> --out dir/`.
- IDAES / WaterTAP: import flowsheet module, call its `build()` (not `solve`); iterate `m.component_data_objects(pyomo.network.Arc)`; `arc.source.parent_block()` / `arc.destination.parent_block()` are the unit models; class = Python type name. Ports named `inlet`/`outlet`/`hot_side_inlet` etc. → port normalization in typemap.
- BioSTEAM: import biorefinery package, get `System`; `for u in system.units: for s in u.outs: s.sink` gives edges.
- Flowsheets that require a solver to build (some IDAES examples initialize inside `build`) → run with `ipopt` available in the venv; if build fails, note and skip.
- Domain tags: IDAES → `power`/`chemical`; WaterTAP → `water`; BioSTEAM → `bio`.

Acceptance: ≥ 60 unique graphs; zero core-package imports of IDAES/WaterTAP/BioSTEAM (import-linter rule).

Tests: tiny Pyomo fixture model with 3 units and 2 Arcs in `tests/fixtures/pyomo/` run through the IDAES extractor's generic Arc walker; import-linter contract test; typemap completeness per venv.

Fallback: a venv fails to build on CI → run extractor in Docker; if a library breaks upstream, pin to last-known-good tag in `sources.yaml`.

#### WS1.4 DEXPI Proteus XML (`extractors/dexpi/`, Docker, AGPL) — est. ~83 cases, ~14 realistic

- `ProteusSerializer().load(path)` → `DexpiModel` → `GraphLoader().dexpi_to_graph()` → `GraphLoader.build_process_graph()` (collapses piping segments into edges) → map DEXPI classes to IR.
- Keep instrumentation: DEXPI `ProcessInstrumentationFunction` / `ActuatingFunction` → signal edges → SFILES 2.0 control notation. This is the only Tier A source with rich real control structure outside Hirtreiter.
- Container exposes one command: `dexpi-extract <in_dir> <out_dir>`. Core never imports pyDEXPI.
- Also hosts WS2 pyDEXPI synthetic generator behind the same boundary.

Acceptance: all 14 realistic cases parse; the remaining test-case folders either parse or are noted as "schema test only".

Tests: one TrainingTestCase XML fixture (CC-BY) → expected IR; container build test in CI; license guard test: `grep -r pydexpi gpd/` returns nothing.

Fallback: pyDEXPI fails on specific Proteus dialects → fall back to a minimal own Proteus XML reader (equipment + piping network segments + nozzles) that covers the 14 realistic cases only.

#### WS1.5 PID2Graph GraphML (`extractors/pid2graph/`) — 72 real

- `nx.read_graphml`; node labels are symbol classes → `typemap/pid2graph.yaml`.
- Only the **72 real** P&IDs enter the flowsheet corpus. The 700 random-topology synthetic ones are tagged `is_real=False, domain=adjacent` and reserved for the Phase 3 digitizer only. Random topology ≠ engineering-valid.
- License CC-BY-SA-4.0 → record flag `share_alike=True`; corpus release artifacts containing these inherit SA.

Acceptance: 72/72 parse; mapping coverage ≥ 95% of node labels.

Tests: hand-made GraphML fixture; SA propagation test (any shard containing an SA record has SA in its manifest license summary).

Fallback: none needed; GraphML is trivial.

#### WS1.6 Modelica (`extractors/modelica/`) — thousands, pretraining only

- Parse `.mo` files: locate `model X … equation … end X;` blocks; collect component declarations (`Type name(...)`) and `connect(a.port, b.port);` statements. Regex-based first; upgrade to a real grammar only if coverage < 80%.
- Libraries: ClaRa, ThermoSysPro, ThermoPower, TransiEnt via OMPackageManager index; MSL and LBNL Buildings as adjacent.
- Tag `domain=adjacent` (or `power` for ClaRa/ThermoSysPro), `tier=A`, `is_real=True` but **excluded from the real fine-tune set and from the frozen test set**. Used in pretraining only.
- Typemap: coarse (any heat exchanger class → hex, pump → pp, valve → v, turbine → `turb` if in vocab else `unk`). Expect many `unk`; acceptable for pretraining.

Acceptance: ≥ 2,000 graphs with ≥ 4 units; `unk` ≤ 30%.

Tests: hand-made `.mo` fixture with nested `connect`; regex robustness on comments/strings containing "connect(".

Fallback: if pretraining on Modelica hurts downstream metrics (ablation in WS5), drop it. It costs little to collect.

#### WS1.7 COCO `.fsd` (`extractors/coco/`, Windows lane, optional) — ~90

- COM automation on a GitHub Actions `windows-latest` runner with COCO installed silently. Script enumerates unit operations and stream connections.
- Runs only in `coco-windows.yml`, manually triggered. Output IR committed to a release asset, not to git.

Acceptance: ≥ 60 graphs or the lane is declared dropped at M2.

Fallback: drop. Loss ≈ 8% of real yield. No other path without Windows.

#### WS1.8 HYSYS — dropped

Binary `.hsc`, needs license. Not planned. Recorded here so nobody re-investigates.

**WS1 exit (M2 gate)**: `data/corpus/research/stats.json` shows ≥ 1,000 real records with `grammar_ok=True`; per-source yields within 30% of corpus-doc estimates or a written explanation exists.

---

### WS2 — Existing corpora + synthetic scale-up (weeks 3–6)

#### WS2.1 Hirtreiter ingest (`extractors/hirtreiter/`)

- Unzip `Training Data Public Upload.zip` (24.9 MB, CC-BY-4.0). JSON of PFD SFILES ↔ P&ID SFILES pairs with train/test/augmented splits.
- Validate every string parses via `sfiles_to_graph`; failures logged, not dropped (store `grammar_ok=False`).
- Preserve their split assignment in `provenance.attrs.hirtreiter_split`. Their test split becomes benchmark `pfd2pid_hirtreiter_test` and is excluded from all our training.
- Their augmented split is **not** re-augmented.

Acceptance: ≥ 99% of strings parse; benchmark set frozen.

Tests: 5 pairs fixture; parse-rate test; benchmark-isolation test (no Hirtreiter test record in our train).

Fallback: format surprise → adapter; if parse rate < 95%, diff SFILES dialects (their strings may predate SFILES 2.0 final) and add a compatibility shim in `sfiles.py`.

#### WS2.2 SFILES2 random generator (`gpd/synth/sfiles2_gen.py`)

- `Flowsheet().create_random_flowsheet()` in a loop with seeds; target 50k; dedup by canonical hash; validity layer applied; keep only `rules_ok`.
- Distribution control: sample `n_units` to match the real corpus histogram ±; log KL divergence of class distribution vs real.

Acceptance: 50k unique grammatical, rule-passing flowsheets; class-distribution KL vs real ≤ 0.5.

Tests: determinism under fixed seed; every output parses; distribution test on 1k sample.

Fallback (likely needed — WS0 verification item): generator module absent → write `gpd/synth/template_gen.py`: compose from a library of 15–25 engineering motifs (feed-pump-heat-react-separate-recycle, distillation train, absorber-stripper, compressor-cooler-flash) with random branching and recycle insertion. Validate with rules. This is less diverse but valid by construction.

#### WS2.3 pyDEXPI synthetic P&IDs (`extractors/dexpi/` service + `gpd/synth/pydexpi_client.py`)

- Inside the AGPL container: `pydexpi.syndata` generator with `RandomGeneratorFunction` over pattern library → DexpiModel → `build_process_graph` → IR JSON. Target 20–50k.
- Core consumes IR files only.

Acceptance: 20k unique with control structure; rules pass ≥ 90%.

Fallback: pattern library too narrow → extend patterns from DEXPI TrainingTestCases (inside container). Or lean on WS2.2 only.

#### WS2.4 RL-synthesized flowsheets (optional)

- SynGameZero / drl4procsyn (MIT) emit purposeful topologies but narrow chemistry. Run their trained policies if checkpoints are published; else skip. Budget: 2 days. Tag `is_real=False, domain=chemical, extraction_method=generator`.

Fallback: skip. Not on the critical path.

#### WS2.5 Augmentation (`gpd/corpus/augment.py`)

- `SFILES2.data_augmentation.canonical_to_noncanonical_sfile(sfiles, version=2, sfiles_amount=k)`; k=5–10 per real train record.
- Augmented records point to parent via `augmented_from`; same split as parent; never in val/test.

Tests (property): every augmented string parses to a graph isomorphic to the parent; no augmented string equals the canonical parent; augmented records never appear in val/test.

**WS2 exit (M2)**: corpus has ≥ 1,000 real + Hirtreiter + ≥ 50k synthetic; frozen test set committed.

---

### WS3 — Extraction mining (weeks 4–12, parallel, **gated**)

Go/no-go at **M4 (week 12)** on validity-layer pass rate. Until then, WS3 produces a 200-sample pilot per modality, not a corpus.

#### WS3.1 Text → flowsheet (`gpd/extract/text/`)

- Engine: ETH Text-to-Flowsheet pipeline (Zenodo 19910216, `text2flowsheet.zip`, CC-BY-4.0). Adapt output to IR.
- Inputs, in order: NREL/PNNL design reports (US-gov open), NETL Bituminous Baseline, IEAGHG reports, Northwestern/Penn capstone reports, then CPC C/B01-filtered USPTO grant XML.
- LLM backend: Claude via the API (default `claude-sonnet-5-5` for throughput; `claude-opus-5-5` for the 200-sample pilot quality ceiling). Every call logs prompt hash, model id, cost.
- `extraction_confidence` = model self-reported confidence × grammar pass × rules pass (0–1).
- Human spot-check protocol: 100 random pilot outputs, two reviewers, binary "topology correct"; agreement ≥ 0.8 required for the protocol itself to count.

Gate: human-judged topology-correct rate ≥ 60% **and** rules pass ≥ 70% → go, target 5k–50k. Else → Tier D excluded from training; pipeline kept for future.

Tests: 3 short paragraph fixtures with known flowsheets → IR within graph edit distance ≤ 2; LLM calls mocked in unit tests (recorded responses); live test runs nightly only.

#### WS3.2 Image → flowsheet (`gpd/extract/image/`)

- Digitizer trained on PID2Graph (772 incl. 700 synthetic) + PID_dataset (6.7 GB, CC-BY-4.0) + Digitize-PID (500 synthetic). Relationformer-style or Azure sample pipeline (MIT) as baseline.
- Expected ~75% edge mAP pre-repair → mandatory rule-based repair → still noisy.
- Pilot: 200 P&ID/PFD drawings from DOE/NETL FEED studies.

Gate: post-repair human-judged topology-correct ≥ 50% → go, else excluded. Expectation: **no-go** for v1. Budget capped at 3 engineer-weeks.

Fallback: use PID2Graph real 72 only (already in WS1.5). Skip digitization.

---

### WS4 — Validity layer (weeks 2–8, then continuous)

#### WS4.1 Grammar (`gpd/validity/grammar.py`)

Checks, each a boolean with a reason:
- G1 parses with SFILES2.
- G2 parse → regenerate canonical → parse again is isomorphic (idempotence).
- G3 every node class ∈ vocab.
- G4 every material-connected component contains ≥ 1 `raw` and ≥ 1 `prod` (reachability from a feed to a product).
- G5 no self-loops; no duplicate parallel material edges without distinct port tags.
- G6 degree constraints per class (table in `vocab.py`): e.g. `mix` in ≥ 2 / out = 1; `splt` in = 1 / out ≥ 2; `dist` in ≥ 1 / out ≥ 2 with `tout`/`bout` tags; `hex` has paired passes; `pp`, `comp`, `v` in = 1 / out = 1.
- G7 control: every `C` node has a signal edge to a valid actuator class (`v`, `pp`, `comp`, …) and a measurement source.

#### WS4.2 Engineering rules (`gpd/validity/rules/R01…R33.py`)

- Transcribe the 33 rules from arXiv 2502.18493 (no public code found; reimplement). Each rule:
  ```python
  class Rule(Protocol):
      id: str; severity: Literal["error","warning"]; description: str
      def check(self, g: nx.MultiDiGraph) -> list[Violation]
      def fix(self, g: nx.MultiDiGraph, v: Violation) -> nx.MultiDiGraph | None   # None = not auto-fixable
  ```
- Engine runs rules to a fixed point (max 10 passes), records violations before and after.
- Correction-model training data (WS5 Task C) is generated by **inverting** fixable rules: apply a corruption that produces exactly one known violation.

Acceptance: 33/33 rules implemented; each has ≥ 1 positive (violation detected) and ≥ 1 negative (clean graph passes) test; each `fix` is idempotent and reduces violation count.

Tests: per-rule unit tests; property: `fix` never introduces a new error-severity violation; engine terminates.

Fallback: a rule needs attributes the corpus lacks (e.g. pressure class) → implement as `warning`, document as "not evaluable on v1 data".

#### WS4.3 Simulator round-trip (`gpd/validity/sim/`, Docker, optional lane)

- Docker image: DWSIM + Mono/.NET + `pythonnet`. Client builds a `.dwxmz` from a graph with default components (water/ethanol binary) and default specs, runs steady-state, checks convergence flag.
- Only meaningful for a subset of classes; many topologies will not converge for parameter reasons unrelated to topology. Treat as a **positive-only signal**: `sim_converged=True` is strong evidence; `False` is weak evidence.
- Run on: all real records once; 5% sample of synthetic; all model outputs in eval.

Acceptance: container runs the 9 SFILES2 DWSIM graphs; ≥ 50% converge with defaults.

Fallback: if DWSIM automation on Linux is unreliable, lane is dropped; validity = grammar + rules only. Document in `stats.json`.

#### WS4.4 Dedup + license (`gpd/corpus/dedup.py`, `license.py`)

- Dedup per §3.4. Surviving record keeps a `duplicates: list[str]` of merged provenances.
- License profiles:

  | Profile | Includes | Excludes |
  |---|---|---|
  | `research` | all | none |
  | `commercial` | SPDX permissive (MIT, BSD, CC-BY, CC-BY-SA with SA flag, US-gov), GPL **data** files (recorded) | `unknown`, `pending`, any `*-NC-*` |

- `commercial` build fails hard if any record has `license ∈ {unknown, pending}`.

Tests: dedup on constructed near-duplicates; profile filter exact; SA propagation.

---

### WS5 — Models + evaluation (weeks 7–13)

#### WS5.1 Tokenizer (`gpd/canonical/tokenizer.py`)

- Regex tokenizer over SFILES 2.0: unit-class tokens (from vocab), unit numbers, brackets `()[]{}`, recycle markers `<n`/`n`, `|`, `&`, tag tokens (`tout`, `bout`, `hot_in`, …), control tokens. Vocab ≈ 150–300.
- Special tokens: `<bos> <eos> <pad> <sep>`.

Tests: `detok(tok(s)) == s` on every corpus string (contract test); every token in vocab; unknown token raises.

#### WS5.2 Task A — Autocompletion (`gpd/models/lm.py`)

- Decoder-only transformer, 8 layers, d=384, 6 heads, ~12M params (nanoGPT-style). Context 512 tokens (covers > 99% of real strings; verify at M2).
- Curriculum: (1) synthetic + Modelica, 3 epochs; (2) Hirtreiter PFD side + real augmented, until val loss plateaus.
- Decoding: top-k over **next unit operation** (multi-token), with grammar-constrained decoding (`decode.py`): mask tokens that cannot extend to a parsable string using an incremental SFILES2 parse check every unit boundary.
- Metrics: next-unit top-1/top-5 on frozen real test; fraction of 20-sample completions that pass grammar + rules; perplexity.

#### WS5.3 Task B — PFD→P&ID (`gpd/models/seq2seq.py`)

- Encoder-decoder, 6+6 layers, d=384, ~25M params. Input PFD SFILES, output control-extended SFILES.
- Train on Hirtreiter train (+ augmented) + DEXPI real control graphs + pyDEXPI synthetic. Evaluate on Hirtreiter test (published comparability) **and** on our frozen real set where control exists.
- Metrics: exact match top-1, top-5 (beam 5), per-control-loop precision/recall; validity of outputs.

Target: within 5 points of 74.8 / 89.2 at M3; at or above at M5.

**PoC-0 outcome (2026-10-11):** the string-to-string model above (7.5M params, 100k pairs) reaches top-5 81.1%, top-1 42.0%, parses 98.2%. It is a character copier with controllers sprinkled in: ~60% of each output is the PFD copied verbatim, every error is a syntax error before it is an engineering error, and the validity layer sits outside the generator. The revision below moves engineering structure inside the model. The seq2seq model stays as the published-comparability baseline.

**Design revision — Task B v2: loop-set prediction (`gpd/models/loops.py`)**

*Output space.* A P&ID is the PFD plus a set of control loops. Predict the set, render the string.

```python
class Loop(BaseModel):
    measured: str                       # PFD node id, e.g. "r-1"
    variable: Literal["F","T","P","L"]  # flow, temperature, pressure, level
    kind: Literal["C","I","IR","M"]     # controller, indicator, indicating-recorder, motor
    actuated: str | None                # PFD node id of the valve/pump/compressor; None for indicators
```
The P&ID string = `render(pfd_graph, loops)` — deterministic, grammar-valid by construction. `render` inserts `(C){xy}` nodes inline after the measured node, adds `(v)` actuators where the PFD lacks one on the actuated stream, numbers signal lines, and calls `graph_to_sfiles`. `parse_loops(pid_graph) -> list[Loop]` is the inverse; `render(pfd, parse_loops(pid)) == canonical(pid)` is a contract test over the entire Hirtreiter corpus.

*Model.*
- Encoder: transformer over PFD unit tokens with graph features per token (class, in/out degree, depth from feed, recycle membership, column tag) and an edge-aware attention bias (adjacency, 2-hop). Reuse the WS5.1 tokenizer; add the feature embeddings.
- Per-node heads: for every PFD node, a multi-label classifier over `(variable, kind)` pairs — ≤ 12 options. Nodes can carry several loops (reactor: TC + LC).
- Pointer head: for each predicted controller loop, attention over PFD nodes selects `actuated`. Candidates are restricted to actuator-capable classes (`v`, `pp`, `comp`, `blwr`, `turb`) plus "insert new valve on stream *s*", where *s* ranges over the measured node's incident streams.
- Hard masks encode engineering invariants: one controller per actuator; no actuator on a stream already actuated; indicators have no actuator; level loops only on inventory-holding classes (`tank`, `dist`, `flash`, `r`, `sep`, …); DOF balance — number of actuated streams ≤ number of manipulable streams in the component.
- Size: ~5–10M params. The decision space is ~12 choices per node, not 118 tokens × 130 steps.

*Training.*
- Supervised: per-node BCE on `(variable, kind)`, cross-entropy on the pointer. Labels from `parse_loops` over Hirtreiter pairs, DEXPI real P&IDs (the only real control data we have), pyDEXPI synthetic.
- Data-efficiency target: match the seq2seq model's 81% top-5 (rendered strings, same benchmark) using the **10k** set, not 100k. That is the test of whether the structured output space generalizes better — which is what matters for real PFDs, where we will have hundreds of examples.
- Phase 2 (after WS4): RL fine-tune with GRPO on the loop decoder. Reward = rule-pass rate (33 rules) + DOF balance + loop-level F1 against the label. The model internalizes what the verifier checks.
- Phase 3 (after WS4.3, if the sim lane works): sparse DWSIM-convergence reward. Last, because it is slow and noisy.

*Metrics.* Loop-level precision / recall / F1 (primary — what an engineer would check); rendered-string exact match top-1/top-5 on the Hirtreiter test split (comparability); parse rate (expected 100%); rule-violation rate.

*Decoding.* Greedy per node is usually enough; top-k alternatives come from the per-node distributions (e.g. "TC vs TI on the reactor"), which is exactly the ambiguity that cost the seq2seq model 33 top-1 points. Beam search over the whole string disappears.

*Acceptance.* Contract test `render(parse_loops(pid)) == canonical(pid)` ≥ 99% on Hirtreiter; loop F1 ≥ 0.90 on their test split; rendered top-5 ≥ 81% trained on 10k; parse rate 100%.

*Risks.* (1) `render` must reproduce the authors' string conventions (controller placement inline vs bracket, valve insertion, signal numbering) or exact-match comparability is lost — the contract test is the gate. (2) The 33 rules are not implemented yet; DOF masks ship first, the rules reward later. (3) Real control placement depends on attributes the data lacks (phase, pressure level). Topology-only prediction is pattern-matching the generator's rules; node `attrs` from the WS1 simulator harvest are the way beyond that.

*What this does not change.* Task A (autocompletion) and open-ended synthesis stay token/graph generation; their analogue of this revision is graph-native autoregressive generation with degree masks (WS5.2 constrained decoding first, graph generation later).

*Sequencing.* Governed by **PoC-1** (§8): shared contract layer and constrained decoding first, then the bake-off against the seq2seq baseline at 1k / 10k, then the fixed decision rule. RL fine-tuning only after WS4 and only for whichever model survives PoC-1.

#### WS5.4 Task C — Correction (`gpd/models/seq2seq.py`, shared arch)

- Data: corrupt clean records via inverted rules (WS4.2) and random edits (drop edge, swap classes, dangling stream). Pairs (corrupt → clean).
- Baseline: rules-only engine. Model must beat baseline on fix rate without raising false positives (model must return input unchanged for already-valid inputs ≥ 95% of the time).
- Metrics: fix rate (output isomorphic to clean), false-positive rate, rule-violation delta.

#### WS5.5 Track B fallback

- LoRA fine-tune `Qwen2.5-0.5B` (then 1.5B) on the same token streams rendered as text, same tasks. Run if any Track A metric is > 10 points below target at M3.

#### WS5.6 Evaluation harness (`gpd/eval/`)

- `benchmarks.py` runs all metrics on: frozen real test, Hirtreiter test, synthetic held-out. Outputs `eval/runs/<ts>.json`.
- `baselines.json` holds the current accepted numbers. Nightly CI fails if any primary metric drops > 2 points. Updating baselines requires a PR.
- Ablations logged at M3: with/without Modelica pretrain; with/without augmentation; with/without synthetic.

Tests: harness runs end-to-end on a 20-record fixture corpus with a randomly initialized model (smoke); metric functions unit-tested on hand-constructed predictions.

---

### WS6 — Serving (weeks 12–15)

- FastAPI, `gpd/serve/api.py`:
  - `POST /complete {sfiles_prefix, k}` → `[{sfiles, next_unit, score, validity}]`
  - `POST /pfd2pid {sfiles, k}` → `[{sfiles_ctrl, score, validity}]`
  - `POST /check {sfiles}` → `{grammar, violations[]}`
  - `POST /fix {sfiles}` → `{sfiles_fixed, applied_rules[], model_used}`
  - Every generated candidate passes through the validity engine before return. Invalid candidates are returned only if `include_invalid=true`.
- CLI `gpd complete|pfd2pid|check|fix|corpus stats|corpus build`.
- Latency target: p95 < 500 ms for k=5 on CPU for Task A; < 2 s for Task B.

Tests: API contract tests with a stub model; validity-filter test (an injected invalid candidate never leaks by default); CLI smoke.

Fallback: model too slow on CPU → int8 dynamic quantization; still too slow → GPU serving only, documented.

---

## 5. Test strategy

### 5.1 Pyramid

| Layer | Where | Runs | Budget |
|---|---|---|---|
| Lint + types | `ruff`, `mypy --strict` on `canonical/`, `validity/` | every PR | < 1 min |
| Unit | `tests/unit/` — parsers, typemaps, rules, tokenizer, metrics | every PR | < 3 min |
| Property | `tests/property/` — Hypothesis, 200 examples in CI, 2,000 nightly | every PR / nightly | < 3 min |
| Contract | `tests/unit/test_contracts.py` — every fixture record validates against `Record`; IR schema version pinned | every PR | included |
| Integration (smoke) | `tests/integration/test_smoke.py` — fixtures → extract → canonicalize → validate → build mini corpus → train 20 steps → eval → one API call | every PR | < 5 min |
| Data quality | `tests/data_quality/` against `data/corpus/<profile>/stats.json` | nightly, after full build | < 2 min |
| Model regression | `gpd/eval` vs `baselines.json` | nightly | GPU job |
| Human spot-check | WS3 only, protocol in §WS3.1 | at M4 gate | manual |

### 5.2 Coverage targets (enforced in CI via `pytest-cov --fail-under` per package)

| Package | Line coverage |
|---|---|
| `gpd/canonical` | ≥ 90% |
| `gpd/validity` | ≥ 90%; every rule has ≥ 1 positive and ≥ 1 negative test (enforced by a meta-test that introspects `rules/`) |
| `gpd/corpus` | ≥ 85% |
| `extractors/*` | ≥ 85% each (golden fixtures) |
| `gpd/models`, `gpd/eval` | ≥ 70% |
| `gpd/serve` | ≥ 80% |

### 5.3 Data-quality gates (nightly; fail the build)

| Gate | Threshold |
|---|---|
| Per-source parse-success rate | ≥ WS1 acceptance per source; drop > 5 points vs last run fails |
| Real records with `grammar_ok` | ≥ 1,000 at M2+ |
| `unk` node fraction, real Tier A | ≤ 5% |
| Split leakage (shared `canonical_hash` or `wl_hash` across splits) | 0 |
| Augmented records in val/test | 0 |
| Hirtreiter test records in any train shard | 0 |
| `license ∈ {unknown, pending}` in `commercial` profile | 0 |
| Class-distribution drift vs previous build (JS divergence) | ≤ 0.05 unless a new source was added |
| Duplicate rate after dedup (sampled isomorphism check on 1k pairs with equal `n_units`) | ≤ 0.5% |

### 5.4 Fixture policy

- Fixtures are tiny and license-clean. Real files only when the license permits redistribution (DEXPI CC-BY, DWSIM built-in GPL data with attribution, Hirtreiter CC-BY). Otherwise hand-written synthetic files that exercise the same code paths.
- Every fixture has a sibling `*.expected.ir.json`. Updating an expected file requires a PR comment explaining why.
- Hypothesis strategies live in `tests/strategies.py` and generate graphs from `vocab.py` degree constraints so they stay valid as the vocab evolves.

---

## 6. Fallback matrix

| Dependency / assumption | Failure mode | Detection | Fallback | Cost |
|---|---|---|---|---|
| SFILES2 `create_random_flowsheet` | generator module absent | WS0 check | Own template/motif generator (`template_gen.py`) | +1 week |
| SFILES2 canonicalization | unstable on edge cases | property tests, dedup sampling | WL hash + isomorphism as dedup fallback | built in |
| DWSIM XML schema | tag names differ by version | WS0 check, parse-rate gate | Version-branched constants | +2 days |
| FOSSEE license | refused / no reply | email, week 1 | `pending` → excluded from `commercial`; yield −275 raw | yield −20% |
| Aspen `.bkp` ASCII | FLOWSHEET section absent or encoded | WS0 check | Reduced yield; no COM | yield −10–20% |
| COCO (Windows) | no runner / COM fails | lane status at M2 | Drop lane | yield −8% |
| pyDEXPI on Proteus dialects | load errors | WS1.4 acceptance | Minimal own Proteus reader for 14 cases | +1 week |
| Pyomo venvs | upstream breakage | CI | Pin to last-good tag; Docker | +1 day |
| 33-rule paper | rules need missing attributes | WS4.2 | Implement as `warning` | none |
| DWSIM automation on Linux | unreliable | WS4.3 acceptance | Drop sim lane; grammar + rules only | lose one validity signal |
| Hirtreiter SFILES dialect | parse rate < 95% | WS2.1 | Compatibility shim | +2 days |
| Track A model misses target by > 10 pts | eval at M3 | `baselines.json` | Track B LoRA fine-tune | +2 weeks, GPU cost |
| Text extraction noisy | < 60% human-correct | M4 gate | Exclude Tier D from training | Phase 3 text dropped |
| Image digitization noisy | < 50% human-correct | M4 gate | Exclude; keep PID2Graph real only | Phase 3 image dropped |
| Context length 512 insufficient | > 1% of real strings truncated | M2 stats | 1024 context; or hierarchical chunking by subsystem | small |
| Single-GPU budget exceeded | training > 12 h per run | W&B | Reduce d_model; fewer synthetic epochs | quality −small |

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Real yield lands at ~600, not 1,000 | Medium | Model quality on real test drops | Hirtreiter pairs still train Task B well; Task A leans on synthetic; email TU Delft for withheld corpora (week 1) |
| License ambiguity blocks a public corpus release | Medium | Cannot publish corpus; model still usable internally | Per-record provenance; `commercial` profile is release-ready by construction |
| Synthetic data dominates and model learns generator artifacts | Medium | Poor real-world suggestions | Curriculum ends on real; ablation at M3; cap synthetic:real ratio at 50:1 in final fine-tune |
| Grammar-constrained decoding is slow | Medium | Latency target missed | Check only at unit boundaries; cache parser state; precompute allowed-next-token tables |
| Phase 3 consumes time without passing gate | High | Schedule slip | Hard cap: 3 engineer-weeks per modality before M4 |
| SFILES2 upstream changes break vendored API | Low | Build break | Pinned submodule; vocab test |
| Simulator validity false negatives discourage good topologies | Medium | Over-filtering | Treat `sim_converged` as positive-only evidence |

---

## 8. Milestones

| Milestone | Week | Gate (all must hold) |
|---|---|---|
| **M0 Foundation** | 1 | CI green; round-trip property tests pass; `fetch` works; WS0 verification items resolved and fallbacks chosen |
| **PoC-0 Vertical slice** ✅ 2026-10-11 | 2–3 | PFD→P&ID seq2seq trained on Hirtreiter data alone, evaluated on their test split. Pass: top-5 exact ≥ 80% (published 89.2%), top-1 grammar-valid ≥ 95%. **Result: 81.1% / 98.2% with 100k pairs** (`docs/poc0.md`). Track A confirmed. Public repo live. |
| **PoC-1 Output-space bake-off** ← **next** | 3–5 | Decides Task B's architecture by data efficiency, not peak accuracy. Seq2seq (string generation) vs loop-set prediction (§WS5.3 v2), same test split, same metrics (loop F1, rendered top-5), training sizes 1k / 10k / 100k. Gate: the decision rule below is applied and recorded in §10. Blocks M1 (WS1 harvest) only for the two shared-infrastructure items (loop contract layer, constrained decoding); extractor work may start in parallel in week 5. |
| **M1 First harvest** | 7 | DWSIM + Pyomo + DEXPI + PID2Graph extractors pass acceptance; ≥ 600 real records `grammar_ok`; data-quality gates wired |
| **M2 Corpus v1** | 9 | ≥ 1,000 real (or written variance explanation); Hirtreiter ingested; ≥ 50k synthetic; frozen test set committed; `commercial` profile builds with 0 unknown licenses; rules R01–R33 implemented |
| **M3 Models v1** | 11 | Tasks A/B/C trained; eval harness + `baselines.json`; Task B within 5 pts of published; ablations logged; Track B decision made |
| **M4 Copilot + Phase 3 gate** | 14 | API + CLI pass contract tests; latency targets met on CPU or GPU-only documented; WS3 pilots judged; go/no-go recorded in `BLUEPRINT.md` §10 |
| **M5 Release v1** | 18 | Success criteria table (§1) all green or waived in writing; Phase 3 data integrated if go; corpus + model cards + license summary written |

### PoC-1 protocol and decision rule

**Why this is the next proof point.** The real corpus will hold hundreds of control-annotated flowsheets (DEXPI ~14 realistic P&IDs plus whatever TU Delft releases), never 100k. The model that learns the task from 1k–10k examples is the one that will work on real data. Matching published numbers on 100k synthetic pairs is a comparability checkpoint, not the goal. PoC-0 showed the seq2seq model needs 100k to reach 81% top-5 and gets 36% at 10k.

**Shared work (week 3, both branches need it, nothing wasted):**
1. `gpd/models/loops.py`: `Loop`, `parse_loops(pid_graph)`, `render(pfd_graph, loops)`. Contract test `render(pfd, parse_loops(pid)) == canonical(pid)` over the full Hirtreiter corpus; gate ≥ 99%.
2. Loop-level precision / recall / F1 added to `gpd/eval/benchmarks.py` for **any** model (seq2seq outputs are parsed with `parse_loops`).
3. Grammar-constrained decoding for the seq2seq baseline (`gpd/models/decode.py`): mask tokens that cannot extend a parseable string, checked at unit boundaries.

**Diagnostic (week 3, one day):** oracle re-ranking on the existing 100k seq2seq model. For each test pair, can a rule/DOF check pick the correct hypothesis out of the beam-5? Report "oracle top-1". If it recovers most of the 33-point top-1 gap, the seq2seq already holds the engineering knowledge; if little, the misses are modeling errors and structured output is the fix. Either way the number is recorded.

**Bake-off (weeks 4–5):**
- Loop model per §WS5.3 v2 (graph-featured encoder, per-node heads, pointer head, DOF masks), ~5–10M params, trained at 1k and 10k (100k if cheap).
- Seq2seq baseline already has 10k and 100k results; add 1k. Optional: 25M-param seq2seq on 100k, if a GPU is available, to test whether string generation saturates.
- Same test split (`test_data_1k`), same metrics, scaling curve plotted in `docs/poc1.md`.

**Decision rule (fixed now, applied at week 5):**

| Loop model, trained on 10k, rendered top-5 on Hirtreiter test | Decision |
|---|---|
| **≥ 81%** (seq2seq needed 100k for this) | **Switch.** Loop-set prediction becomes Task B. Seq2seq kept only as the published-comparability baseline. |
| **36–81%** (between seq2seq at 10k and at 100k) | **Keep both** through M3. Run both scaling curves to 100k; choose at M3 by the slope at 1k–10k, the real-data regime. |
| **≤ 36%** (no better than seq2seq at 10k) | **Formulation or bug.** One week of debugging, hard cap. If still ≤ 36%, stay with seq2seq; invest in constrained decoding, oracle-guided re-ranking, and the 25M model. |

Secondary criteria, recorded alongside: loop F1 (primary engineering metric), parse rate (loop model expected 100%), training wall-time per 1k pairs, and the 1k-pair result for each model (the regime that matters for real data).

**Kill criterion for the scaling branch:** if a 25M-param seq2seq on 100k does not reach top-1 ≥ 60%, string generation is saturating on this task and only the structured path reaches engineer-grade top-1.

**Budget.** Two weeks, one engineer, this machine (loop model is small and needs no beam search). Cloud GPU only for the optional 25M seq2seq.

---

## 9. Task list

### Weeks 1–3: WS0 → PoC-0 — done 2026-10-11

1. ~~Scaffold repo, `pyproject.toml`, `Makefile`, CI skeleton, `ruff`/`mypy` config.~~
2. ~~Add SFILES2 submodule; write `vocab.py` from upstream mapping; vocab test.~~
3. ~~Implement `ir.py`, `record.py`, `sfiles.py`, `hashing.py`; Hypothesis strategies; round-trip + hash property tests.~~
4. ~~`sources.yaml` + `fetch.py` with checksum lock; fetch Hirtreiter.~~
5. ~~Hirtreiter ingest (WS2.1) → `PairRecord` JSONL with their splits preserved; parse-rate report.~~
6. ~~Tokenizer (WS5.1) with round-trip contract test over the whole Hirtreiter corpus.~~
7. ~~Task B seq2seq (WS5.3), trainer, beam decoder, eval harness with `baselines.json`.~~
8. ~~Train on Hirtreiter; record PoC-0 result; Track A/B decision in §10.~~
9. ~~Publish repo (MIT). CI green on first push.~~

### Weeks 3–5: PoC-1 output-space bake-off — **next**

10. `gpd/models/loops.py`: `Loop`, `parse_loops`, `render`; contract test over the Hirtreiter corpus (gate ≥ 99%).
11. Loop-level P/R/F1 in `gpd/eval/benchmarks.py`; re-score the PoC-0 100k model with it.
12. Oracle re-ranking diagnostic on the PoC-0 beam-5 outputs; record oracle top-1 in `docs/poc1.md`.
13. Grammar-constrained decoding in `gpd/models/decode.py`; re-evaluate the 100k seq2seq.
14. Loop model: graph-featured encoder, per-node `(variable, kind)` heads, pointer head, DOF masks; trainer; greedy per-node decoding.
15. Train loop model at 1k and 10k; seq2seq at 1k; same test split; scaling curve in `docs/poc1.md`.
16. Apply the PoC-1 decision rule (§8); record the decision in §10. Update WS5.3 accordingly.

### Week 5+: WS1 (may start in parallel with 14–16)

17. Resolve remaining WS0 verification items (DWSIM tags, Aspen `.bkp`).
18. WS1.1 DWSIM extractor against the 9 SFILES2 sample graphs + downloaded `.dwxmz`.
19. Send three emails: TU Delft PIR (withheld 312 / 1,005 / 100k; also ask which train size / split produced 74.8 / 89.2), IIT Bombay (SynthPID), FOSSEE (license).

---

## 10. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-10-11 | Blueprint v0.1 accepted as build basis | — |
| 2026-10-11 | SFILES2 vendored as uv workspace member at `third_party/SFILES2`, pinned to `fe0aa9c` (v1.2.0). Import name `Flowsheet_Class`. | Upstream ships a hatchling `pyproject.toml`; no shim needed. |
| 2026-10-11 | WS0 verification: `create_random_flowsheet` is a **no-op** upstream — `PID_generation` module is not in the repo (`PID_generator=False`). WS2.2 fallback (own motif generator) is now the plan. | Verified by import; `tests/unit/test_sfiles2_dep.py` guards it. |
| 2026-10-11 | "unk" class renders as SFILES2 `X`. | Keeps strings parseable by upstream; `pattern_node` accepts any class but downstream tooling expects vocab members. |
| 2026-10-11 | Canonical string uses `remove_hex_tags=True`; hot/cold side labels are not part of the dedup key. | Only option that round-trips all 9 real graphs. See §3.3. |
| 2026-10-11 | Wrapper normalizes graph insertion order before SFILES2 conversion. | Found by property test: canonical string was not invariant under edge reordering for parallel edges. |
| 2026-10-11 | `Record.sfiles_ctrl` renamed `sfiles_pfd` (control-stripped variant). | The canonical string already carries control notation; the derived field is the stripped one. |
| 2026-10-11 | Graph JSON uses an own versioned shape, not `nx.node_link_data`. | networkx 3.4 renamed `links`→`edges`; avoid stored-format churn. networkx pinned `>=3.5` for stable directed WL hashes. |
| 2026-10-11 | Hirtreiter Zenodo corpus is **synthetic** (ReadMe states it). All its records are `is_real=False`. The corpus doc's Tier A listing stands for usability, not for realness. | Affects the real-flowsheet count: Hirtreiter contributes 0 real records. The ~1,000 real target rests on WS1 alone. |
| 2026-10-11 | Hirtreiter strings: 100% parse, 83–87% canonically idempotent (older SFILES2 ordering of recycle numbers). Train/eval on their strings verbatim for comparability; store canonical forms alongside. | `PairRecord.{pfd,pid}_is_canonical`. |
| 2026-10-11 | PoC-0 model: 7.5M-param encoder-decoder (d=256, 4+4 layers), tied embeddings with std d_model^-0.5 init. | Default N(0,1) tied init gave initial loss 238 vs ln(V)=4.8. `tests/unit/test_model_init.py` guards it. |
| 2026-10-11 | **PoC-0 run 1 (10k pairs): FAIL.** top-1 16.5%, top-5 35.8%, parses 80.6% (bars 80% / 95%). Underfit: val loss still falling at epoch 30. 100k run launched; Track A/B decision deferred to it. | `docs/poc0.md`. 19% unparseable outputs motivate grammar-constrained decoding (WS5.2) as a near-term item, not a later polish. |
| 2026-10-11 | **PoC-0 run 2 (100k pairs): PASS.** top-1 42.0%, top-5 81.1%, parses 98.2%. Published 74.8 / 89.2. **Track A continues; Track B not triggered.** Milestone PoC-0 closed. | Same 7.5M model; data volume was the constraint. Next levers for top-1: larger model + longer schedule, validity-layer beam re-ranking, grammar-constrained decoding. `regression_floor` set in `baselines.json`. |
| 2026-10-11 | **PoC-1 (output-space bake-off) is the next proof point**, ahead of WS1 harvest. Decision rule fixed in §8: loop model at 10k ≥ 81% top-5 → switch; 36–81% → keep both to M3, choose by 1k–10k slope; ≤ 36% → one week debug cap, then stay with seq2seq. Kill criterion for scaling: 25M seq2seq on 100k must reach top-1 ≥ 60%. M1–M5 shift +2 weeks. | Data efficiency decides, because the real corpus will be hundreds of examples, not 100k. Shared work (loop contract layer, loop F1, constrained decoding) is needed by both branches. |
| 2026-10-11 | **Task B design revision: loop-set prediction (WS5.3 v2).** Predict control loops `(measured, variable, kind, actuated)` over the PFD graph with engineering masks (DOF, one-actuator-per-loop, inventory classes); render the string deterministically. Seq2seq stays as the comparability baseline. Target: ≥ 81% top-5 from 10k pairs. | The seq2seq model is a copier with a verifier outside it. Structured output puts validity and the engineering decision space inside the model and should transfer to real PFDs with hundreds, not 100k, examples. |
| | WS0 verification: DWSIM XML tags | pending |
| | WS0 verification: Aspen `.bkp` FLOWSHEET section | pending |
| | FOSSEE license reply | pending |
| | M4 Phase 3 go/no-go (text) | pending |
| | M4 Phase 3 go/no-go (image) | pending |
