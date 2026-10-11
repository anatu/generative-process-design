# Generative Process Design

**Models that help chemical engineers design plants — starting with the part
that is pure pattern work.**

A chemical process is drawn as a flowsheet: boxes for equipment, arrows for
streams. Turning that drawing into something that can be built and run means
adding hundreds of routine decisions — which temperatures to control, which
valves to move, where to put a level controller. This repo builds the data,
the validity checks, and the models to make those decisions suggestible,
checkable, and correctable. It is a copilot, not an autonomous designer; the
section on data explains why.

- Plan, workstreams, tests, fallbacks: [`BLUEPRINT.md`](BLUEPRINT.md)
- Source inventory (72 verified public datasets): [`generative-process-design-corpus.md`](generative-process-design-corpus.md)
- First result: [`docs/poc0.md`](docs/poc0.md)

---

## 1. The problem in one picture

![PFD to P&ID example](docs/img/pfd_to_pid_example.png)

**Top — a PFD (Process Flow Diagram).** Two feeds. One is pumped (`pp-1`)
through a valve into a reactor (`r-1`); the other enters the reactor directly.
The reactor has a top outlet (`tout`) and a bottom outlet (`bout`), each through
a valve to a product. This is what a process engineer hands over: *what* the
plant does.

**Bottom — the P&ID (Piping and Instrumentation Diagram) for the same plant.**
Same equipment, same streams, plus the control layer in orange: a motor (`M`)
and pressure indicator (`PI`) on the pump, two flow controllers (`FC`) on the
feed, a temperature indicator (`TI`) and level controller (`LC`) on the
reactor — the level controller drives the bottoms valve — and a pressure
controller (`PC`) driving the top-product valve. Dashed orange lines are
control signals: *which measurement moves which valve*. This is what the plant
is built and operated from.

Going from top to bottom is the task. In this example the model's output is
identical to the ground truth.

The decisions involved are not creative. They follow conventions that every
process-control engineer applies by hand on every project: inventories (levels,
pressures) must be controlled; each manipulated valve buys one controlled
variable; a reactor gets temperature and level; a compressor gets a motor and
pressure control. Repetitive, consequential, and encoded nowhere except in
engineers' heads and in finished drawings.

## 2. Why this matters, with numbers

**It is the bottleneck step, and it is manual.** Every plant, every
revamp, every debottlenecking study produces a P&ID set. Control-structure
design is done by hand from the PFD, reviewed by hand (HAZOP), and redrawn by
hand when the process changes. There is no autocomplete.

**The published results say it is learnable.**
Hirtreiter, Schulze Balhorn & Schweidtmann (2023) trained a transformer to add
control structures to PFDs and reached **74.8% top-1 / 89.2% top-5 exact match**
on held-out synthetic flowsheets. Graph-to-SFILES work reports 73.2% top-5 on
related topology tasks. Flowsheet autocompletion (Vogel et al. 2023) works the
same way, like a language model over unit operations.

**The data is the constraint, not the modelling.** From the source inventory:

| Fact | Number |
|---|---|
| Public sources verified for this project | 72 (14 direct, 27 convertible, 31 weak) |
| Real flowsheets assemblable from all of them, after dedup | ~1,000–1,200 |
| Training molecules behind molecular generative models (the analogy this field leans on) | ~10⁸ |
| Largest real corpora ever assembled — all unreleased for copyright | Theisen 1,005 PFDs; Hirtreiter 312 P&IDs; Vogel 223 flowsheets |
| Public Hirtreiter training pairs | 100,000 — **all synthetic** |
| Digitizing drawings back into graphs (PID2Graph) | ~75% edge accuracy; needs repair and human checks |
| Public sources carrying operating parameters, stream conditions, or economics at scale | none |

Real public topologies number ~10³. Molecular generation had ~10⁵× more. That
gap sets the ceiling: syntactically valid flowsheets, autocompletion, control
structures, and error correction are achievable at the published quality;
open-ended, economically sound novel process synthesis is not. Hence a copilot.

**What this repo measured.** Same 7.5M-parameter model, same test split, only
the training-set size changed:

| Training pairs | top-1 exact | top-5 exact | outputs that parse |
|---|---|---|---|
| 10,000 | 16.5% | 35.8% | 80.6% |
| 100,000 | 42.0% | **81.1%** | **98.2%** |

Ten times the data moved top-5 by 45 points with the model held fixed. Data
volume, not architecture, was binding — and the real corpus will be hundreds of
examples, never 100k. That tension drives the next milestone (§6).

## 3. How a flowsheet becomes text: SFILES 2.0

Every model here reads and writes **SFILES 2.0** (Simplified Flowsheet Input
Line Entry System), a one-line notation for a flowsheet graph modeled on
SMILES for molecules, from TU Delft's Process Intelligence group. The whole
grammar:

| Token | Meaning |
|---|---|
| `(hex)` `(dist)` `(r)` `(pp)` `(v)` `(mix)` `(splt)` … | a unit operation; ~30 classes, `raw`/`prod` for feeds and products |
| adjacent units | a stream from one to the next |
| `[ … ]` | a branch: a second outlet |
| `<1` … `1` | a recycle: the stream leaves at `1` and re-enters at `<1` |
| `<&| … &|` | a second feed stream joining a unit |
| `n|` | start of a separate sub-flowsheet (heat integration) |
| `{tout}` `{bout}` | stream tags: column top / bottom outlet |
| `{1}` on two `(hex)` | the same physical heat exchanger, two sides |
| `(C){TC}` `_1` `<_1` | SFILES 2.0: a controller (here temperature) and the signal line to what it actuates |

The example above, as text:

```
PFD:   (raw)(pp)(v)(r)<&|(raw)&|[{bout}(v)(prod)]{tout}(v)(prod)

P&ID:  (raw)(pp)[(C){M}<_1](C){PI}(C){FC}_1(C){FC}_2(v)<_2(r)<&|(raw)&|
       [(C){TI}][(C){LC}_3][{bout}(v)<_3(prod)]{tout}(C){PC}_4(v)<_4(prod)
```

Read the P&ID: feed → pump with a motor branch `[(C){M}<_1]` → pressure
indicator → flow controller `_1` (its signal drives the motor) → flow
controller `_2` → valve `<_2` (driven by `_2`) → reactor, second feed joins
`<&|(raw)&|`; on the reactor a temperature indicator and a level controller
`_3`; bottoms valve `<_3` driven by the level; top outlet through a pressure
controller `_4` driving valve `<_4`.

Why this representation:
- **Lossless and canonical.** A graph becomes one string; the string parses back
  to an isomorphic graph. The canonical string is the deduplication key.
- **Sequence models apply directly.** PFD→P&ID is string-to-string. The
  vocabulary is ~120 tokens.
- **It is the field's shared format.** Every public corpus and generator in the
  inventory speaks it.

Things this codebase had to fix around the reference implementation: it
breaks canonicalization ties by iteration order (we sort edges before
conversion), it parses malformed strings without complaint (our grammar check
is canonical idempotence), and it drops hot/cold side labels on heat exchangers
in the canonical form.

## 4. What is built

```
 sources (DWSIM, Aspen, Pyomo, DEXPI XML, GraphML, Modelica, Hirtreiter JSON)
        │   one isolated extractor per source; core never imports their stacks
        ▼
 IR JSON ──► canonical graph ──► SFILES 2.0 string ──► canonical hash (dedup)
                 │                     │
                 │                     ├──► tokenizer ──► seq2seq model ──► beam search
                 │                     │                                     │
                 ▼                     ▼                                     ▼
          validity layer: grammar (G1/G2) · degree rules · 33 P&ID rules · [simulator]
                                       │
                                       ▼
                       corpus: provenance + license on every record, two build profiles
```

| Package | What it does | Status |
|---|---|---|
| `gpd/canonical` | IR schema, per-source typemaps, IR→graph, SFILES2 wrapper (deterministic), canonical + Weisfeiler-Lehman hashes, graph JSON, tokenizer, frozen vocab with degree constraints | done, 90%+ coverage, mypy strict |
| `gpd/corpus` | `Record` / `PairRecord` schemas with provenance and license | done |
| `gpd/models` | encoder-decoder transformer, trainer, greedy + batched beam search | done (PoC-0) |
| `gpd/eval` | benchmarks, `baselines.json` regression floor | done (PoC-0) |
| `gpd/viz` | renders SFILES strings to drawings (the figures on this page) | done |
| `extractors/hirtreiter` | Zenodo PFD→P&ID pairs → JSONL, upstream splits preserved | done |
| `scripts/fetch.py` | checksummed downloads, lock file, zip-slip safe | done |
| `gpd/validity` rules, simulator lane | the 33 engineering rules; DWSIM round-trip | WS4, not started |
| other extractors (DWSIM, Aspen, Pyomo, DEXPI, PID2Graph, Modelica) | real-flowsheet harvest toward ~1,000 | WS1, not started |

Design rules that matter (BLUEPRINT §2–3):
- Every extractor runs in its own environment and emits a common IR. The core
  package never imports IDAES, WaterTAP, BioSTEAM, or pyDEXPI (AGPL). Enforced
  by `import-linter`.
- Every record carries provenance and license. Two build profiles: `research`
  (everything) and `commercial` (no non-commercial licenses, no unconfirmed
  ones). The commercial build fails hard on an unknown license.
- Property tests (Hypothesis) generate random valid flowsheets and check
  graph→string→graph round trips, canonical invariance under relabeling and
  edge reordering, and hash invariance. They found a real canonicalization
  instability on day one.

## 5. Results so far: PoC-0

Task: PFD → P&ID on the Hirtreiter benchmark (Zenodo 7658798, CC-BY-4.0,
synthetic). 7.49M-parameter encoder-decoder transformer, beam 5, evaluated on
the authors' 1,000-pair test split. Trained on an Apple M4 Pro in 64 minutes.

| Metric | 10k pairs | **100k pairs** | Published | Pass bar |
|---|---|---|---|---|
| top-1 exact match | 16.5% | **42.0%** | 74.8% | — |
| top-5 exact match | 35.8% | **81.1%** | 89.2% | ≥ 80% ✓ |
| top-1 output parses | 80.6% | **98.2%** | — | ≥ 95% ✓ |
| top-1 canonical-equivalent to target | 18.0% | 45.7% | — | — |

Exact match is strict: one wrong tag, recycle number, or branch order is a
miss. Full write-up with training curves and observations: [`docs/poc0.md`](docs/poc0.md).

### A near miss, and what it says about the model

![near miss](docs/img/near_miss.png)

Ground truth (top) vs the model's first choice (bottom). The plant: two feeds
mixed, compressed, reacted; bottoms to product; tops through a tank to
product. Everything matches except one decision on the reactor: the truth has
a temperature **indicator** (`TI`, measure only); the model put a temperature
**controller** (`TC`) with a signal line back to the reactor. Every signal
number downstream shifts by one, so the whole string counts as wrong. Both are
defensible engineering; the benchmark's generator chose the indicator.

The model's second-ranked hypothesis was the exact ground truth. Across the
test set that is the pattern: the right answer is in the top 5 for 81% of
cases but ranked first for only 42%. The model knows the candidates; it does
not reliably know which engineering convention applies. That is the gap the
next milestone attacks.

## 6. What is next: PoC-1, the output-space bake-off

The string model regenerates the entire PFD character by character and
sprinkles controllers in. ~60% of each output is copying. Every error is a
syntax error before it is an engineering error, and the validity layer sits
outside the generator.

The alternative, specified in BLUEPRINT §WS5.3 v2: predict the P&ID as a
**set of control loops over the PFD graph** — `(measured unit, variable
F/T/P/L, controller or indicator, actuated valve)` — with engineering invariants
(one valve per loop, degrees-of-freedom balance, levels only on
inventory-holding units) as hard masks, and render the string
deterministically. Topology is copied by construction; the decision space is
~12 choices per unit instead of 118 tokens × 130 steps.

The two are judged by **data efficiency at 1k–10k pairs**, the regime real
data will be in. The decision rule is fixed in advance (BLUEPRINT §8): the loop
model, trained on 10k, must reach the 81% top-5 that the string model needed
100k for — then it becomes Task B. Between 36% and 81%, both continue and the
scaling slope decides. Below 36%, one week of debugging and then back to the
string model.

After that: the WS1 simulator harvest (DWSIM, Aspen, Pyomo, DEXPI) toward
~1,000 real flowsheets, the 33-rule validity layer, and the copilot API.

## 7. Quick start

```sh
make install                      # uv sync, Python 3.11; SFILES2 is a pinned submodule
make check                        # ruff + mypy + import contracts + 150 tests (<1 min)

uv sync --extra models            # PyTorch (CPU wheel on Linux, MPS on macOS)
make poc0                         # fetch Hirtreiter → JSONL → train → beam-5 eval

# draw any flowsheet
uv run python -m gpd.viz --pfd "(raw)(hex)(dist)[{tout}(prod)]{bout}(prod)" --out fs.png
```

`make help` lists every target. CI runs the same `make check` on every push.

## 8. Layout

```
gpd/canonical/     IR schema, typemaps, IR→graph, SFILES2 wrapper, hashing, tokenizer, vocab
gpd/corpus/        Record / PairRecord schemas
gpd/models/        seq2seq transformer, trainer, greedy + beam decoding
gpd/eval/          benchmarks + baselines.json regression gate
gpd/viz.py         flowsheet renderer
extractors/        isolated source parsers → IR JSON (hirtreiter so far)
scripts/           fetch.py + sources.yaml (checksummed downloads, lock file)
docs/              poc0.md, figures
tests/             unit / property (Hypothesis) / integration
third_party/SFILES2  pinned submodule (MIT)
```

## 9. Data and licenses

Nothing under `data/` is committed. `scripts/sources.yaml` lists every source
with URL, license, size, and checksum; `scripts/sources.lock.json` records what
was fetched and its hash. The Hirtreiter pairs are **synthetic** (CC-BY-4.0);
the 312 real pairs from the paper are not public. Code is MIT; third-party
material is listed at the bottom of [`LICENSE`](LICENSE).

## 10. References

- Hirtreiter, Schulze Balhorn, Schweidtmann. *Towards automatic generation of
  control structures for process flow diagrams with artificial intelligence.*
  2023. [arXiv:2211.05583](https://doi.org/10.48550/arXiv.2211.05583). Data:
  [Zenodo 7658798](https://zenodo.org/records/7658798).
- Vogel, Schulze Balhorn, Hirtreiter, Schweidtmann. *SFILES 2.0: an extended
  text-based flowsheet representation.* Code:
  [process-intelligence-research/SFILES2](https://github.com/process-intelligence-research/SFILES2).
- Vogel, Schulze Balhorn, Schweidtmann. *Learning from flowsheets: a generative
  transformer model for autocompletion of flowsheets.* Computers & Chemical
  Engineering, 2023.
- Rule-based P&ID autocorrection (33 rules): [arXiv:2502.18493](https://arxiv.org/abs/2502.18493).
- PID2Graph (DLR): [Zenodo 14803338](https://zenodo.org/records/14803338).
- Full inventory with licenses and verification status:
  [`generative-process-design-corpus.md`](generative-process-design-corpus.md).
