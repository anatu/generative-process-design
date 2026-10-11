# Generative Process Design

Corpus, validity layer, and copilot models for chemical process flowsheets:
flowsheet autocompletion, PFD→P&ID control-structure generation, and topology
error correction. Everything is built on the SFILES 2.0 string ⇄ graph
representation from TU Delft's [SFILES2](https://github.com/process-intelligence-research/SFILES2).

- **Plan and rationale:** [`BLUEPRINT.md`](BLUEPRINT.md) — architecture, workstreams, tests, fallbacks, milestones.
- **Source inventory:** [`generative-process-design-corpus.md`](generative-process-design-corpus.md) — 72 verified public data sources and why the public ceiling is ~10³ real flowsheets.
- **PoC-0 result:** [`docs/poc0.md`](docs/poc0.md) — PFD→P&ID on the Hirtreiter synthetic benchmark.

## Status

2026-10-11. Foundation (`gpd/canonical`), Hirtreiter ingest, tokenizer, seq2seq
model, beam decoder, and eval harness are in and tested end to end.

**PoC-0 passed.** A 7.5M-parameter encoder-decoder trained on the 100k
synthetic Hirtreiter pairs reaches top-5 exact 81.1% and 98.2% parseable
outputs on their test split (bars 80% / 95%; published 89.2% top-5, 74.8%
top-1; ours 42.0% top-1). Details and next levers in
[`docs/poc0.md`](docs/poc0.md). Next: WS1 simulator harvest (DWSIM, Aspen,
Pyomo, DEXPI) toward ~1,000 real flowsheets.

## Quick start

```sh
make install                      # uv sync, Python 3.11
make check                        # lint + mypy + import contracts + tests (<1 min)

uv sync --extra models            # PyTorch
make poc0                         # fetch Hirtreiter → JSONL → train → beam-5 eval
```

`make help` lists every target.

## Layout

```
gpd/canonical/     IR schema, typemaps, IR→graph, SFILES2 wrapper, hashing, tokenizer, vocab
gpd/corpus/        Record / PairRecord schemas
gpd/models/        seq2seq transformer, trainer, greedy + beam decoding
gpd/eval/          benchmarks + baselines.json regression gate
extractors/        isolated source parsers → IR JSON (only hirtreiter so far)
scripts/           fetch.py + sources.yaml (checksummed downloads)
tests/             unit / property (Hypothesis) / integration
third_party/SFILES2  pinned submodule (MIT)
```

Design rules that matter (see BLUEPRINT §2–3):

- Every extractor runs in its own environment and emits a common IR. The core
  package never imports IDAES, WaterTAP, BioSTEAM, or pyDEXPI (enforced by
  import-linter).
- The canonical SFILES 2.0 string is the dedup key. The wrapper normalizes
  graph insertion order before conversion because SFILES2 breaks ties by
  iteration order.
- Every record carries provenance and license. Two build profiles:
  `research` (everything) and `commercial` (no NC, no unconfirmed licenses).

## Data

Nothing under `data/` is committed. `scripts/sources.yaml` lists every source
with URL, license, and checksum; `scripts/sources.lock.json` records what was
fetched. The Hirtreiter pairs are **synthetic** (CC-BY-4.0); the 312 real pairs
from the paper are not public.

## License

MIT. Third-party material is listed at the bottom of [`LICENSE`](LICENSE).
