# PoC-0 — PFD→P&ID control-structure generation on the Hirtreiter benchmark

Date: 2026-10-11 · Milestone: BLUEPRINT §8 "PoC-0 Vertical slice"

## Question

Does the pipeline (fetch → ingest → tokenizer → seq2seq → beam decode → eval)
work end to end on real published data, and does a small from-scratch
transformer (Track A) get close to the published PFD→P&ID numbers?

## Data

- Zenodo 7658798, "Towards automatic generation of control structures for
  PFDs with AI" (Hirtreiter, Schulze Balhorn, Schweidtmann 2023), CC-BY-4.0.
- **Synthetic** pairs only; the paper's 312 real pairs are withheld.
- Splits used verbatim: `train_data_10k.json` (10,000), `eval_data_1k.json`
  (1,000, used as validation), `test_data_1k.json` (1,000). Standard variant
  (with valves). Augmented sets not used.
- All strings parse with SFILES2 v1.2.0. 83–87% equal their canonical form;
  the rest differ in recycle-number ordering from an older SFILES2 release.
  Training and scoring use the published strings verbatim.
- Token lengths (upstream parser tokens): PFD max 145, PID max 267, p99 254.
  `max_len=320`. Vocabulary from the 10k train set: 118 tokens incl. 4 specials.

## Model and training

| | |
|---|---|
| Architecture | Encoder-decoder transformer, pre-norm, learned positions, tied in/out embeddings |
| Size | d_model 256, 4 heads, 4+4 layers, FFN 1024 — 7.49M parameters |
| Optimizer | AdamW (β 0.9/0.98, wd 0.01), lr 5e-4, 500 warmup steps, cosine to 10% |
| Regularization | dropout 0.1, label smoothing 0.1, grad clip 1.0 |
| Batching | 64 pairs, length-bucketed |
| Schedule | up to 30 epochs, early stop patience 6 on greedy val exact match (500 samples) |
| Hardware | Apple M4 Pro, MPS backend |
| Command | `make poc0` (fetch → extract → train → eval) |

One bug found and fixed on the way: with tied embeddings and PyTorch's default
N(0,1) embedding init, initial logits had std ≈ 16 and loss ≈ 238. Init std
d_model^-0.5 brings the initial loss to ≈ ln(V). Guarded by
`tests/unit/test_model_init.py`.

## Result — run 1: 10k training pairs

**FAIL** on both pass bars.

| Metric (beam 5, test 1k) | Ours (10k) | Published | Pass bar |
|---|---|---|---|
| top-1 exact match | 16.5% | 37.7% (10k) | — |
| top-5 exact match | 35.8% | 74.8% (10k) | ≥ 80% |
| top-1 parses (G1) | 80.6% | — | ≥ 95% |
| top-1 is canonical (G2) | 65.3% | — | — |
| top-1 canonical-equivalent to target | 18.0% | — | — |

Training: 30 epochs (4,710 steps), best greedy val exact 16.8% at epoch 28,
38 min wall time. Validation loss was still decreasing at the end (0.843) —
the model is underfit, not overfit. Beam-5 evaluation of 1,000 pairs took 8.8 min.

Observations:
- 19% of top-1 outputs do not parse. Typical failure: a signal-recycle opener
  `<_n` without its matching `_n`. This is the case for grammar-constrained
  decoding (BLUEPRINT §WS5.2) — the grammar is small and checkable at every
  unit boundary.
- Canonical-equivalent match (18.0%) barely exceeds exact match (16.5%): the
  misses are structural, not formatting.

## Result — run 2: 100k training pairs

**PASS** on both bars. Same 7.49M-parameter model, 100,000 pairs, 12 epochs
(18,756 steps), warmup 1,000, patience 3. 64 min training, 9 min beam-5 eval.

| Metric (beam 5, test 1k) | 10k run | **100k run** | Published | Pass bar |
|---|---|---|---|---|
| top-1 exact match | 16.5% | **42.0%** | 48.6% (100k) | — |
| top-5 exact match | 35.8% | **81.1%** | 89.2% (100k) | ≥ 80% |
| top-1 parses (G1) | 80.6% | **98.2%** | — | ≥ 95% |
| top-1 is canonical (G2) | 65.3% | 78.1% | — | — |
| top-1 canonical-equivalent to target | 18.0% | 45.7% | — | — |

Validation trajectory (greedy exact on 500 pairs): 1.0 → 9.6 → 17.0 → 20.0 →
25.0 → 32.2 → 39.0 → 42.0 → 44.0 → 45.0 → 47.0 → 47.0% over epochs 0–11.
Val loss 0.945 → 0.808, still edging down at the end.

Observations:
- 10× data moved top-5 from 35.8% to 81.1% with the model held fixed. Data
  volume, not architecture, was the binding constraint at 10k.
- Against the published 100k figures (48.6% top-1, 89.2% top-5) the gaps are 6.6
  and 8.1 points. Against the published 10k figures (37.7 / 74.8) our 10k run is
  far behind, which marks our 10k baseline as under-trained, not the architecture
  as weak. The model ranks the right answer in its beam far more often than it
  ranks it first; the published model shows the same 40-point spread, which the
  Graph-to-SFILES authors attribute to the one-to-many mapping from a PFD to its
  valid control structures. Beam re-ranking by the validity layer is a cheap next
  step.
- Unparseable outputs fell from 19% to 1.8%. Grammar-constrained decoding
  would close the rest.
- Training was still improving at epoch 12 with the learning rate at its
  floor. More epochs and the originally planned larger model (d=384, 6+6
  layers, ~25M params) are the obvious levers for top-1.

## Decision

**Track A continues. Track B (LLM fine-tune) is not triggered.** The pass bars
are met; the top-5 gap to published is under 10 points. Next steps for this
task, in order: (1) larger model + longer schedule on 100k, (2) validity-layer
beam re-ranking, (3) grammar-constrained decoding. These run alongside WS1,
not instead of it.

## Caveats

- Published figures, verified against the papers on 2026-10-11: Hirtreiter et
  al. report top-5 74.8% at 10k and 89.2% at 100k training pairs; Graph-to-SFILES
  reports the matching top-1 figures, 37.7% and 48.6%, and its own graph-encoder
  results of 15.2 / 28.4 at 1k and 41.9 / 73.2 at 10k. All on the synthetic test
  split of 1,000 pairs. An earlier version of this file read 74.8 / 89.2 as
  top-1 / top-5. See `docs/lit_review_structured_generation.md` §2.2.
- Exact match is a strict metric on non-canonical strings. The
  canonical-equivalent column reports matches after re-canonicalizing both
  sides; it is the more meaningful number for downstream use.
- No real flowsheets were involved. Transfer to real PFDs is untested until
  WS1 delivers the harvest.
