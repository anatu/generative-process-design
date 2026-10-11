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
| top-1 exact match | 16.5% | 74.8% | — |
| top-5 exact match | 35.8% | 89.2% | ≥ 80% |
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

_In progress. Same model, 100,000 pairs, 12 epochs._ Rationale: the paper most
likely trained on the 100k set; the 10k run was data-limited.

## Decision

Deferred to the 100k result. Rule (BLUEPRINT §WS5.5): if Track A remains > 10
points under the published numbers with the full 100k set, start Track B
(LoRA fine-tune of a ≤1.5B open LLM) in parallel with WS1.

## Caveats

- The published 74.8 / 89.2 figures are quoted from the corpus document; the
  training-set size and exact test split behind them need confirming against
  the paper before any claim of parity.
- Exact match is a strict metric on non-canonical strings. The
  canonical-equivalent column reports matches after re-canonicalizing both
  sides; it is the more meaningful number for downstream use.
- No real flowsheets were involved. Transfer to real PFDs is untested until
  WS1 delivers the harvest.
