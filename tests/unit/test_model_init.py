"""Initial loss must be near ln(vocab): guards the tied-embedding init (found in PoC-0)."""

from __future__ import annotations

import math

import torch
from torch import nn

from gpd.canonical.tokenizer import Vocab
from gpd.models.data import PairDataset, collate
from gpd.models.seq2seq import Seq2Seq, Seq2SeqConfig

PAIRS = [
    ("(raw)(hex)(prod)", "(raw)(hex)(C){TC}_1(v)<_1(prod)"),
    ("(raw)(pp)(prod)", "(raw)(pp)(C){FC}_1(v)<_1(prod)"),
    ("(raw)(tank)(prod)", "(raw)(tank)[(C){LC}_1](v)<_1(prod)"),
]


def test_initial_cross_entropy_is_near_uniform() -> None:
    torch.manual_seed(0)
    v = Vocab.build(s for p in PAIRS for s in p)
    cfg = Seq2SeqConfig(vocab_size=len(v), pad_id=v.pad_id, d_model=256, nhead=4, max_len=64)
    m = Seq2Seq(cfg).eval()
    batch = collate([PairDataset(PAIRS, v, max_len=64)[i] for i in range(3)], v.pad_id)
    with torch.no_grad():
        logits = m(batch["src"], batch["tgt_in"])
    loss = nn.CrossEntropyLoss(ignore_index=v.pad_id)(
        logits.reshape(-1, len(v)), batch["tgt_out"].reshape(-1)
    )
    uniform = math.log(len(v))
    # Tied embeddings carry the input token through the residual stream, so the untrained
    # model favours "repeat the previous token" and sits a few nats above uniform. The bug
    # this guards against (N(0,1) tied init) gave logits of std ~16 and a loss near 240.
    assert float(logits.std()) < 4.0, float(logits.std())
    assert float(loss) < uniform + 6.0, (float(loss), uniform)
    assert abs(float(m.tok_emb.weight.std()) - cfg.d_model**-0.5) < 0.01
    assert torch.equal(m.tok_emb.weight[v.pad_id], torch.zeros(cfg.d_model))
