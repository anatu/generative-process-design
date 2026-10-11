"""Model stack smoke tests: shapes, decode invariants, and a tiny end-to-end training run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gpd.canonical.tokenizer import Vocab
from gpd.corpus.pairs import PairRecord, write_jsonl
from gpd.eval.benchmarks import compare, load_baselines, score_pfd2pid
from gpd.models import train as train_mod
from gpd.models.data import PairDataset, bucket_batches, collate, pad_batch
from gpd.models.decode import beam_search, greedy
from gpd.models.seq2seq import Seq2Seq, Seq2SeqConfig

PAIRS = [
    ("(raw)(hex)(prod)", "(raw)(hex)(C){TC}_1(v)<_1(prod)"),
    ("(raw)(pp)(prod)", "(raw)(pp)(C){FC}_1(v)<_1(prod)"),
    ("(raw)(v)(prod)", "(raw)(C){FC}_1(v)<_1(prod)"),
    ("(raw)(hex)(pp)(prod)", "(raw)(hex)(C){TC}_1(pp)(C){FC}_2(v)<_1<_2(prod)"),
    ("(raw)(tank)(prod)", "(raw)(tank)[(C){LC}_1](v)<_1(prod)"),
    ("(raw)(comp)(prod)", "(raw)(comp)(C){PC}_1(v)<_1(prod)"),
]
CPU = torch.device("cpu")


def _vocab() -> Vocab:
    return Vocab.build(s for p in PAIRS for s in p)


def test_pad_and_collate_shapes() -> None:
    v = _vocab()
    ds = PairDataset(PAIRS, v, max_len=64)
    assert len(ds) == len(PAIRS) and ds.dropped == 0
    batch = collate([ds[i] for i in range(3)], v.pad_id)
    assert batch["src"].shape[0] == 3
    assert batch["tgt_in"].shape == batch["tgt_out"].shape
    assert torch.equal(batch["tgt_in"][:, 1:], batch["tgt_out"][:, :-1])
    assert pad_batch([[1, 2], [3]], 0).tolist() == [[1, 2], [3, 0]]


def test_dataset_drops_overlong_pairs() -> None:
    ds = PairDataset(PAIRS, _vocab(), max_len=6)
    assert ds.dropped == len(PAIRS) - len(ds) > 0


def test_bucket_batches_cover_everything_once() -> None:
    ds = PairDataset(PAIRS, _vocab(), max_len=64)
    seen = [t for b in bucket_batches(ds, batch_size=2, shuffle=True, seed=1) for _, t in b]
    assert sorted(map(tuple, seen)) == sorted(tuple(t) for _, t in ds.items)


def test_model_forward_shape_and_tied_output() -> None:
    v = _vocab()
    cfg = Seq2SeqConfig(
        vocab_size=len(v),
        pad_id=v.pad_id,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        max_len=32,
    )
    m = Seq2Seq(cfg)
    batch = collate([PairDataset(PAIRS, v, max_len=32)[0]], v.pad_id)
    logits = m(batch["src"], batch["tgt_in"])
    assert logits.shape == (1, batch["tgt_in"].shape[1], len(v))
    assert m.tok_emb.weight.shape == (len(v), 32)


def test_untrained_decoders_return_well_formed_output() -> None:
    v = _vocab()
    cfg = Seq2SeqConfig(
        vocab_size=len(v),
        pad_id=v.pad_id,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        max_len=16,
    )
    m = Seq2Seq(cfg).eval()
    srcs = [s for s, _ in PAIRS[:3]]
    g = greedy(m, v, srcs, device=CPU, max_len=16)
    assert len(g) == 3 and all(isinstance(x, str) for x in g)
    b = beam_search(m, v, srcs, device=CPU, beam=3, max_len=16)
    assert len(b) == 3
    for hyps in b:
        assert 1 <= len(hyps) <= 3
        scores = [sc for _, sc in hyps]
        assert scores == sorted(scores, reverse=True)
        assert all(sc <= 0.0 for sc in scores)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    v = _vocab()
    cfg = Seq2SeqConfig(
        vocab_size=len(v),
        pad_id=v.pad_id,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        max_len=16,
    )
    m = Seq2Seq(cfg)
    m.save(tmp_path / "m.pt", extra={"epoch": 3})
    m2, extra = Seq2Seq.load(tmp_path / "m.pt")
    assert extra == {"epoch": 3} and m2.cfg == cfg
    for a, b in zip(m.state_dict().values(), m2.state_dict().values(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.slow
def test_tiny_training_run_learns_copy_like_task(tmp_path: Path) -> None:
    """End-to-end: jsonl -> train -> best.pt -> beam eval -> metrics. Must fit in CI time."""
    recs = []
    for i, (p, q) in enumerate(PAIRS * 8):
        recs.append(
            PairRecord(
                id=f"t:{i}",
                source="t",
                license="MIT",
                is_real=False,
                split="train",
                pfd=p,
                pid=q,
                pfd_parses=True,
                pid_parses=True,
                pfd_is_canonical=True,
                pid_is_canonical=True,
                pfd_tokens=1,
                pid_tokens=1,
            )
        )
    write_jsonl(recs, tmp_path / "train.jsonl")
    write_jsonl(recs[: len(PAIRS)], tmp_path / "val.jsonl")
    cfg = train_mod.TrainConfig(
        train=tmp_path / "train.jsonl", val=tmp_path / "val.jsonl", out=tmp_path / "run",
        epochs=40, batch_size=8, lr=2e-3, warmup_steps=20, d_model=64, nhead=4, layers=2, ff=128,
        dropout=0.0, max_len=48, max_val=6, patience=100, device="cpu", log_every=1000,
    )  # fmt: skip
    summary = train_mod.run(cfg)
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "vocab.json").exists()
    assert summary["best_val_exact_greedy"] >= 0.5, summary["history"][-3:]

    model, _ = Seq2Seq.load(tmp_path / "run" / "best.pt")
    vocab = Vocab.load(tmp_path / "run" / "vocab.json")
    hyps = beam_search(model, vocab, [p for p, _ in PAIRS], device=CPU, beam=3)
    metrics = score_pfd2pid(hyps, [q for _, q in PAIRS])
    assert metrics["n"] == len(PAIRS) and metrics["top5_exact"] >= metrics["top1_exact"] >= 0.5
    assert 0.0 <= metrics["top1_parses"] <= 1.0
    lines = compare(metrics, load_baselines(), "pfd2pid")
    assert any(line.startswith(("PASS", "FAIL")) for line in lines)


def test_baselines_file_shape() -> None:
    b = load_baselines()
    assert set(b["pfd2pid"]["pass"]) == {"top5_exact", "top1_parses"}
    assert b["pfd2pid"]["published"]["metrics"]["top5_exact"] == 0.892
    json.dumps(b)
