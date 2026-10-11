"""Greedy and batched beam-search decoding (BLUEPRINT WS5.3). Grammar-constrained decoding comes with WS5.2."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from gpd.canonical.tokenizer import Vocab
from gpd.models.data import pad_batch
from gpd.models.seq2seq import Seq2Seq


@torch.no_grad()
def greedy(
    model: Seq2Seq,
    vocab: Vocab,
    srcs: Sequence[str],
    *,
    device: torch.device,
    max_len: int | None = None,
    batch_size: int = 64,
) -> list[str]:
    model.eval()
    max_len = max_len or model.cfg.max_len
    out: list[str] = []
    for start in range(0, len(srcs), batch_size):
        chunk = srcs[start : start + batch_size]
        src = pad_batch([vocab.encode(s) for s in chunk], vocab.pad_id).to(device)
        memory, src_pad = model.encode(src)
        seqs = torch.full((len(chunk), 1), vocab.bos_id, dtype=torch.long, device=device)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits = model.decode(seqs, memory, src_pad)[:, -1]
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, vocab.pad_id), nxt)
            seqs = torch.cat([seqs, nxt.unsqueeze(1)], dim=1)
            done |= nxt.eq(vocab.eos_id)
            if bool(done.all()):
                break
        out += [vocab.decode(row.tolist()) for row in seqs]
    return out


@torch.no_grad()
def beam_search(
    model: Seq2Seq,
    vocab: Vocab,
    srcs: Sequence[str],
    *,
    device: torch.device,
    beam: int = 5,
    max_len: int | None = None,
    batch_size: int = 32,
    length_alpha: float = 0.0,
) -> list[list[tuple[str, float]]]:
    """Return, per source, up to ``beam`` (string, score) hypotheses sorted best first.

    Score is summed log-probability divided by ``len ** length_alpha``.
    """
    model.eval()
    max_len = max_len or model.cfg.max_len
    k = beam
    results: list[list[tuple[str, float]]] = []
    neg_inf = float("-inf")

    for start in range(0, len(srcs), batch_size):
        chunk = srcs[start : start + batch_size]
        b = len(chunk)
        src = pad_batch([vocab.encode(s) for s in chunk], vocab.pad_id).to(device)
        memory, src_pad = model.encode(src)
        # Expand to (b*k, ...)
        memory = (
            memory.unsqueeze(1).expand(-1, k, -1, -1).reshape(b * k, memory.size(1), memory.size(2))
        )
        src_pad = src_pad.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)

        seqs = torch.full((b * k, 1), vocab.bos_id, dtype=torch.long, device=device)
        scores = torch.full((b, k), neg_inf, device=device)
        scores[:, 0] = 0.0  # only the first beam is alive at step 0
        finished = torch.zeros(b * k, dtype=torch.bool, device=device)
        lengths = torch.ones(b * k, dtype=torch.long, device=device)
        base = (torch.arange(b, device=device) * k).unsqueeze(1)

        for _ in range(max_len - 1):
            logits = model.decode(seqs, memory, src_pad)[:, -1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            # Finished hypotheses only extend with PAD at zero cost.
            logp[finished] = neg_inf
            logp[finished, vocab.pad_id] = 0.0
            vocab_size = logp.size(-1)
            cand = (scores.reshape(b * k, 1) + logp).reshape(b, k * vocab_size)
            top_scores, top_idx = cand.topk(k, dim=-1)
            beam_idx = top_idx // vocab_size
            tok_idx = top_idx % vocab_size
            sel = (base + beam_idx).reshape(-1)
            seqs = torch.cat([seqs[sel], tok_idx.reshape(-1, 1)], dim=1)
            finished = finished[sel] | tok_idx.reshape(-1).eq(vocab.eos_id)
            lengths = lengths[sel] + (~finished[sel] | tok_idx.reshape(-1).eq(vocab.eos_id)).long()
            scores = top_scores
            if bool(finished.all()):
                break

        seqs = seqs.reshape(b, k, -1)
        scores_cpu: Tensor = scores.detach().cpu()
        for i in range(b):
            hyps: list[tuple[str, float]] = []
            for j in range(k):
                sc = float(scores_cpu[i, j])
                if sc == neg_inf:
                    continue
                text = vocab.decode(seqs[i, j].tolist())
                norm = max(1, len(vocab.encode(text, add_special=False)) + 1) ** length_alpha
                hyps.append((text, sc / norm))
            hyps.sort(key=lambda h: h[1], reverse=True)
            results.append(hyps)
    return results
