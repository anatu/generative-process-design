"""SFILES 2.0 tokenizer (BLUEPRINT WS5.1).

Tokens are exactly SFILES2's parser tokens (``sfiles.tokenize``): one token per
unit ``(hex)``, tag ``{tout}``, recycle marker ``<1`` / ``1`` / ``_1`` / ``<_1``,
bracket, and separator. Detokenization is concatenation, so
``decode(encode(s)) == s`` for every string whose tokens are in the vocab.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from gpd.canonical.sfiles import tokenize

PAD: Final = "<pad>"
BOS: Final = "<bos>"
EOS: Final = "<eos>"
UNK: Final = "<unk>"
SPECIALS: Final = (PAD, BOS, EOS, UNK)
VOCAB_VERSION: Final = 1


class Vocab:
    def __init__(self, tokens: list[str]) -> None:
        for s in SPECIALS:
            if s in tokens:
                raise ValueError(f"special token {s!r} collides with corpus token")
        self.itos: list[str] = [*SPECIALS, *tokens]
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    @classmethod
    def build(cls, strings: Iterable[str], *, min_freq: int = 1) -> Vocab:
        counts: Counter[str] = Counter()
        for s in strings:
            counts.update(tokenize(s))
        tokens = sorted(t for t, c in counts.items() if c >= min_freq)
        return cls(tokens)

    def encode(self, s: str, *, add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokenize(s)]
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def unknown_tokens(self, s: str) -> list[str]:
        return [t for t in tokenize(s) if t not in self.stoi]

    def decode(self, ids: Iterable[int], *, stop_at_eos: bool = True) -> str:
        out: list[str] = []
        for i in ids:
            if i == self.eos_id and stop_at_eos:
                break
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            out.append(self.itos[i])
        return "".join(out)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": VOCAB_VERSION, "tokens": self.itos[len(SPECIALS) :]}, indent=0)
            + "\n"
        )

    @classmethod
    def load(cls, path: Path) -> Vocab:
        data = json.loads(path.read_text())
        if data.get("version") != VOCAB_VERSION:
            raise ValueError(f"unsupported vocab version {data.get('version')!r}")
        return cls(list(data["tokens"]))
