"""Thin wrapper around SentencePiece for use in the MT training loop.

Mimics the small slice of the HF tokenizer interface we actually need:
encode/decode, pad/bos/eos ids, vocab size, and batch encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import sentencepiece as spm
import torch


@dataclass
class BatchEncoding:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class SPMTokenizer:
    """Joint en+eo SentencePiece tokenizer with direction prefix tokens.

    Convention: source-side sentences are prefixed with the *target* language
    tag, so a single shared model can be steered: ``<eo> Hello world``.
    """

    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_id = self.sp.pad_id()  # 0
        self.unk_id = self.sp.unk_id()  # 1
        self.bos_id = self.sp.bos_id()  # 2
        self.eos_id = self.sp.eos_id()  # 3
        self.en_id = self.sp.piece_to_id("<en>")
        self.eo_id = self.sp.piece_to_id("<eo>")
        assert self.en_id > 0 and self.eo_id > 0, "tokenizer missing <en>/<eo>"

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def lang_id(self, lang: str) -> int:
        return {"en": self.en_id, "eo": self.eo_id}[lang]

    def encode(self, text: str, lang: str | None = None, add_eos: bool = True) -> list[int]:
        ids = self.sp.encode(text)
        if lang is not None:
            ids = [self.lang_id(lang)] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        ids = [i for i in ids if i not in {self.pad_id, self.bos_id, self.eos_id, self.en_id, self.eo_id}]
        return self.sp.decode(ids)

    def batch_encode(
        self,
        texts: Iterable[str],
        lang: str | None,
        add_eos: bool,
        max_length: int,
    ) -> list[list[int]]:
        out = []
        for t in texts:
            ids = self.encode(t, lang=lang, add_eos=add_eos)
            if len(ids) > max_length:
                ids = ids[: max_length - 1] + [self.eos_id] if add_eos else ids[:max_length]
            out.append(ids)
        return out

    def pad_batch(self, batches: list[list[int]]) -> BatchEncoding:
        max_len = max(len(b) for b in batches)
        input_ids = torch.full((len(batches), max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batches), max_len), dtype=torch.long)
        for i, ids in enumerate(batches):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        return BatchEncoding(input_ids=input_ids, attention_mask=attention_mask)


if __name__ == "__main__":
    import sys
    tok = SPMTokenizer(sys.argv[1] if len(sys.argv) > 1 else "mt/data/tokenizer/spm_eneo_32k.model")
    print(f"vocab_size={tok.vocab_size} pad={tok.pad_id} eos={tok.eos_id} <en>={tok.en_id} <eo>={tok.eo_id}")
    sample = "Hello, how are you today?"
    src = tok.encode(sample, lang="eo")
    print(f"src ids ({len(src)}): {src}")
    print(f"roundtrip: {tok.decode(src)!r}")
