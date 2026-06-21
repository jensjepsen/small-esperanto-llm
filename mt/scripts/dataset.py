"""Parallel JSONL dataset + Seq2Seq collator for the MT trainer."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from sp_tokenizer import SPMTokenizer


_VALID_DIRECTIONS = {"en2eo", "eo2en", "bidir"}


class ParallelDataset(Dataset):
    """Parallel EN↔EO pairs from local JSONL or HF dataset repos.

    Each `path` entry is one of:
      - local file path / Path  → reads {"en", "eo"} rows from JSONL
      - "hf://repo[/split]"    → loads via datasets.load_dataset
        (default split = "train"); requires `en` and `eo` columns

    `direction`:
      - "en2eo" / "eo2en": fixed direction
      - "bidir": each __getitem__ picks a direction (50/50) — gives both
        encoder and decoder exposure to both languages. Per-item randomness
        is reproducible via `seed` (each index has a deterministic direction).
    """

    def __init__(self, paths: list, direction: str = "en2eo", seed: int = 1337):
        assert direction in _VALID_DIRECTIONS, f"direction must be one of {_VALID_DIRECTIONS}"
        self.direction = direction
        self.seed = seed
        self.pairs: list[tuple[str, str]] = []
        for p in paths:
            s = str(p)
            if s.startswith("hf://"):
                from datasets import load_dataset
                parts = s[len("hf://"):].split("/")
                # hf://user/name → split=train; hf://user/name/split → that split
                if len(parts) == 2:
                    repo, split = "/".join(parts), "train"
                elif len(parts) == 3:
                    repo, split = "/".join(parts[:2]), parts[2]
                else:
                    raise ValueError(f"bad hf:// path: {s}")
                ds = load_dataset(repo, split=split)
                for row in ds:
                    self.pairs.append((row["en"], row["eo"]))
            else:
                with Path(s).open() as f:
                    for line in f:
                        r = json.loads(line)
                        self.pairs.append((r["en"], r["eo"]))

    def __len__(self) -> int:
        return len(self.pairs)

    def _pick_direction(self, i: int) -> str:
        if self.direction != "bidir":
            return self.direction
        # Deterministic per-index direction so resumes are reproducible and
        # each pair appears in BOTH directions across the dataset (not flipped
        # randomly every epoch, which would waste signal).
        return "en2eo" if ((i ^ self.seed) & 1) == 0 else "eo2en"

    def __getitem__(self, i: int) -> dict[str, str]:
        en, eo = self.pairs[i]
        d = self._pick_direction(i)
        if d == "en2eo":
            return {"src": en, "tgt": eo, "src_lang": "en", "tgt_lang": "eo"}
        return {"src": eo, "tgt": en, "src_lang": "eo", "tgt_lang": "en"}


class Seq2SeqCollator:
    """Tokenizes a batch on the fly. Source is prefixed with the *target*
    language tag (Marian-style steering). Labels are pad-masked to -100.

    Per-example tagging — supports batches that mix en→eo and eo→en items
    (i.e. bidirectional training).
    """

    def __init__(self, tokenizer: SPMTokenizer, max_src_len: int = 128, max_tgt_len: int = 128,
                 decoder_start_token_id: int | None = None):
        self.tok = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else tokenizer.pad_id
        )

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        # Per-example tag with each item's own target language → enables bidir.
        src_ids = [
            self.tok.encode(b["src"], lang=b["tgt_lang"], add_eos=True)[: self.max_src_len]
            for b in batch
        ]
        tgt_ids = [
            self.tok.encode(b["tgt"], lang=None, add_eos=True)[: self.max_tgt_len]
            for b in batch
        ]
        src = self.tok.pad_batch(src_ids)
        tgt = self.tok.pad_batch(tgt_ids)

        labels = tgt.input_ids.clone()
        labels[labels == self.tok.pad_id] = -100

        # MarianMT does not auto-shift labels → build decoder_input_ids by hand.
        decoder_input_ids = tgt.input_ids.clone()
        decoder_input_ids[:, 1:] = tgt.input_ids[:, :-1]
        decoder_input_ids[:, 0] = self.decoder_start_token_id

        return {
            "input_ids": src.input_ids,
            "attention_mask": src.attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
