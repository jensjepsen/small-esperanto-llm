"""Build en↔eo parallel from Tatoeba dumps.

Inputs (download from downloads.tatoeba.org):
  - eng_sentences.tsv.bz2     id  lang  text
  - epo-eng_links.tsv.bz2     epo_id  eng_id
  - epo_sentences.tsv         id  lang  text     (already present at data/)
"""
from __future__ import annotations

import argparse
import bz2
import json
import re
from pathlib import Path

HTML_TAG = re.compile(r"<[^>]+>")


def normalize(text: str) -> str:
    return " ".join(text.split())


def quality_ok(en: str, eo: str, min_chars: int, max_chars: int,
               ratio_lo: float, ratio_hi: float) -> bool:
    en, eo = en.strip(), eo.strip()
    if not en or not eo:
        return False
    if len(en) < min_chars or len(eo) < min_chars:
        return False
    if len(en) > max_chars or len(eo) > max_chars:
        return False
    en_w, eo_w = en.split(), eo.split()
    if not en_w or not eo_w:
        return False
    ratio = len(en_w) / len(eo_w)
    if ratio < ratio_lo or ratio > ratio_hi:
        return False
    if HTML_TAG.search(en) or HTML_TAG.search(eo):
        return False
    return True


def load_sentences_tsv(path: Path, open_fn=open) -> dict[int, str]:
    out: dict[int, str] = {}
    with open_fn(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                sid = int(parts[0])
            except ValueError:
                continue
            out[sid] = parts[2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epo-sentences", type=Path, default=Path("data/epo_sentences.tsv"))
    ap.add_argument("--eng-sentences", type=Path, default=Path("mt/data/raw/eng_sentences.tsv.bz2"))
    ap.add_argument("--links", type=Path, default=Path("mt/data/raw/epo-eng_links.tsv.bz2"))
    ap.add_argument("--out", type=Path, default=Path("mt/data/parallel/tatoeba_train.jsonl"))
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--ratio-lo", type=float, default=0.4)
    ap.add_argument("--ratio-hi", type=float, default=2.5)
    args = ap.parse_args()

    print(f"Loading eo sentences: {args.epo_sentences}")
    eo_sent = load_sentences_tsv(args.epo_sentences)
    print(f"  {len(eo_sent):,} eo sentences")

    print(f"Loading en sentences: {args.eng_sentences}")
    en_sent = load_sentences_tsv(args.eng_sentences, open_fn=bz2.open)
    print(f"  {len(en_sent):,} en sentences")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept, dropped, missing = 0, 0, 0

    with bz2.open(args.links, "rt") as fin, args.out.open("w") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                epo_id, eng_id = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            eo = eo_sent.get(epo_id)
            en = en_sent.get(eng_id)
            if eo is None or en is None:
                missing += 1
                continue
            eo, en = normalize(eo), normalize(en)
            if not quality_ok(en, eo, args.min_chars, args.max_chars,
                              args.ratio_lo, args.ratio_hi):
                dropped += 1
                continue
            key = en + " ||| " + eo
            if key in seen:
                continue
            seen.add(key)
            fout.write(json.dumps({"en": en, "eo": eo, "src": "tatoeba"}, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\nkept={kept:,}  dropped={dropped:,}  missing={missing:,}  -> {args.out}")


if __name__ == "__main__":
    main()
