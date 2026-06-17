"""Stream CCMatrix en↔eo, apply quality filters, dedup, write JSONL.

CCMatrix is LASER-mined and noisy — most entries pass but the failure modes
are predictable: copy-overs (URLs, numbers), HTML artifacts, length-ratio
mismatches, and near-duplicates of opus-100 subtitle lines.

Filter knobs default to a conservative pass; tune via flags.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

HTML_TAG = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://|www\.")


def normalize(text: str) -> str:
    return " ".join(text.split())


def quality_ok(en: str, eo: str, min_chars: int, max_chars: int,
               min_words: int, ratio_lo: float, ratio_hi: float) -> str | None:
    """Returns None if OK, else a reason string."""
    en, eo = en.strip(), eo.strip()
    if not en or not eo:
        return "empty"
    if len(en) < min_chars or len(eo) < min_chars:
        return "short_chars"
    if len(en) > max_chars or len(eo) > max_chars:
        return "long_chars"
    en_words, eo_words = en.split(), eo.split()
    if len(en_words) < min_words or len(eo_words) < min_words:
        return "short_words"
    if not en_words or not eo_words:
        return "empty_words"
    ratio = len(en_words) / len(eo_words)
    if ratio < ratio_lo or ratio > ratio_hi:
        return "ratio"
    # copy-over: source == target
    if en.lower() == eo.lower():
        return "copy"
    # HTML / URL contamination
    if HTML_TAG.search(en) or HTML_TAG.search(eo):
        return "html"
    if URL_RE.search(en) and URL_RE.search(eo):
        return "url"
    # mostly digits / punctuation
    en_alpha = sum(c.isalpha() for c in en)
    if en_alpha < 0.5 * len(en):
        return "non_alpha"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("mt/data/parallel/ccmatrix_filtered.jsonl"))
    ap.add_argument("--hf-cache", type=str, default="/mnt/data/hf_cache")
    ap.add_argument("--max-records", type=int, default=2_000_000,
                    help="Stop after writing this many pairs (0 = unlimited)")
    ap.add_argument("--min-chars", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--ratio-lo", type=float, default=0.5)
    ap.add_argument("--ratio-hi", type=float, default=2.0)
    ap.add_argument("--dedup", action="store_true", default=True)
    ap.add_argument("--progress-every", type=int, default=50_000)
    args = ap.parse_args()

    os.environ["HF_HOME"] = args.hf_cache

    from datasets import load_dataset

    print("Streaming sentence-transformers/parallel-sentences-ccmatrix en-eo …")
    ds = load_dataset("sentence-transformers/parallel-sentences-ccmatrix", "en-eo",
                      split="train", streaming=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept = 0
    drops: dict[str, int] = {}
    total = 0

    with args.out.open("w") as f:
        for r in ds:
            total += 1
            en = normalize(r["english"])
            eo = normalize(r["non_english"])
            reason = quality_ok(en, eo, args.min_chars, args.max_chars,
                                args.min_words, args.ratio_lo, args.ratio_hi)
            if reason is not None:
                drops[reason] = drops.get(reason, 0) + 1
                continue
            if args.dedup:
                key = en + " ||| " + eo
                if key in seen:
                    drops["dup"] = drops.get("dup", 0) + 1
                    continue
                seen.add(key)
            f.write(json.dumps({"en": en, "eo": eo, "src": "ccmatrix"}, ensure_ascii=False) + "\n")
            kept += 1
            if kept and kept % args.progress_every == 0:
                print(f"  kept={kept:>9d}  seen={total:>9d}  drops={drops}")
            if args.max_records and kept >= args.max_records:
                break

    print(f"\nFinal: kept={kept} / seen={total}  -> {args.out}")
    print("Drop breakdown:", drops)


if __name__ == "__main__":
    main()
