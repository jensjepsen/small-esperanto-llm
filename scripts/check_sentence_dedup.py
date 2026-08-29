"""Measure sentence-level duplication INSIDE our EN Gutenberg corpus.

If books share lots of sentences (boilerplate, common phrases, common
quotations), translating UNIQUE sentences once and reusing across books
saves a proportional fraction of MT cost.

For each .txt under /mnt/data2/sci_books/:
  - split into sentences
  - normalize (lowercase, strip whitespace, drop too short / too long)
  - hash and count
Reports: total / unique / dedup ratio / most-duplicated sample sentences.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

BOOKS_DIR = Path("/mnt/data2/sci_books")
_SENT_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_BREAK.split(text) if s.strip()]


def normalize(s: str) -> str:
    # Lowercase, collapse whitespace, strip punctuation noise at edges
    s = re.sub(r"\s+", " ", s.lower().strip())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-len", type=int, default=30,
                    help="ignore sentences shorter than this (noisy)")
    ap.add_argument("--max-len", type=int, default=1000,
                    help="ignore sentences longer than this (paragraph slop)")
    args = ap.parse_args()

    files = sorted(BOOKS_DIR.glob("*.txt"))
    print(f"books: {len(files)}")

    sent_counter: Counter[str] = Counter()
    n_sentences = 0
    n_chars = 0
    book_unique: dict[str, int] = {}

    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="replace")
        sents = split_sentences(text)
        seen_in_book = set()
        for s in sents:
            norm = normalize(s)
            if len(norm) < args.min_len or len(norm) > args.max_len:
                continue
            sent_counter[norm] += 1
            n_sentences += 1
            n_chars += len(s)
            seen_in_book.add(norm)
        book_unique[fp.name] = len(seen_in_book)

    n_unique = len(sent_counter)
    repeat_count = n_sentences - n_unique
    dedup_ratio = 1 - n_unique / max(1, n_sentences)

    print(f"\nsentences (after length filter):")
    print(f"  total occurrences: {n_sentences:,}")
    print(f"  unique sentences:  {n_unique:,}")
    print(f"  repeats:           {repeat_count:,}")
    print(f"  **dedup ratio:     {100*dedup_ratio:.1f}%** of sentences are repeats")

    # Char-weighted: would translating unique only save proportional cost?
    char_total = sum(len(k) * v for k, v in sent_counter.items())
    char_unique = sum(len(k) for k in sent_counter)
    char_dedup_ratio = 1 - char_unique / max(1, char_total)
    print(f"\nchars (proxy for MT cost):")
    print(f"  total chars in sentences: {char_total:,} ({char_total/1e6:.1f}M)")
    print(f"  chars if translate unique only: {char_unique:,} ({char_unique/1e6:.1f}M)")
    print(f"  **char dedup ratio: {100*char_dedup_ratio:.1f}%** savings")

    print(f"\nmost-duplicated sentences (top 15):")
    for sent, count in sent_counter.most_common(15):
        print(f"  {count:>4}×  {sent[:100]}{'…' if len(sent) > 100 else ''}")


if __name__ == "__main__":
    main()
