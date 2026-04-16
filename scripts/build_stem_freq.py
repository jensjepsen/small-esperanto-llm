"""Build a stem-frequency dictionary from the pretraining corpus.

Strips Esperanto grammatical endings to get stems, then counts occurrences.
Output is a JSON file keyed by stem → count, used by the verifier's lexicon
check to distinguish real (frequent) words from likely confabulations.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import datasets  # type: ignore
from esperanto_lm.verify import _strip_ending

# Same word-tokenization regex as the verifier.
_WORD_RE = re.compile(
    r"[a-zĉĝĥĵŝŭàáâãäåæèéêëìíîïñòóôõöøùúûüýÿœßšžčćďěľňřťůąęłńśźż]+"
    r"(?:-[a-zĉĝĥĵŝŭàáâãäåæèéêëìíîïñòóôõöøùúûüýÿœßšžčćďěľňřťůąęłńśźż]+)*",
    re.IGNORECASE,
)


def stems_in(text: str) -> list[str]:
    out = []
    for m in _WORD_RE.finditer(text.lower()):
        stem, _, _, _ = _strip_ending(m.group(0))
        if len(stem) >= 2:
            out.append(stem)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki", type=Path, default=Path("data/eo_wiki"),
                        help="HF dataset dir with eo_wiki")
    parser.add_argument("--gutenberg", type=Path, default=Path("data/gutenberg"),
                        help="Dir of Gutenberg .txt files")
    parser.add_argument("--output", type=Path, default=Path("resources/stem_freq.json"))
    parser.add_argument("--min-count", type=int, default=2,
                        help="Only persist stems with count >= this")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Cap documents processed per source (0=all)")
    args = parser.parse_args()

    counts: Counter = Counter()

    if args.wiki.exists():
        print(f"Reading {args.wiki}...", flush=True)
        ds = datasets.load_from_disk(str(args.wiki))["train"]
        total = len(ds) if not args.max_docs else min(args.max_docs, len(ds))
        for i in range(total):
            counts.update(stems_in(ds[i]["text"]))
            if (i + 1) % 20000 == 0:
                print(f"  wiki {i+1}/{total}  unique={len(counts):,}", flush=True)

    if args.gutenberg.exists():
        print(f"Reading {args.gutenberg}...", flush=True)
        files = sorted(args.gutenberg.glob("*.txt"))
        for j, f in enumerate(files):
            try:
                counts.update(stems_in(f.read_text()))
            except Exception as e:
                print(f"  skip {f.name}: {e}", flush=True)
            if (j + 1) % 100 == 0:
                print(f"  gutenberg {j+1}/{len(files)}  unique={len(counts):,}", flush=True)

    # Persist only stems above threshold
    kept = {s: c for s, c in counts.items() if c >= args.min_count}
    print(f"Total unique stems: {len(counts):,}")
    print(f"Stems kept (count>={args.min_count}): {len(kept):,}")
    print(f"Total tokens: {sum(counts.values()):,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(kept, f)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
