"""Dedup en↔eo parallel JSONLs across corpora.

Iterates input files in priority order. The first occurrence of a normalized
(en, eo) pair wins; later occurrences in the same or lower-priority files are
dropped. Writes <name>_dedup.jsonl alongside each input.

Normalization for the dedup key: lowercased, whitespace-collapsed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def key_of(en: str, eo: str) -> tuple[str, str]:
    return (" ".join(en.lower().split()), " ".join(eo.lower().split()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True,
                    help="Paths to dedup. Priority is the order given: first-seen wins.")
    ap.add_argument("--suffix", default="_dedup",
                    help="Suffix added to each input stem for the output. Default: _dedup")
    args = ap.parse_args()

    seen: set[tuple[str, str]] = set()
    summary = []
    for inp in args.inputs:
        out = inp.with_name(inp.stem + args.suffix + inp.suffix)
        kept = dropped = total = 0
        with inp.open() as fi, out.open("w") as fo:
            for line in fi:
                total += 1
                r = json.loads(line)
                k = key_of(r["en"], r["eo"])
                if k in seen:
                    dropped += 1
                    continue
                seen.add(k)
                fo.write(line)
                kept += 1
        pct_drop = 100 * dropped / total if total else 0
        print(f"  {inp.name:35s} total={total:>8,d}  kept={kept:>8,d}  dropped={dropped:>7,d} ({pct_drop:5.1f}%)  -> {out.name}")
        summary.append((inp.name, total, kept, dropped))

    total_in = sum(s[1] for s in summary)
    total_kept = sum(s[2] for s in summary)
    total_dropped = sum(s[3] for s in summary)
    print(f"\n=== Totals: in={total_in:,}  kept={total_kept:,}  cross-corpus dups removed={total_dropped:,} "
          f"({100*total_dropped/total_in:.1f}%) ===")


if __name__ == "__main__":
    main()
