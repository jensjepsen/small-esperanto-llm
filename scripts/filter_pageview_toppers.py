"""Filter top-N most-viewed English Wikipedia articles NOT already in our set.

Reads /mnt/data2/wiki_gaps/en_pageviews.tsv (sorted desc by views) and
emits the top-N titles that:
  - are mainspace articles (no `:` prefix like Special: / Wikipedia: / etc.)
  - aren't meta or purely-punctuation entries
  - aren't in the existing wiki_quality_titles.txt set

Titles use underscore form in the pageview file; converted to spaces to
match MediaWiki conventions and our existing lists.

Usage:
    uv run python scripts/filter_pageview_toppers.py \\
        --pageviews /mnt/data2/wiki_gaps/en_pageviews.tsv \\
        --existing /mnt/data2/wiki_gaps/wiki_quality_titles.txt \\
        --n 50000 \\
        --out /mnt/data2/wiki_gaps/pageview_toppers.txt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

NS_PREFIXES = (
    "Special:", "Wikipedia:", "Category:", "File:", "Talk:", "User:",
    "User_talk:", "Help:", "Portal:", "Template:", "Template_talk:",
    "Module:", "MediaWiki:", "Draft:", "Book:", "TimedText:",
    "Wikipedia_talk:", "Category_talk:", "File_talk:", "Help_talk:",
    "Portal_talk:", "Module_talk:", "MediaWiki_talk:", "Draft_talk:",
    "Book_talk:", "TimedText_talk:",
)


def is_valid_title(title: str) -> bool:
    if not title or title == "-":
        return False
    # namespace prefixes
    for p in NS_PREFIXES:
        if title.startswith(p):
            return False
    # Any colon-namespace we didn't catch (rare custom / cross-wiki)
    if ":" in title and title.split(":", 1)[0] in {"en", "de", "fr", "es"}:
        return False
    # Pure punctuation / very short
    if not re.search(r"[A-Za-z]", title):
        return False
    if len(title) < 2:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pageviews", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/en_pageviews.tsv"))
    ap.add_argument("--existing", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/wiki_quality_titles.txt"))
    ap.add_argument("--n", type=int, default=50_000,
                    help="Number of top-viewed new titles to emit")
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/pageview_toppers.txt"))
    ap.add_argument("--merged-out", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/wiki_all_titles.txt"),
                    help="Optional: existing + new merged list")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    existing = {l.strip() for l in args.existing.read_text().splitlines() if l.strip()}
    print(f"existing titles: {len(existing):,}", flush=True)

    picked = []
    n_scanned = 0
    n_ns_filtered = 0
    n_dup_filtered = 0
    n_invalid = 0
    with args.pageviews.open() as f:
        for line in f:
            n_scanned += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            title_us, views = parts[0], parts[1]
            # Convert underscores to spaces
            title = title_us.replace("_", " ")
            if not is_valid_title(title):
                n_invalid += 1
                continue
            if title.startswith(NS_PREFIXES) or ":" in title.split()[0]:
                n_ns_filtered += 1
                continue
            if title in existing:
                n_dup_filtered += 1
                continue
            picked.append((title, int(views)))
            if len(picked) >= args.n:
                break
            if len(picked) % 10_000 == 0:
                print(f"  scanned {n_scanned:,}  picked {len(picked):,}", flush=True)

    print(f"\nscanned: {n_scanned:,} pageview rows")
    print(f"  invalid: {n_invalid:,}")
    print(f"  ns-filtered: {n_ns_filtered:,}")
    print(f"  in-existing: {n_dup_filtered:,}")
    print(f"  picked: {len(picked):,}")
    if picked:
        print(f"  view range: {picked[-1][1]:,} to {picked[0][1]:,}")
    print()
    print("top 15:")
    for t, v in picked[:15]:
        print(f"  {v:>10,}  {t}")

    if args.stats_only:
        return

    args.out.write_text("\n".join(t for t, _ in picked) + "\n")
    print(f"\nwrote pageview toppers → {args.out}")

    merged = sorted(existing | {t for t, _ in picked})
    args.merged_out.write_text("\n".join(merged) + "\n")
    print(f"wrote merged list ({len(merged):,} titles) → {args.merged_out}")


if __name__ == "__main__":
    main()
