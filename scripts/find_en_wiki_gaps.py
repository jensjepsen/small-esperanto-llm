"""Discover EN Wikipedia articles with NO <target-lang> Wikipedia counterpart.

These are high-value translation candidates for continued pretrain/mid-train:
- guaranteed orthogonal (not redundant with existing target-lang Wikipedia)
- rankable by pageviews so we prioritize impactful articles

Three phases:
  fetch-langlinks:  download + parse enwiki-latest-langlinks.sql.gz, build set
                    of enwiki page_ids that already have a target-lang sitelink.
  fetch-pageviews:  download one month of pageview_complete, parse to
                    {title: monthly_views} map for EN Wikipedia.
  find-gaps:        stream wikimedia/wikipedia EN dataset, emit articles whose
                    page_id is NOT in target-lang-linked set, filtered by
                    --min-length, ranked by --rank-by, capped by --max-articles.

--target-lang defaults to "da" (Danish). Set to "eo" for Esperanto etc.
Output paths in DATA_DIR are suffixed with the target-lang so multiple runs
don't collide.

Usage:
    uv run python scripts/find_en_wiki_gaps.py fetch-langlinks --target-lang da
    uv run python scripts/find_en_wiki_gaps.py fetch-pageviews --month 2026-05
    uv run python scripts/find_en_wiki_gaps.py find-gaps --target-lang da \\
        --rank-by views --max-articles 50000 --min-length 2000 \\
        --out data/wiki_gaps/en_only_da_candidates.jsonl
"""
import argparse
import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("/mnt/data2/wiki_gaps")
LANGLINKS_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-langlinks.sql.gz"
LANGLINKS_FILE = DATA_DIR / "enwiki-latest-langlinks.sql.gz"
PAGEVIEWS_FILE = DATA_DIR / "en_pageviews.tsv"  # title \t monthly_views


def linked_pageids_file(target_lang: str) -> Path:
    return DATA_DIR / f"en_pageids_with_{target_lang}.txt"

# MySQL dump INSERT rows look like:
# INSERT INTO `langlinks` VALUES (1234,'eo','Esperanta Titolo'),(5678,'fr','...'),...
INSERT_LINE_RE = re.compile(rb"^INSERT INTO `langlinks` VALUES ")
ROW_RE = re.compile(rb"\((\d+),'([a-z\-]+)',")


def cmd_fetch_langlinks(args):
    """Download enwiki langlinks SQL dump, extract EN page_ids with target-lang sitelink."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tgt = args.target_lang.encode()
    out_file = linked_pageids_file(args.target_lang)

    if not LANGLINKS_FILE.exists() or args.refresh:
        print(f"downloading {LANGLINKS_URL} -> {LANGLINKS_FILE}", flush=True)
        subprocess.run(
            ["curl", "-L", "--fail", "--retry", "3",
             "-o", str(LANGLINKS_FILE), LANGLINKS_URL],
            check=True,
        )
        size_gb = LANGLINKS_FILE.stat().st_size / 1e9
        print(f"  downloaded {size_gb:.2f} GB", flush=True)
    else:
        print(f"reusing existing {LANGLINKS_FILE}", flush=True)

    print(f"parsing langlinks; extracting EN page_ids with ll_lang={args.target_lang!r}...", flush=True)
    en_linked: set[int] = set()
    n_lines = 0
    with gzip.open(LANGLINKS_FILE, "rb") as f:
        for line in f:
            n_lines += 1
            if not INSERT_LINE_RE.match(line):
                continue
            for m in ROW_RE.finditer(line):
                page_id, lang = m.group(1), m.group(2)
                if lang == tgt:
                    en_linked.add(int(page_id))
            if n_lines % 10000 == 0:
                print(f"  scanned {n_lines:,} lines, found {len(en_linked):,} EN pages with {args.target_lang} sitelink",
                      flush=True)

    print(f"writing {len(en_linked):,} page_ids -> {out_file}", flush=True)
    with out_file.open("w") as f:
        for pid in sorted(en_linked):
            f.write(f"{pid}\n")
    print("done.", flush=True)


def cmd_fetch_pageviews(args):
    """Download one month of pageview_complete, build {title: monthly_views} for EN."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    year, month = args.month.split("-")
    fname = f"pageviews-{year}{month}-user.bz2"
    url = (f"https://dumps.wikimedia.org/other/pageview_complete/monthly/"
           f"{year}/{year}-{month}/{fname}")
    raw_path = DATA_DIR / fname

    if not raw_path.exists() or args.refresh:
        print(f"downloading {url}", flush=True)
        subprocess.run(["curl", "-L", "--fail", "--retry", "3",
                        "-o", str(raw_path), url], check=True)
        print(f"  downloaded {raw_path.stat().st_size/1e9:.2f} GB", flush=True)
    else:
        print(f"reusing existing {raw_path}", flush=True)

    print(f"parsing {fname} for en.wikipedia entries...", flush=True)
    # Format per line (space-separated, 6 fields):
    #   <domain> <title> <agent> <access> <monthly_views> <hourly_pattern>
    # e.g. "en.wikipedia Esperanto user desktop 12345 A100B50..."
    # We want domain == 'en.wikipedia' (full-text articles, not 'en.m' mobile etc.)
    import bz2
    titles: dict[str, int] = {}
    n_lines = 0
    with bz2.open(raw_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 5:
                continue
            if parts[0] != "en.wikipedia":
                continue
            title, views = parts[1], parts[4]
            try:
                v = int(views)
            except ValueError:
                continue
            # Sum across agent × access variants for same title
            titles[title] = titles.get(title, 0) + v
            if n_lines % 5_000_000 == 0:
                print(f"  scanned {n_lines:,} lines; en.wikipedia titles so far: {len(titles):,}",
                      flush=True)

    print(f"writing {len(titles):,} titles -> {PAGEVIEWS_FILE}", flush=True)
    # Sort desc by views for friendlier inspection
    with PAGEVIEWS_FILE.open("w") as f:
        for title, views in sorted(titles.items(), key=lambda kv: -kv[1]):
            f.write(f"{title}\t{views}\n")
    print(f"done. top 5:", flush=True)
    for title, views in sorted(titles.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {views:>10,} {title}")


def _load_pageviews() -> dict[str, int]:
    if not PAGEVIEWS_FILE.exists():
        sys.exit(f"ERROR: {PAGEVIEWS_FILE} missing — run fetch-pageviews first")
    print(f"loading pageviews from {PAGEVIEWS_FILE}...", flush=True)
    views: dict[str, int] = {}
    with PAGEVIEWS_FILE.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                views[parts[0]] = int(parts[1])
    print(f"  {len(views):,} titles loaded", flush=True)
    return views


def _pageview_key(title: str) -> str:
    """Wikipedia URL form: spaces → underscores. Pageview keys use this form."""
    return title.replace(" ", "_")


def cmd_find_gaps(args):
    """Stream EN Wikipedia, emit articles whose page_id is NOT in target-lang-linked set."""
    en_linked: set[int] = set()
    if not args.no_target_filter:
        linked_file = linked_pageids_file(args.target_lang)
        if not linked_file.exists():
            sys.exit(f"ERROR: {linked_file} missing — run fetch-langlinks first "
                     f"with --target-lang {args.target_lang}")
        print(f"loading {args.target_lang}-linked page_ids from {linked_file}...", flush=True)
        with linked_file.open() as f:
            en_linked = set(int(line.strip()) for line in f if line.strip())
        print(f"  {len(en_linked):,} loaded", flush=True)
    else:
        print("--no-target-filter: emitting ALL articles matching filters "
              "(target-lang existence ignored)", flush=True)

    title_whitelist: set[str] | None = None
    if args.title_filter:
        print(f"loading title whitelist from {args.title_filter}...", flush=True)
        with open(args.title_filter) as f:
            title_whitelist = set(line.strip() for line in f if line.strip())
        print(f"  {len(title_whitelist):,} titles", flush=True)

    views_map: dict[str, int] | None = None
    if args.rank_by == "views":
        views_map = _load_pageviews()

    from datasets import load_dataset
    print(f"streaming wikimedia/wikipedia/{args.wiki_config}...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", args.wiki_config,
                      split="train", streaming=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_seen = n_gap = n_emitted = 0
    candidates: list[dict] = []

    for row in ds:
        n_seen += 1
        pid = int(row["id"])
        text = row["text"]
        if len(text) < args.min_length:
            continue
        if pid in en_linked:
            continue
        if title_whitelist is not None and row["title"] not in title_whitelist:
            continue
        n_gap += 1
        rec = {
            "page_id": pid,
            "title": row["title"],
            "url": row["url"],
            "length": len(text),
            "text": text,
        }
        if views_map is not None:
            rec["views"] = views_map.get(_pageview_key(row["title"]), 0)
        candidates.append(rec)
        if n_seen % 50000 == 0:
            print(f"  scanned {n_seen:,}; gap-candidates so far: {n_gap:,}",
                  flush=True)

    print(f"\nscan complete: {n_seen:,} articles seen, {n_gap:,} gap candidates "
          f"(>= {args.min_length:,} chars)", flush=True)

    if args.rank_by == "length":
        candidates.sort(key=lambda r: -r["length"])
    elif args.rank_by == "views":
        candidates.sort(key=lambda r: -r["views"])
    candidates = candidates[: args.max_articles]
    print(f"emitting top {len(candidates):,} -> {out_path}", flush=True)

    with out_path.open("w") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            n_emitted += 1
    total_chars = sum(c["length"] for c in candidates)
    print(f"  wrote {n_emitted:,} rows, total {total_chars/1e6:.1f}M chars "
          f"(~{total_chars/4/1e6:.1f}M tokens at ~4 chars/tok)", flush=True)


def main():
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("fetch-langlinks",
                        help="download + parse enwiki langlinks SQL dump")
    p1.add_argument("--target-lang", default="da",
                    help="ISO 639-1 wiki code (default da for Danish; eo, fr, es, ...)")
    p1.add_argument("--refresh", action="store_true",
                    help="redownload even if local file exists")
    p1.set_defaults(func=cmd_fetch_langlinks)

    p1b = sub.add_parser("fetch-pageviews",
                         help="download + parse one month of pageview_complete")
    p1b.add_argument("--month", default="2026-05",
                     help="YYYY-MM (default: 2026-05; use a complete past month)")
    p1b.add_argument("--refresh", action="store_true",
                     help="redownload even if local file exists")
    p1b.set_defaults(func=cmd_fetch_pageviews)

    p2 = sub.add_parser("find-gaps",
                        help="stream EN wiki, emit gap candidates as JSONL")
    p2.add_argument("--target-lang", default="da",
                    help="ISO 639-1 wiki code (default da for Danish)")
    p2.add_argument("--wiki-config", default="20231101.en",
                    help="HF wikimedia/wikipedia config (default: 20231101.en)")
    p2.add_argument("--min-length", type=int, default=2000,
                    help="drop articles shorter than this many chars (default 2000)")
    p2.add_argument("--max-articles", type=int, default=50000,
                    help="cap output count (default 50,000)")
    p2.add_argument("--no-target-filter", action="store_true",
                    help="Skip the target-lang-linked filter — emit ALL articles "
                         "matching --title-filter regardless of target-lang coverage.")
    p2.add_argument("--title-filter", default=None,
                    help="Optional path to newline-delimited title whitelist. "
                         "Only articles whose title matches will be emitted.")
    p2.add_argument("--rank-by", choices=["length", "views", "stream"], default="length",
                    help="length: emit longest first; views: by monthly pageviews "
                         "(requires fetch-pageviews first); stream: dataset order")
    p2.add_argument("--out", default="/mnt/data2/wiki_gaps/en_only_candidates.jsonl",
                    help="output JSONL path")
    p2.set_defaults(func=cmd_find_gaps)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
