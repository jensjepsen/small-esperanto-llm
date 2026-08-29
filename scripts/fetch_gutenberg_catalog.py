"""Download Project Gutenberg's full catalog CSV — no books, just metadata.

Output: /mnt/data2/sci_books/gutenberg_catalog.csv
Fields:  Text#, Type, Issued, Title, Language, Authors, Subjects, LoCC, Bookshelves

After this lands, we can filter/rank/cap before deciding which books
to actually download + translate.

Usage:
    uv run python scripts/fetch_gutenberg_catalog.py
"""
import csv
import sys
import urllib.request
from pathlib import Path

OUT = Path("/mnt/data2/sci_books/gutenberg_catalog.csv")
URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
USER_AGENT = "espllm-pretrain-data/1.0 (jens.jepsen@gmail.com)"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}...", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    print(f"  downloaded {len(raw)/1e6:.2f} MB compressed", flush=True)

    import gzip, io
    decompressed = gzip.decompress(raw)
    print(f"  decompressed {len(decompressed)/1e6:.2f} MB", flush=True)

    OUT.write_bytes(decompressed)
    print(f"wrote {OUT}", flush=True)

    # quick sanity scan
    with OUT.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"\ntotal entries: {len(rows):,}")
    print(f"columns: {list(rows[0].keys())}")
    en = [r for r in rows if r.get("Language", "").lower() == "en"]
    print(f"english entries: {len(en):,}")
    # Unique-ish subject sample
    subjects = set()
    for r in en[:5000]:
        for s in (r.get("Subjects") or "").split(";"):
            s = s.strip()
            if s:
                subjects.add(s)
    print(f"distinct subjects in first 5k EN entries: {len(subjects):,}")
    print(f"\nsample row: {rows[0]!r}")


if __name__ == "__main__":
    main()
