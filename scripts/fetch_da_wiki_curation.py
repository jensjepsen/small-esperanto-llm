"""Build a curation whitelist (pageid, title) from Danish Wikipedia SQL dumps.

Uses local SQL dumps rather than the MediaWiki API — orders of magnitude
faster (no rate limits, no HTTP overhead). Downloads three tables from
dumps.wikimedia.org/dawiki/latest, parses them in-memory, and walks the
category tree recursively from a set of root categories to produce a
pageid whitelist for downstream filtering.

Tables used:
  - page:          page_id → (namespace, title)
  - categorylinks: cl_from (child pageid) → cl_to (parent category name)
                   with cl_type ∈ {'page','subcat','file'}
  - category:      not strictly needed (name is in cl_to)

Output: pageid<TAB>title per line, sorted by pageid.

Usage:
    uv run --no-project python scripts/fetch_da_wiki_curation.py \\
        --out /mnt/data2/da_wiki_curation/pageids.tsv

    # Custom roots + depth
    uv run --no-project python scripts/fetch_da_wiki_curation.py \\
        --roots "Personer fra Danmark" "Fysik" --depth 5 \\
        --out out.tsv
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

DUMPS = "https://dumps.wikimedia.org/dawiki/latest"
FILES = {
    "page":          f"{DUMPS}/dawiki-latest-page.sql.gz",
    "categorylinks": f"{DUMPS}/dawiki-latest-categorylinks.sql.gz",
    "linktarget":    f"{DUMPS}/dawiki-latest-linktarget.sql.gz",
}
CACHE_DIR = Path("/mnt/data2/da_wiki_dumps")

DEFAULT_ROOTS = [
    "Fremragende artikler",
    # Danish umbrellas
    "Danmark", "Danmarks historie", "Danmarks geografi",
    "Danmarks politik",
    "Personer fra Danmark", "Forfattere fra Danmark",
    "Dansk kultur", "Sundhedsvæsen i Danmark",
    "Grønland", "Færøerne",
    # Universal knowledge (verified names — empty ones like "Medicin"
    # replaced with populated variants)
    "Videnskab", "Matematik", "Fysik", "Kemi", "Biologi", "Astronomi",
    "Geologi", "Lægevidenskab", "Sundhed", "Sygdomme", "Teknik", "Datalogi",
    "Filosofi", "Religion", "Psykologi", "Sociologi", "Økonomi",
    "Historie", "Geografi", "Sprog",
    "Kunst", "Musik", "Litteratur", "Film", "Teater", "Arkitektur",
    "Sport", "Mad og drikke", "Miljø", "Kultur", "Samfund",
]

NS_MAIN = 0
NS_CATEGORY = 14


def download_if_needed(url: str, cache: Path) -> Path:
    """Idempotent download to cache dir; skip if file exists + non-empty."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_size > 0:
        return cache
    print(f"  downloading {url} → {cache}", flush=True)
    t0 = time.time()
    req = urllib.request.Request(url, headers={
        "User-Agent": "espllm-da-curation/1.0 (jens.jepsen@gmail.com)",
    })
    with urllib.request.urlopen(req, timeout=120) as resp, cache.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"    ok, {cache.stat().st_size/1e6:.1f} MB in {time.time()-t0:.1f}s",
          flush=True)
    return cache


# ── SQL parsing ────────────────────────────────────────────────────────────
# MediaWiki dumps use MySQL INSERT syntax. Each INSERT contains many tuples.
# We iterate rows within each INSERT via a lightweight state machine that
# handles quoted strings (with backslash escapes) and unquoted numeric/enum
# fields.

_ROW_RE_HEADER = re.compile(rb"^INSERT INTO `(\w+)` VALUES\s*")


def _iter_sql_rows(gz_path: Path):
    """Yield each row as a list of Python-typed values (int, str, bytes, None).
    Streams over gzipped SQL dump line-by-line; each INSERT line contains
    many tuples."""
    with gzip.open(gz_path, "rb") as f:
        for line in f:
            m = _ROW_RE_HEADER.match(line)
            if not m:
                continue
            body = line[m.end():]
            # body ends with ");\n"; strip trailing ";\n" and outer "(...)"
            # then split at tuple boundaries.
            # Walk char by char.
            yield from _walk_tuples(body)


def _walk_tuples(buf: bytes):
    """Given a bytes buffer of `(...),(...),(...);`, yield each tuple as list."""
    i, n = 0, len(buf)
    while i < n:
        if buf[i:i+1] == b"(":
            row, i = _parse_tuple(buf, i + 1)
            yield row
            # skip optional comma
            while i < n and buf[i:i+1] in b", \n":
                i += 1
        elif buf[i:i+1] == b";":
            return
        else:
            i += 1


def _parse_tuple(buf: bytes, i: int):
    """Parse `field,field,...)` starting at position i, return (list, i_after_closing_paren)."""
    fields = []
    n = len(buf)
    while i < n:
        c = buf[i:i+1]
        if c == b"'":
            # quoted string
            i += 1
            start = i
            out = bytearray()
            while i < n:
                cc = buf[i:i+1]
                if cc == b"\\":
                    if i + 1 < n:
                        nxt = buf[i+1:i+2]
                        # escape sequences: \n \t \\ \' \0 \r \" \b \Z
                        out.extend({
                            b"n": b"\n", b"t": b"\t", b"r": b"\r",
                            b"\\": b"\\", b"'": b"'", b"\"": b"\"",
                            b"0": b"\x00", b"b": b"\x08", b"Z": b"\x1a",
                        }.get(nxt, nxt))
                        i += 2
                        continue
                if cc == b"'":
                    i += 1
                    break
                out.extend(cc)
                i += 1
            try:
                fields.append(out.decode("utf-8", errors="replace"))
            except Exception:
                fields.append(bytes(out))
        elif c == b")":
            return fields, i + 1
        elif c == b",":
            i += 1
        elif c in b" \n\r\t":
            i += 1
        else:
            # unquoted: NULL, integer, decimal, or bare enum
            start = i
            while i < n and buf[i:i+1] not in b",)":
                i += 1
            token = buf[start:i].strip().decode("ascii", errors="replace")
            if token == "NULL":
                fields.append(None)
            else:
                try:
                    fields.append(int(token))
                except ValueError:
                    try:
                        fields.append(float(token))
                    except ValueError:
                        fields.append(token)
    return fields, i


# ── Loaders ────────────────────────────────────────────────────────────────
# page columns (MediaWiki, as of MW 1.42):
#   page_id, page_namespace, page_title, page_is_redirect,
#   page_is_new, page_random, page_touched, page_links_updated,
#   page_latest, page_len, page_content_model, page_lang
# We only need columns 0, 1, 2.

def load_page_table(gz_path: Path):
    """Return two dicts:
       pageid_to_title (namespace 0 only): int → str
       catname_to_pageid: str → int (namespace 14, category)"""
    pageid_to_title: dict[int, str] = {}
    catname_to_pageid: dict[str, int] = {}
    t0 = time.time()
    total = 0
    for row in _iter_sql_rows(gz_path):
        total += 1
        if len(row) < 3:
            continue
        pid, ns, title = row[0], row[1], row[2]
        if not isinstance(pid, int) or not isinstance(ns, int):
            continue
        if not isinstance(title, str):
            title = str(title)
        # MediaWiki titles use underscores in place of spaces internally.
        title = title.replace("_", " ")
        if ns == NS_MAIN:
            pageid_to_title[pid] = title
        elif ns == NS_CATEGORY:
            catname_to_pageid[title] = pid
    print(f"  page: {total:,} rows → {len(pageid_to_title):,} articles + "
          f"{len(catname_to_pageid):,} categories in {time.time()-t0:.1f}s",
          flush=True)
    return pageid_to_title, catname_to_pageid


# linktarget columns:
#   lt_id, lt_namespace, lt_title

def load_linktarget(gz_path: Path) -> dict[int, str]:
    """Return lt_id → category NAME (only rows with lt_namespace = 14).
    We don't need articles/other namespaces from linktarget."""
    lt_id_to_catname: dict[int, str] = {}
    t0 = time.time()
    total = 0
    for row in _iter_sql_rows(gz_path):
        total += 1
        if len(row) < 3:
            continue
        lt_id, ns, title = row[0], row[1], row[2]
        if not isinstance(lt_id, int) or not isinstance(ns, int):
            continue
        if ns != NS_CATEGORY:
            continue
        if not isinstance(title, str):
            title = str(title)
        lt_id_to_catname[lt_id] = title.replace("_", " ")
    print(f"  linktarget: {total:,} rows → {len(lt_id_to_catname):,} category "
          f"targets in {time.time()-t0:.1f}s", flush=True)
    return lt_id_to_catname


# categorylinks (MW 1.42+) columns:
#   cl_from, cl_sortkey, cl_timestamp, cl_sortkey_prefix,
#   cl_type, cl_collation_id, cl_target_id
# We need 0 (cl_from), 4 (cl_type), 6 (cl_target_id).

def load_categorylinks(gz_path: Path,
                        pageid_to_title: dict[int, str],
                        catname_to_pageid: dict[str, int],
                        lt_id_to_catname: dict[int, str]):
    """Return two dicts:
       cat_pages[catname]     = set of pageids (articles in this cat)
       cat_subcats[catname]   = set of subcategory NAMES"""
    cat_pages: dict[str, set[int]] = defaultdict(set)
    cat_subcats: dict[str, set[str]] = defaultdict(set)
    pageid_to_catname = {pid: name for name, pid in catname_to_pageid.items()}
    t0 = time.time()
    total = 0
    skipped_no_target = 0
    for row in _iter_sql_rows(gz_path):
        total += 1
        if len(row) < 7:
            continue
        cl_from, cl_type, cl_target_id = row[0], row[4], row[6]
        if not isinstance(cl_from, int) or not isinstance(cl_target_id, int):
            continue
        parent_catname = lt_id_to_catname.get(cl_target_id)
        if parent_catname is None:
            skipped_no_target += 1
            continue
        if cl_type == "page":
            if cl_from in pageid_to_title:
                cat_pages[parent_catname].add(cl_from)
        elif cl_type == "subcat":
            sub_name = pageid_to_catname.get(cl_from)
            if sub_name is not None:
                cat_subcats[parent_catname].add(sub_name)
    print(f"  categorylinks: {total:,} rows → {len(cat_pages):,} cats have "
          f"pages, {len(cat_subcats):,} cats have subcats "
          f"({skipped_no_target:,} skipped: linktarget miss) "
          f"in {time.time()-t0:.1f}s", flush=True)
    return cat_pages, cat_subcats


# ── Category tree walk ─────────────────────────────────────────────────────

def walk(root: str, max_depth: int,
         cat_pages: dict[str, set[int]],
         cat_subcats: dict[str, set[str]]) -> tuple[set[int], set[str]]:
    """BFS from root, return (pageids, walked-categories) under root within
    depth. Walk uses per-root seen-set (no cross-root sharing), so each
    root's count is genuine; global pageid dedup is done by the caller."""
    out: set[int] = set()
    seen: set[str] = set()
    q = deque([(root, 0)])
    while q:
        cat, d = q.popleft()
        if cat in seen:
            continue
        seen.add(cat)
        out.update(cat_pages.get(cat, ()))
        if d < max_depth:
            for sub in cat_subcats.get(cat, ()):
                if sub not in seen:
                    q.append((sub, d + 1))
    return out, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=None)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()

    roots = args.roots or DEFAULT_ROOTS

    print("=== fetching SQL dumps ===", flush=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    page_gz = download_if_needed(FILES["page"],
                                  args.cache / "dawiki-latest-page.sql.gz")
    cl_gz   = download_if_needed(FILES["categorylinks"],
                                  args.cache / "dawiki-latest-categorylinks.sql.gz")
    lt_gz   = download_if_needed(FILES["linktarget"],
                                  args.cache / "dawiki-latest-linktarget.sql.gz")

    print("\n=== parsing page table ===", flush=True)
    pageid_to_title, catname_to_pageid = load_page_table(page_gz)

    print("\n=== parsing linktarget ===", flush=True)
    lt_id_to_catname = load_linktarget(lt_gz)

    print("\n=== parsing categorylinks ===", flush=True)
    cat_pages, cat_subcats = load_categorylinks(
        cl_gz, pageid_to_title, catname_to_pageid, lt_id_to_catname)

    print(f"\n=== walking category tree from {len(roots)} roots, depth={args.depth} ===",
          flush=True)
    t0 = time.time()
    global_pages: set[int] = set()
    all_walked_cats: set[str] = set()
    per_root_counts: dict[str, int] = {}
    for root in roots:
        found, walked = walk(root, args.depth, cat_pages, cat_subcats)
        global_pages.update(found)
        all_walked_cats.update(walked)
        per_root_counts[root] = len(found)
    print(f"  walked {len(all_walked_cats):,} unique categories → "
          f"{len(global_pages):,} unique articles in {time.time()-t0:.1f}s",
          flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{pid}\t{pageid_to_title[pid]}"
             for pid in sorted(global_pages) if pid in pageid_to_title]
    args.out.write_text("\n".join(lines))
    print(f"  wrote {len(lines):,} pageid<TAB>title lines → {args.out}",
          flush=True)

    if args.manifest:
        args.manifest.write_text(json.dumps({
            "roots": roots,
            "depth": args.depth,
            "categories_walked": len(all_walked_cats),
            "unique_pages": len(global_pages),
            "per_root_counts": per_root_counts,
        }, ensure_ascii=False, indent=2))

    print()
    print("=== per-root counts (top 20) ===")
    for k, v in sorted(per_root_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {v:>7,}  {k}")
    zero_roots = [k for k, v in per_root_counts.items() if v == 0]
    if zero_roots:
        print(f"\n⚠  {len(zero_roots)} roots returned 0 (name may be wrong):")
        for k in zero_roots:
            print(f"     {k}")


if __name__ == "__main__":
    main()
