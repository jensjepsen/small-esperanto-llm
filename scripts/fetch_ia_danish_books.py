"""Fetch Danish public-domain books from Internet Archive.

Pipeline:
  1. Enumerate all `mediatype:texts AND language:dan` items (~15.5k).
  2. Exclude items in known junk collections (magazine porn, mens_mags).
  3. Parallel workers download {id}_djvu.txt for each item; a 404 means
     no OCR text is available for that item (skip).
  4. Apply length + fasttext Danish language-ID filter (drop wrong-language
     or empty OCR).
  5. Write JSONL shards to /workspace/work/ia_danish/.

Rough numbers: 15k items × ~50% have usable OCR text × ~100k chars/book
  = ~1.5 GB text = ~400-800M tokens after filter.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import multiprocessing as mp
import re
import time
import urllib.request
from functools import partial
from pathlib import Path

UA = "espllm-danish/1.0 (jens.jepsen@gmail.com)"
WORK_ROOT = Path("/workspace/work/ia_danish")
CACHE_LID = Path("/workspace/cache/lid.176.bin")

MIN_CHARS = 500
LANG_THRESHOLD = 0.55
SHARD_ROWS = 500

JUNK_COLLECTIONS = {
    "mensmagazines_post70s", "magazine_packs", "magazine_rack",
    "mensmagazines_pre70s", "mensmagazines",
    "commercialsexworkers", "underground_erotica",
}


def http_get(url: str, timeout: int = 60, headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception:
        return 0, b""


def enumerate_ia_ids(max_items: int | None = None) -> list[dict]:
    """Return list of {identifier, collection[]} for Danish text items."""
    items = []
    per_page = 10000
    for start in range(0, max_items or 20000, per_page):
        rows = per_page if not max_items else min(per_page, max_items - start)
        url = (
            "https://archive.org/advancedsearch.php?"
            "q=language%3Adan+AND+mediatype%3Atexts"
            f"&fl%5B%5D=identifier&fl%5B%5D=collection"
            f"&rows={rows}&page={start//per_page + 1}&output=json"
        )
        status, body = http_get(url, timeout=60)
        if status != 200:
            print(f"[enum] page {start//per_page + 1} status {status}", flush=True)
            break
        data = json.loads(body)
        docs = data.get("response", {}).get("docs", [])
        items.extend(docs)
        total = data.get("response", {}).get("numFound", 0)
        print(f"[enum]   page {start//per_page + 1}: fetched {len(docs)} "
              f"(total so far {len(items)}/{total})", flush=True)
        if len(docs) < rows:
            break
        time.sleep(0.5)
    return items


def filter_junk(items: list[dict]) -> list[str]:
    """Keep item IDs whose collections don't include JUNK_COLLECTIONS."""
    keep = []
    for it in items:
        colls = it.get("collection", [])
        if isinstance(colls, str):
            colls = [colls]
        if any(c in JUNK_COLLECTIONS for c in colls):
            continue
        keep.append(it["identifier"])
    return keep


def _init_lid():
    global _LID
    import fasttext
    _LID = fasttext.load_model(str(CACHE_LID))


def fetch_and_filter(iid: str) -> dict | None:
    """Download {iid}_djvu.txt, apply filters, return dict or None."""
    url = f"https://archive.org/download/{iid}/{iid}_djvu.txt"
    status, body = http_get(url, timeout=90)
    if status != 200 or not body:
        return None
    try:
        text = body.decode("utf-8", errors="replace").strip()
    except Exception:
        return None
    if len(text) < MIN_CHARS:
        return None
    # lang-id on first 2000 chars, replacing newlines
    sample = text[:2000].replace("\n", " ")
    labels, probs = _LID.predict(sample, k=1)
    if labels[0] != "__label__da" or probs[0] < LANG_THRESHOLD:
        return None
    return {"text": text, "source": "ia_danish", "id": f"ia_{iid}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-items", type=int, default=None,
                    help="Cap for testing (default: fetch all ~15.5k)")
    ap.add_argument("--num-workers", type=int, default=32)
    args = ap.parse_args()

    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    # Prefetch fasttext model
    if not CACHE_LID.exists():
        print("[lid] downloading fasttext lid model", flush=True)
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
            CACHE_LID)

    # Phase 1: enumerate
    print(f"[phase1] enumerating IA Danish text items", flush=True)
    items = enumerate_ia_ids(args.max_items)
    print(f"[phase1] {len(items):,} items enumerated", flush=True)

    # Phase 2: filter junk
    ids = filter_junk(items)
    dropped = len(items) - len(ids)
    print(f"[phase2] {len(ids):,} pass junk-filter ({dropped:,} dropped)",
          flush=True)

    # Phase 3: parallel download + filter
    print(f"[phase3] downloading + filtering with {args.num_workers} workers",
          flush=True)
    t0 = time.time()
    shard_idx = 0
    n_shard = 0
    n_kept = 0
    n_no_text = 0
    n_bad_lang = 0
    n_too_short = 0
    fh = gzip.open(WORK_ROOT / f"shard_{shard_idx:06d}.jsonl.gz",
                   "wt", encoding="utf-8", compresslevel=6)
    with mp.Pool(args.num_workers, initializer=_init_lid) as pool:
        for i, result in enumerate(pool.imap_unordered(fetch_and_filter, ids,
                                                       chunksize=4)):
            if result is None:
                # Distinguish reasons only approximately from log messages
                n_no_text += 1  # covers all failure modes (404 / short / bad-lang)
                if (i + 1) % 500 == 0:
                    el = time.time() - t0
                    rate = (i + 1) / el
                    eta = (len(ids) - i - 1) / rate / 60
                    print(f"[phase3]   {i+1:,}/{len(ids):,}  kept={n_kept:,}  "
                          f"({rate:.1f}/s  eta {eta:.1f}min)", flush=True)
                continue
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            n_kept += 1
            n_shard += 1
            if n_shard >= SHARD_ROWS:
                fh.close()
                shard_idx += 1
                n_shard = 0
                fh = gzip.open(WORK_ROOT / f"shard_{shard_idx:06d}.jsonl.gz",
                               "wt", encoding="utf-8", compresslevel=6)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(ids) - i - 1) / rate / 60
                print(f"[phase3]   {i+1:,}/{len(ids):,}  kept={n_kept:,}  "
                      f"({rate:.1f}/s  eta {eta:.1f}min)", flush=True)
    fh.close()

    total_bytes = sum(p.stat().st_size for p in WORK_ROOT.glob("*.jsonl.gz"))
    print(f"[done] kept={n_kept:,} in {shard_idx+1} shards, "
          f"{total_bytes/1e9:.2f} GB compressed, "
          f"{(time.time()-t0)/60:.1f}min", flush=True)
    print(f"       {n_no_text:,} rejected (404 / <{MIN_CHARS}c / non-Danish)",
          flush=True)


if __name__ == "__main__":
    main()
