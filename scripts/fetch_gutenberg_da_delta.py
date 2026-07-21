"""Fetch Danish Gutenberg books not already in Danish Gigaword.

Compares Gutenberg's /browse/languages/da against gigaword/gutenberg IDs;
downloads the delta as UTF-8 plaintext (strips PG header/footer).

Writes to /workspace/work/gutenberg_da_delta/shard_000000.jsonl.gz
with schema {text, source: "gutenberg_delta", id: "gutenberg_<N>"}
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

UA = "espllm-danish/1.0 (jens.jepsen@gmail.com)"
OUT_DIR = Path("/workspace/work/gutenberg_da_delta")


def gutenberg_text(gid: str) -> str | None:
    for url in (
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    ):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:
            continue
    return None


def strip_boilerplate(text: str) -> str:
    m = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", text)
    if m:
        text = text[:m.start()]
    return text.strip()


def catalog_ids() -> list[str]:
    """Fetch current Gutenberg Danish book IDs."""
    req = urllib.request.Request(
        "https://www.gutenberg.org/browse/languages/da",
        headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    return sorted(set(re.findall(r"/ebooks/(\d+)", html)), key=int)


def existing_ids() -> set[str]:
    """IDs already in gigaword/gutenberg subset."""
    path = hf_hub_download("danish-foundation-models/danish-gigaword",
                           filename="gutenberg/gutenberg.parquet",
                           repo_type="dataset")
    tbl = pq.read_table(path)
    ids = tbl.column("id").to_pylist()
    return {str(i).replace("gutenberg_", "") for i in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=1.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_ids = catalog_ids()
    have = existing_ids()
    delta = [i for i in all_ids if i not in have]
    print(f"catalog: {len(all_ids)}  have: {len(have)}  fetch: {len(delta)}",
          flush=True)

    out_path = OUT_DIR / "shard_000000.jsonl.gz"
    n_ok = 0
    n_fail = 0
    total_chars = 0
    t0 = time.time()
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6) as fh:
        for gid in delta:
            raw = gutenberg_text(gid)
            if raw is None:
                print(f"  [{gid}] download failed", flush=True)
                n_fail += 1
                continue
            body = strip_boilerplate(raw)
            if len(body) < 500:
                print(f"  [{gid}] too short ({len(body)} chars)", flush=True)
                n_fail += 1
                continue
            fh.write(json.dumps({
                "text": body,
                "source": "gutenberg_delta",
                "id": f"gutenberg_{gid}",
            }, ensure_ascii=False) + "\n")
            n_ok += 1
            total_chars += len(body)
            print(f"  [{gid}] {len(body):,} chars  (running total {n_ok})",
                  flush=True)
            if args.pause > 0:
                time.sleep(args.pause)

    print(f"\ndone: {n_ok} ok, {n_fail} failed in {(time.time()-t0)/60:.1f}min",
          flush=True)
    print(f"total chars: {total_chars/1e6:.1f}M "
          f"(~{total_chars/4/1e6:.1f}M tokens)", flush=True)


if __name__ == "__main__":
    main()
