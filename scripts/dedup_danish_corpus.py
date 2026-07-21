"""Exact-hash dedup for the Danish pretrain corpus.

Rationale: MinHash is overkill for a first-pass corpus. FineWeb-2 already
does internal near-dupe dedup. Cross-source dupes are mostly EXACT copies
(Wikipedia articles verbatim on crawl mirrors). Exact xxh64 dedup runs in
~10 min on 40M docs and gives us the low-hanging fruit.

Pipeline:
  Phase 1 (parallel per shard): hash every doc, emit (hash, source, ordinal)
  Phase 2 (serial, in-memory): merge — for each hash, pick highest-priority
    source; break ties by lowest ordinal (first-seen).
  Phase 3 (parallel per shard): re-read shards, write out only kept docs.

On 256 cores / 1TB RAM: ~5-15 min total for 40M docs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import xxhash

WORK_ROOT = Path("/workspace/work")
OUT_DIR = WORK_ROOT / "dedup"

SOURCE_PRIORITY = {
    "wikipedia": 100,
    "dynaword": 95,          # curated multi-source (subsumes gigaword)
    "gigaword": 90,          # legacy — same team as dynaword
    "gutenberg_da_delta": 85,  # curated PD literature
    "wikisource": 80,
    "ia_danish": 60,         # OCR quality varies; keep below curated but above raw web
    "oscar": 50,
    "fineweb2": 40,          # web crawl
    "culturax": 30,
}

SHARD_ROWS = 100_000
MIN_CHARS = 200


def list_shards(sources: list[str]) -> list[tuple[str, Path]]:
    out = []
    for src in sources:
        d = WORK_ROOT / src
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jsonl.gz")):
            out.append((src, p))
    return out


def hash_shard(args) -> list[tuple[int, str, int]]:
    """Return list of (text_hash, source, ordinal_in_shard) for every doc."""
    src, path = args
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for ord_, line in enumerate(fh):
            try:
                r = json.loads(line)
            except Exception:
                continue
            text = r.get("text", "")
            if len(text) < MIN_CHARS:
                continue
            # xxh64 is ~10 GB/s single-thread; hash the full text
            h = xxhash.xxh64_intdigest(text)
            out.append((h, src, ord_))
    return out


_KEEP_PER_SHARD: dict | None = None


def _init_writer(keep_per_shard):
    global _KEEP_PER_SHARD
    _KEEP_PER_SHARD = keep_per_shard


def filter_and_write_shard(args) -> tuple[str, int, int]:
    """Re-read a shard, keep only docs whose ordinal is in the per-shard
    keep set (inherited via fork from parent's `keep_per_shard`).
    Writes to OUT_DIR/keep_<src>_<shard_basename>.jsonl.gz.

    Task tuple is tiny (src, path, shard_idx, out_dir) — the big keep-set
    dict is shared via COW fork inheritance, NOT pickled per task.

    Returns (source, read_count, kept_count).
    """
    src, path, shard_idx, out_dir = args
    keep_ordinals = _KEEP_PER_SHARD.get(shard_idx, set()) if _KEEP_PER_SHARD else set()
    out_path = Path(out_dir) / f"keep_{src}_{path.name}"
    n_read = 0
    n_kept = 0
    with gzip.open(path, "rt", encoding="utf-8") as fin, \
         gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6) as fout:
        for ord_, line in enumerate(fin):
            n_read += 1
            if ord_ not in keep_ordinals:
                continue
            fout.write(line)  # already valid JSON
            n_kept += 1
    return src, n_read, n_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+",
                    default=["wikipedia", "gigaword", "wikisource", "fineweb2"])
    ap.add_argument("--num-workers", type=int, default=min(mp.cpu_count() - 8, 128))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUT_DIR.glob("*.jsonl.gz"):
        f.unlink()

    shards = list_shards(args.sources)
    print(f"[cfg] {len(shards)} shards across sources={args.sources}  "
          f"workers={args.num_workers}", flush=True)

    # ---- Phase 1: parallel hash (pool.map preserves input order) ----
    print(f"[phase1] hashing all docs", flush=True)
    t0 = time.time()
    with mp.Pool(min(args.num_workers, max(len(shards), 4))) as pool:
        per_shard_records = pool.map(hash_shard, shards)
    all_records: list[tuple[int, str, int, int]] = []  # (hash, src, shard_idx, ord)
    for i, recs in enumerate(per_shard_records):
        src_i = shards[i][0]
        for h, s, o in recs:
            all_records.append((h, src_i, i, o))
    print(f"[phase1] done. {len(all_records):,} records in "
          f"{(time.time()-t0)/60:.1f}min", flush=True)
    del per_shard_records

    # ---- Phase 2: pick winner per hash ----
    print(f"[phase2] resolving duplicates by source priority", flush=True)
    t0 = time.time()
    # winner: hash -> (priority, shard_input_idx, ord, src)
    winner: dict[int, tuple[int, int, int, str]] = {}
    for h, src, shard_i, ord_ in all_records:
        prio = SOURCE_PRIORITY.get(src, 0)
        cur = winner.get(h)
        if cur is None or (prio, -shard_i, -ord_) > (cur[0], -cur[1], -cur[2]):
            winner[h] = (prio, shard_i, ord_, src)
    n_unique = len(winner)
    n_total = len(all_records)
    n_dup = n_total - n_unique
    print(f"[phase2] {n_total:,} docs → {n_unique:,} unique  "
          f"({n_dup:,} dupes, {100*n_dup/n_total:.1f}%)", flush=True)

    # Per-source keep counts
    per_src_in = defaultdict(int)
    per_src_keep = defaultdict(int)
    for h, src, shard_i, ord_ in all_records:
        per_src_in[src] += 1
    for h, (_, _, _, src) in winner.items():
        per_src_keep[src] += 1
    for src in sorted(per_src_in):
        pin = per_src_in[src]
        pk = per_src_keep[src]
        print(f"  {src}: in={pin:,}  keep={pk:,}  drop={pin-pk:,} "
              f"({100*(pin-pk)/pin:.1f}%)", flush=True)
    print(f"[phase2] done in {(time.time()-t0):.1f}s", flush=True)
    del all_records

    # Build keep_set: per-shard set of ordinals to keep
    print(f"[phase3] building keep-sets", flush=True)
    t0 = time.time()
    keep_per_shard: dict[int, set[int]] = defaultdict(set)
    for _, (_, shard_i, ord_, _) in winner.items():
        keep_per_shard[shard_i].add(ord_)
    del winner
    print(f"[phase3] done in {(time.time()-t0):.1f}s", flush=True)

    # ---- Phase 4: parallel re-read + write filtered shards ----
    print(f"[phase4] writing filtered shards", flush=True)
    t0 = time.time()
    write_tasks = [(src, path, i, str(OUT_DIR))
                   for i, (src, path) in enumerate(shards)]
    # Pass keep_per_shard via `initializer` (fork inheritance, no per-task
    # pickle). Per-task tuple stays tiny — just (src, path, shard_idx, dir).
    with mp.Pool(min(args.num_workers, len(shards) * 2),
                 initializer=_init_writer,
                 initargs=(keep_per_shard,)) as pool:
        totals = defaultdict(lambda: [0, 0])
        for i, (src, n_read, n_kept) in enumerate(pool.imap_unordered(
                filter_and_write_shard, write_tasks, chunksize=1)):
            totals[src][0] += n_read
            totals[src][1] += n_kept
            if (i + 1) % 10 == 0:
                print(f"[phase4]   {i+1}/{len(write_tasks)} shards written  "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[phase4] done in {(time.time()-t0)/60:.1f}min", flush=True)
    for src, (rd, kp) in sorted(totals.items()):
        print(f"  {src}: read={rd:,}  written={kp:,}", flush=True)

    total_out_bytes = sum(p.stat().st_size for p in OUT_DIR.glob("*.jsonl.gz"))
    print(f"[done] total output: {total_out_bytes/1e9:.2f} GB in "
          f"{len(list(OUT_DIR.glob('*.jsonl.gz')))} shards → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
