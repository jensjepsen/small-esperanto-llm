"""Convert the 94 already-completed HF map cache arrow shards to parquet
and push to HF as `jensjepsen/danish-pretokenized-16k` (v1 with 2 shards
missing — dropped for the missing shards 3 and 95).

Runs in parallel with pretok_recover_shards.py which is filling shards 3
and 95. Once recovery completes, a v1.1 supplement dataset can be pushed
separately (jensjepsen/danish-pretokenized-16k-supp) with the recovered
rows and concatenated at load time via datasets.concatenate_datasets.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import time
from pathlib import Path

CACHE_DIR = Path(
    "/workspace/cache/hf/datasets/json/default-2a6b3b098102a578/0.0.0/"
    "2752a09ea3d59feeb3ad5ea2af11086f41ecd725fd528c98a89d75f546aba397")
OUT_DIR = Path("/workspace/work/pretokenized_parquet")


def convert_shard(args) -> tuple[str, int, int]:
    """Read one HF map cache arrow file, write parquet with zstd.

    Returns (out_name, n_rows, out_bytes).
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq

    arrow_path, out_path = args
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path.name, -1, out_path.stat().st_size  # -1 = skipped
    with pa.OSFile(str(arrow_path), "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()
    pq.write_table(table, out_path, compression="zstd", compression_level=6)
    return out_path.name, table.num_rows, out_path.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--push-repo", default="jensjepsen/danish-pretokenized-16k")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all completed arrow shards
    pattern = re.compile(r"cache-(\w+)_(\d{5})_of_(\d{5})\.arrow$")
    arrows = []
    for f in sorted(CACHE_DIR.glob("cache-*_of_00096.arrow")):
        m = pattern.search(f.name)
        if m:
            shard_idx = int(m.group(2))
            out_path = OUT_DIR / f"data-{shard_idx:05d}.parquet"
            arrows.append((f, out_path, shard_idx))

    print(f"[cfg] found {len(arrows)} arrow shards → parquet", flush=True)
    tasks = [(a, o) for a, o, _ in arrows]

    print(f"[convert] converting arrow → parquet with {args.num_workers} workers",
          flush=True)
    t0 = time.time()
    total_rows = 0
    total_bytes = 0
    n_skipped = 0
    with mp.Pool(args.num_workers) as pool:
        for i, (name, n_rows, size) in enumerate(pool.imap_unordered(
                convert_shard, tasks, chunksize=1)):
            if n_rows == -1:
                n_skipped += 1
            else:
                total_rows += n_rows
            total_bytes += size
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{len(tasks)}] docs={total_rows:,} "
                      f"parquet={total_bytes/1e9:.1f}GB skipped={n_skipped} "
                      f"({el/60:.1f}min)", flush=True)
    print(f"[convert] done. {total_rows:,} docs, "
          f"{total_bytes/1e9:.1f}GB parquet in {(time.time()-t0)/60:.1f}min",
          flush=True)

    if args.no_push:
        return

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN not set")

    (OUT_DIR / "README.md").write_text(f"""---
license: cc-by-4.0
language:
- da
tags:
- pretokenized
---

# Danish Pretrain — pretokenized (byte-level BPE, 16k vocab) — v1 partial

Pretokenized version of [jensjepsen/danish-pretrain](https://huggingface.co/datasets/jensjepsen/danish-pretrain)
via [jensjepsen/danish-tokenizer](https://huggingface.co/jensjepsen/danish-tokenizer).

**Note:** This is a partial version — 94 of 96 shards. The 2 missing shards
crashed during tokenization (rare pathological content). See
`jensjepsen/danish-pretokenized-16k-supp` for the recovered rows.

- {total_rows:,} docs
- Schema: `input_ids: list<int32>`, `attention_mask: list<int8>`
- {len(tasks)} zstd-compressed parquet shards
""")

    print(f"[push] uploading to {args.push_repo}", flush=True)
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.push_repo, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(
        folder_path=str(OUT_DIR),
        repo_id=args.push_repo,
        repo_type="dataset",
        num_workers=8,
    )
    print(f"[push] done — https://huggingface.co/datasets/{args.push_repo}",
          flush=True)


if __name__ == "__main__":
    main()
