"""Streaming fetch + normalize Danish pretrain sources → JSONL.gz shards.

For each source:
  - stream from HF (no full-parquet materialization; won't fit for FineWeb-2's 150GB)
  - lang-ID filter (fasttext lid.176) — web crawls have language leakage
  - length filter (>= 200 chars)
  - normalize schema: {text, source, id}
  - write 100k rows per shard to /workspace/work/{source}/shard_XXXXXX.jsonl.gz

Runs one source per invocation; spawn N parallel processes for concurrent
downloads. Progress logged every 10k rows to stderr.

Usage:
    python fetch_danish_sources.py <source>

Sources: wikipedia | gigaword | dynaword | oscar | fineweb2
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

WORK_ROOT = Path("/workspace/work")
SHARD_SIZE = 100_000
MIN_CHARS = 200

SOURCES = {
    "wikipedia": {
        "repo": "wikimedia/wikipedia",
        "config": "20231101.da",
        "text_col": "text",
        "id_col": "id",
    },
    "gigaword": {
        "repo": "danish-foundation-models/danish-gigaword",
        "config": None,
        "text_col": "text",
        "id_col": None,
    },
    "dynaword": {
        "repo": "danish-foundation-models/danish-dynaword",
        "config": "default",
        "text_col": "text",
        "id_col": "id",
    },
    "oscar": {
        "repo": "oscar-corpus/OSCAR-2301",
        "config": "da",
        "text_col": "text",
        "id_col": None,
    },
    "fineweb2": {
        "repo": "HuggingFaceFW/fineweb-2",
        "config": "dan_Latn",
        "text_col": "text",
        "id_col": "id",
    },
}

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_PATH = Path("/workspace/cache/lid.176.bin")


def load_lid():
    if not FASTTEXT_MODEL_PATH.exists():
        FASTTEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[lid] downloading {FASTTEXT_MODEL_URL}", flush=True)
        urllib.request.urlretrieve(FASTTEXT_MODEL_URL, FASTTEXT_MODEL_PATH)
    import fasttext
    return fasttext.load_model(str(FASTTEXT_MODEL_PATH))


def is_danish(lid, text: str, threshold: float = 0.55) -> bool:
    sample = text[:2000].replace("\n", " ")
    labels, probs = lid.predict(sample, k=1)
    return labels[0] == "__label__da" and probs[0] >= threshold


def open_shard(out_dir: Path, worker_id: int, shard_idx: int):
    path = out_dir / f"shard_w{worker_id:02d}_{shard_idx:06d}.jsonl.gz"
    return gzip.open(path, "wt", encoding="utf-8", compresslevel=6)


def list_parquet_shards(repo: str, config: str | None) -> list[str]:
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(repo, files_metadata=True)
    files = [f.rfilename for f in info.siblings if f.rfilename.endswith(".parquet")]
    # Skip HF Dataset auxiliary tables (metadata.parquet in Danish-dynaword)
    files = [f for f in files if not f.endswith("metadata.parquet")]
    # "default" == everything in data/**, no per-config filter
    if config and config != "default":
        files = [f for f in files if config in f]
    files = sorted(files)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=list(SOURCES.keys()))
    ap.add_argument("--limit", type=int, default=None, help="cap rows for testing")
    ap.add_argument("--worker-id", type=int, default=0,
                    help="This worker's index in the pool (for shard sharding).")
    ap.add_argument("--num-workers", type=int, default=1,
                    help="Total workers; each takes shards where idx %% num == worker-id.")
    args = ap.parse_args()

    src = SOURCES[args.source]
    out_dir = WORK_ROOT / args.source
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.source}:{args.worker_id}] loading fasttext lid...", flush=True)
    lid = load_lid()

    from datasets import load_dataset
    if args.num_workers > 1:
        all_shards = list_parquet_shards(src["repo"], src["config"])
        my_shards = [s for i, s in enumerate(all_shards) if i % args.num_workers == args.worker_id]
        print(f"[{args.source}:{args.worker_id}] {len(my_shards)}/{len(all_shards)} shards "
              f"(first: {my_shards[0] if my_shards else 'none'})", flush=True)
        # Data_files as HF hub URIs; datasets resolves via `hf://`
        data_files = [f"hf://datasets/{src['repo']}/{p}" for p in my_shards]
        ds = load_dataset("parquet", data_files=data_files, streaming=True, split="train")
    else:
        print(f"[{args.source}] streaming {src['repo']} config={src['config']}", flush=True)
        kwargs = dict(streaming=True, split="train")
        if src["config"]:
            kwargs["name"] = src["config"]
        ds = load_dataset(src["repo"], **kwargs)

    shard_idx = 0
    n_shard = 0
    n_in = 0
    n_out = 0
    n_lang_drop = 0
    n_short_drop = 0
    t0 = time.time()
    fh = open_shard(out_dir, args.worker_id, shard_idx)

    for row in ds:
        n_in += 1
        text = row.get(src["text_col"])
        if not text or not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < MIN_CHARS:
            n_short_drop += 1
            continue
        if not is_danish(lid, text):
            n_lang_drop += 1
            continue
        rec = {
            "text": text,
            "source": args.source,
            "id": str(row.get(src["id_col"])) if src["id_col"] else f"{args.source}:{n_out}",
        }
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_out += 1
        n_shard += 1
        if n_shard >= SHARD_SIZE:
            fh.close()
            shard_idx += 1
            fh = open_shard(out_dir, args.worker_id, shard_idx)
            n_shard = 0
        if n_in % 10_000 == 0:
            el = time.time() - t0
            rate = n_in / el
            print(f"[{args.source}:{args.worker_id}] in={n_in:,}  out={n_out:,}  "
                  f"drop-short={n_short_drop:,}  drop-lang={n_lang_drop:,}  "
                  f"({rate:.0f}/s elapsed={el/60:.1f}min)", flush=True)
        if args.limit and n_in >= args.limit:
            break

    fh.close()
    print(f"[{args.source}:{args.worker_id}] DONE  in={n_in:,}  out={n_out:,}  "
          f"drop-short={n_short_drop:,}  drop-lang={n_lang_drop:,}  "
          f"shards={shard_idx+1}  time={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
