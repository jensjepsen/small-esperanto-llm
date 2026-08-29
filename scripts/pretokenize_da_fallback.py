"""Fallback pretokenize: bypass HF datasets.map — write our own arrow shards.

Use when the main pretokenize_da_corpus.py path dies with "subprocess died"
on a pathological doc. This script:

  1. Loads the raw dataset (hits HF's load cache — fast).
  2. For each of N output shards, either:
     a. If our own shard file already exists AND is valid arrow → skip.
     b. Else → tokenize that shard's row range with batch tokenizer, and
        on batch failure fall back to per-doc try/except (logs the failing
        doc and skips it).
  3. Concat all shards into one HF Dataset.
  4. Push to HF.

Each worker owns one shard end-to-end. Bad docs are logged to
/workspace/logs/pretok_bad_docs.jsonl.
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/workspace/src")

WORK = Path("/workspace/work")
DEDUP_DIR = WORK / "dedup"
OUT_DIR = WORK / "pretokenized_fallback"
TOKENIZER_PATH = "/workspace/tokenizer_da/tokenizer.json"
BAD_DOCS_LOG = Path("/workspace/logs/pretok_bad_docs.jsonl")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_TOK = None


def _init_worker(tokenizer_path: str):
    global _TOK
    from tokenizers import Tokenizer
    _TOK = Tokenizer.from_file(tokenizer_path)


def _tokenize_batch_safe(texts: list[str], sources: list[str], worker_id: int,
                          bad_docs_path: str) -> tuple[list[list[int]], list[str], int]:
    """Try batch first (fast). On any failure, fall back to per-doc, catch
    exceptions, log bad docs. Returns (ids_list, srcs_list, n_bad)."""
    try:
        encs = _TOK.encode_batch(texts, add_special_tokens=False)
        return [e.ids for e in encs], sources, 0
    except Exception as batch_err:
        # Fall back to per-doc
        ids_out = []
        srcs_out = []
        n_bad = 0
        for t, s in zip(texts, sources):
            try:
                enc = _TOK.encode(t, add_special_tokens=False)
                ids_out.append(enc.ids)
                srcs_out.append(s)
            except Exception as doc_err:
                n_bad += 1
                with open(bad_docs_path, "a") as fh:
                    fh.write(json.dumps({
                        "worker": worker_id,
                        "text_prefix": t[:500],
                        "text_len": len(t),
                        "src": s,
                        "err": f"{type(doc_err).__name__}: {doc_err}",
                        "batch_err": f"{type(batch_err).__name__}: {batch_err}",
                    }, ensure_ascii=False) + "\n")
        return ids_out, srcs_out, n_bad


def tokenize_shard(args) -> tuple[str, int, int, int]:
    """Tokenize a row range from all shards concatenated. Writes one parquet.

    Returns (shard_name, n_docs, n_tokens, n_bad).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_paths, worker_id, out_dir, batch_size = args
    out_path = Path(out_dir) / f"shard_{worker_id:04d}.parquet"
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            n_rows = pq.ParquetFile(out_path).metadata.num_rows
            return out_path.name, n_rows, -1, 0  # -1 tokens = skipped
        except Exception:
            out_path.unlink()  # corrupt, redo

    schema = pa.schema([
        ("input_ids", pa.list_(pa.uint16())),
        ("source", pa.string()),
    ])

    n_docs = 0
    n_tokens = 0
    n_bad = 0
    writer = pq.ParquetWriter(out_path, schema, compression="zstd",
                              compression_level=6)

    def flush(texts, srcs):
        nonlocal n_docs, n_tokens, n_bad
        if not texts:
            return
        ids, srcs_kept, bad = _tokenize_batch_safe(
            texts, srcs, worker_id, str(BAD_DOCS_LOG))
        n_bad += bad
        n_docs += len(ids)
        n_tokens += sum(len(x) for x in ids)
        writer.write_table(pa.table({
            "input_ids": pa.array(ids, type=pa.list_(pa.uint16())),
            "source": pa.array(srcs_kept, type=pa.string()),
        }))

    try:
        texts_buf: list[str] = []
        srcs_buf: list[str] = []
        for shard_path in shard_paths:
            src = Path(shard_path).name.removeprefix("keep_").split("_shard_")[0]
            with gzip.open(shard_path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    texts_buf.append(r["text"])
                    srcs_buf.append(r.get("source", src))
                    if len(texts_buf) >= batch_size:
                        flush(texts_buf, srcs_buf)
                        texts_buf = []
                        srcs_buf = []
        flush(texts_buf, srcs_buf)
    finally:
        writer.close()

    return out_path.name, n_docs, n_tokens, n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-workers", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--push-repo", default="jensjepsen/danish-pretokenized-16k")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BAD_DOCS_LOG.parent.mkdir(parents=True, exist_ok=True)

    all_shards = sorted(str(p) for p in DEDUP_DIR.glob("*.jsonl.gz"))
    print(f"[cfg] {len(all_shards)} input shards, {args.num_workers} workers, "
          f"batch_size={args.batch_size}", flush=True)

    # Split input shards evenly across workers
    per_worker = (len(all_shards) + args.num_workers - 1) // args.num_workers
    tasks = []
    for w in range(args.num_workers):
        assigned = all_shards[w*per_worker : (w+1)*per_worker]
        if not assigned:
            break
        tasks.append((assigned, w, str(OUT_DIR), args.batch_size))
    print(f"[cfg] {len(tasks)} tasks, ~{per_worker} input shards each", flush=True)

    print(f"[phase1] tokenizing (resume on existing shards)", flush=True)
    t0 = time.time()
    n_docs_total = 0
    n_tokens_total = 0
    n_bad_total = 0
    n_skipped = 0
    with mp.Pool(args.num_workers, initializer=_init_worker,
                 initargs=(TOKENIZER_PATH,)) as pool:
        for i, (name, n_docs, n_tokens, n_bad) in enumerate(
                pool.imap_unordered(tokenize_shard, tasks, chunksize=1)):
            if n_tokens == -1:
                n_skipped += 1
                print(f"  [skip cached] {name}: {n_docs:,} rows", flush=True)
            else:
                n_docs_total += n_docs
                n_tokens_total += n_tokens
                n_bad_total += n_bad
            if (i + 1) % 5 == 0:
                el = time.time() - t0
                done_frac = (i + 1) / len(tasks)
                eta = el / done_frac * (1 - done_frac) / 60
                print(f"  {i+1}/{len(tasks)} tasks  "
                      f"docs={n_docs_total:,}  toks={n_tokens_total/1e9:.2f}B  "
                      f"skipped_cached={n_skipped}  bad_docs={n_bad_total}  "
                      f"({el/60:.1f}min, eta {eta:.1f}min)", flush=True)
    print(f"[phase1] done. new_docs={n_docs_total:,}  "
          f"new_toks={n_tokens_total/1e9:.2f}B  "
          f"skipped_cached={n_skipped}  bad_docs={n_bad_total}  "
          f"in {(time.time()-t0)/60:.1f}min", flush=True)

    if n_bad_total > 0:
        print(f"[phase1] BAD DOCS logged to {BAD_DOCS_LOG}", flush=True)

    total_bytes = sum(p.stat().st_size for p in OUT_DIR.glob("*.parquet"))
    print(f"[out] {total_bytes/1e9:.2f} GB parquet in {OUT_DIR}", flush=True)

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

# Danish Pretrain — pretokenized (byte-level BPE, 16k vocab)

From [jensjepsen/danish-pretrain](https://huggingface.co/datasets/jensjepsen/danish-pretrain)
via [jensjepsen/danish-tokenizer](https://huggingface.co/jensjepsen/danish-tokenizer).

Schema: `input_ids: list<uint16>`, `source: str`.
Bad docs (that crashed the tokenizer) were logged and skipped —
count in provenance details of the training run.
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
