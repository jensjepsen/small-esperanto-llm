"""Pretokenize the Danish synthetic math datasets (algebra + arith) with the
16k byte-level BPE tokenizer, push to HF as `jensjepsen/danish-math-pretokenized-16k`.

Schema matches jensjepsen/danish-pretokenized-16k so both can be loaded via
concatenate_datasets: `input_ids: list<int32>`, `attention_mask: list<int8>`.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

INPUTS = [
    ("/tmp/da_math/algebra_pretrain_v2.jsonl", "algebra"),
    ("/tmp/da_math/arith_chain_v2.jsonl", "arith"),
]
OUT_DIR = Path("/tmp/da_math/pretokenized")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

_TOK = None
_EOS_ID = 3


def _init_worker(tokenizer_path: str):
    global _TOK
    from tokenizers import Tokenizer
    _TOK = Tokenizer.from_file(tokenizer_path)


def tokenize_chunk(args) -> tuple[int, list[list[int]], list[list[int]]]:
    """Tokenize a chunk of (index, texts). Returns (chunk_idx, ids, masks)."""
    chunk_idx, texts = args
    encs = _TOK.encode_batch(texts, add_special_tokens=False)
    ids = []
    masks = []
    for enc in encs:
        i = list(enc.ids) + [_EOS_ID]
        ids.append(i)
        masks.append([1] * len(i))
    return chunk_idx, ids, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-repo", default="jensjepsen/danish-tokenizer")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--push-repo", default="jensjepsen/danish-math-pretokenized-16k")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=10000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download tokenizer
    from huggingface_hub import hf_hub_download
    tok_path = hf_hub_download(args.tokenizer_repo, "tokenizer.json", repo_type="model")
    print(f"[cfg] tokenizer={tok_path}  num_workers={args.num_workers}", flush=True)

    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
        ("attention_mask", pa.list_(pa.int8())),
    ])

    for jsonl_path, label in INPUTS:
        print(f"\n[{label}] reading {jsonl_path}", flush=True)
        t0 = time.time()
        texts = []
        with open(jsonl_path) as f:
            for line in f:
                try:
                    texts.append(json.loads(line)["text"])
                except Exception:
                    continue
        print(f"[{label}] loaded {len(texts):,} texts in {time.time()-t0:.1f}s",
              flush=True)

        # Chunk for parallel tokenize
        chunks = []
        for i in range(0, len(texts), args.chunk_size):
            chunks.append((i, texts[i:i + args.chunk_size]))
        print(f"[{label}] {len(chunks)} chunks × {args.chunk_size} rows",
              flush=True)

        # Tokenize in parallel, preserve order
        t1 = time.time()
        results = [None] * len(chunks)
        n_total = 0
        n_tokens = 0
        with mp.Pool(args.num_workers, initializer=_init_worker,
                     initargs=(tok_path,)) as pool:
            for chunk_idx, ids, masks in pool.imap_unordered(tokenize_chunk, chunks):
                results[chunk_idx // args.chunk_size] = (ids, masks)
                n_total += len(ids)
                n_tokens += sum(len(i) for i in ids)
                if (chunk_idx // args.chunk_size + 1) % 20 == 0:
                    el = time.time() - t1
                    print(f"  {n_total:,}/{len(texts):,} docs "
                          f"({n_total/el:.0f}/s)", flush=True)
        print(f"[{label}] tokenized {n_total:,} docs, "
              f"{n_tokens/1e6:.1f}M tokens in {(time.time()-t1)/60:.1f}min",
              flush=True)

        # Flatten and write parquet
        all_ids = []
        all_masks = []
        for ids, masks in results:
            all_ids.extend(ids)
            all_masks.extend(masks)
        out_path = OUT_DIR / f"{label}.parquet"
        tbl = pa.table({
            "input_ids": pa.array(all_ids, type=pa.list_(pa.int32())),
            "attention_mask": pa.array(all_masks, type=pa.list_(pa.int8())),
        })
        pq.write_table(tbl, out_path, compression="zstd", compression_level=6)
        print(f"[{label}] wrote {out_path.name} "
              f"({out_path.stat().st_size/1e6:.1f}MB)", flush=True)

    if args.no_push:
        return

    if not os.environ.get("HF_TOKEN"):
        # Try cache
        tok_file = Path.home() / ".cache/huggingface/token"
        if tok_file.exists():
            os.environ["HF_TOKEN"] = tok_file.read_text().strip()
        else:
            raise SystemExit("HF_TOKEN not set")

    (OUT_DIR / "README.md").write_text(f"""---
license: cc-by-4.0
language:
- da
tags:
- pretokenized
- math
- synthetic
---

# Danish Math Pretrain — pretokenized (byte-level BPE, 16k vocab)

Pretokenized version of Danish synthetic math datasets:
- [jensjepsen/danish-algebra-pretrain](https://huggingface.co/datasets/jensjepsen/danish-algebra-pretrain) (5.4M rows)
- [jensjepsen/danish-arith-chain](https://huggingface.co/datasets/jensjepsen/danish-arith-chain) (1.8M rows)

Tokenized with [jensjepsen/danish-tokenizer](https://huggingface.co/jensjepsen/danish-tokenizer).

Schema matches [jensjepsen/danish-pretokenized-16k](https://huggingface.co/datasets/jensjepsen/danish-pretokenized-16k):
`input_ids: list<int32>`, `attention_mask: list<int8>`.

Concatenate for training:
```python
from datasets import load_dataset, concatenate_datasets
main = load_dataset('jensjepsen/danish-pretokenized-16k', split='train')
supp = load_dataset('jensjepsen/danish-pretokenized-16k-supp', split='train')
math = load_dataset('jensjepsen/danish-math-pretokenized-16k', split='train')
full = concatenate_datasets([main, supp, math])
```
""")

    print(f"\n[push] uploading to {args.push_repo}", flush=True)
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.push_repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(OUT_DIR),
        repo_id=args.push_repo,
        repo_type="dataset",
    )
    print(f"[push] done — https://huggingface.co/datasets/{args.push_repo}",
          flush=True)


if __name__ == "__main__":
    main()
