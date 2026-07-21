"""Recover missing HF map shards for the Danish pretokenize job — v2.

Bulletproof design:
  - Per-doc tokenize (never a batch >1) → no batch memory pressure
  - Stream write to arrow IPC file every 1000 rows → no progress lost on crash
  - Aggressive doc-length cap (50k chars) → skips risky content
  - Try/except per doc → tokenizer Python exceptions logged, not fatal
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/src")

CACHE_DIR = Path(
    "/workspace/cache/hf/datasets/json/default-2a6b3b098102a578/0.0.0/"
    "2752a09ea3d59feeb3ad5ea2af11086f41ecd725fd528c98a89d75f546aba397")
FN_HASH = "88dba6f47fdf8888"
TOTAL_SHARDS = 96
MISSING = [3, 95]
TOTAL_ROWS = 95_183_453
ROWS_PER_SHARD = TOTAL_ROWS // TOTAL_SHARDS  # 991,494

MAX_DOC_CHARS = 50_000

TOKENIZER_PATH = "/workspace/tokenizer_da/tokenizer.json"
BAD_DOCS_LOG = Path("/workspace/logs/pretok_recover_bad_docs.jsonl")
FLUSH_EVERY = 1000


def main():
    from tokenizers import Tokenizer
    from datasets import load_dataset
    import pyarrow as pa
    import pyarrow.ipc as ipc

    print(f"[cfg] missing shards: {MISSING}", flush=True)
    print(f"[cfg] max_doc_chars: {MAX_DOC_CHARS:,}  flush_every: {FLUSH_EVERY}",
          flush=True)

    tok = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"[tok] vocab={tok.get_vocab_size()}", flush=True)
    eos_id = 3

    # Load raw dataset (hits load cache)
    shards = sorted(str(p) for p in Path("/workspace/work/dedup").glob("*.jsonl.gz"))
    print(f"[load] loading {len(shards)} shards", flush=True)
    t0 = time.time()
    ds = load_dataset("json", data_files=shards, split="train", num_proc=96)
    print(f"[load] {len(ds):,} docs in {(time.time()-t0)/60:.1f}min", flush=True)

    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
        ("attention_mask", pa.list_(pa.int8())),
    ])

    for shard_idx in MISSING:
        start = shard_idx * ROWS_PER_SHARD
        end = TOTAL_ROWS if shard_idx == TOTAL_SHARDS - 1 else (shard_idx + 1) * ROWS_PER_SHARD
        n = end - start
        out_path = CACHE_DIR / f"cache-{FN_HASH}_{shard_idx:05d}_of_{TOTAL_SHARDS:05d}.arrow"
        tmp_path = out_path.with_suffix(".arrow.tmp")
        print(f"\n[shard {shard_idx}] rows {start:,}-{end:,} ({n:,} docs) "
              f"→ {out_path.name}", flush=True)

        if out_path.exists():
            print(f"  already exists — skipping", flush=True)
            continue

        # Materialize texts (Python list; drop dataset selection overhead)
        t_mat = time.time()
        all_texts = ds.select(range(start, end))["text"]
        print(f"  materialized {n:,} texts in {time.time()-t_mat:.1f}s "
              f"(total_len={sum(len(t) for t in all_texts)/1e6:.0f}M chars)",
              flush=True)

        # Stream write
        n_kept = 0
        n_skipped_long = 0
        n_skipped_err = 0
        n_tokens = 0
        buf_ids: list[list[int]] = []
        buf_am: list[list[int]] = []

        sink = pa.OSFile(str(tmp_path), "wb")
        writer = ipc.new_stream(sink, schema)

        def flush():
            nonlocal buf_ids, buf_am
            if not buf_ids:
                return
            tbl = pa.table({
                "input_ids": pa.array(buf_ids, type=pa.list_(pa.int32())),
                "attention_mask": pa.array(buf_am, type=pa.list_(pa.int8())),
            })
            writer.write_table(tbl)
            buf_ids = []
            buf_am = []

        t1 = time.time()
        for i, text in enumerate(all_texts):
            if len(text) > MAX_DOC_CHARS:
                n_skipped_long += 1
                continue
            try:
                enc = tok.encode(text, add_special_tokens=False)
                ids = list(enc.ids) + [eos_id]
                buf_ids.append(ids)
                buf_am.append([1] * len(ids))
                n_kept += 1
                n_tokens += len(ids)
            except Exception as e:
                n_skipped_err += 1
                with open(BAD_DOCS_LOG, "a") as fh:
                    fh.write(json.dumps({
                        "shard": shard_idx,
                        "row_offset": i,
                        "text_len": len(text),
                        "text_prefix": text[:400],
                        "err": f"{type(e).__name__}: {e}",
                    }, ensure_ascii=False) + "\n")

            if len(buf_ids) >= FLUSH_EVERY:
                flush()

            if (i + 1) % 50000 == 0:
                el = time.time() - t1
                rate = (i + 1) / el
                eta = (n - i - 1) / rate / 60 if rate else 0
                print(f"  [{i+1:,}/{n:,}] kept={n_kept:,} "
                      f"skipped_long={n_skipped_long} skipped_err={n_skipped_err} "
                      f"({rate:.0f}/s eta {eta:.1f}min)", flush=True)

        flush()
        writer.close()
        sink.close()

        # Atomic rename
        tmp_path.rename(out_path)
        print(f"  wrote {out_path.name} ({out_path.stat().st_size/1e6:.1f}MB, "
              f"{n_kept:,} kept, {n_tokens/1e6:.1f}M toks, "
              f"skipped_long={n_skipped_long} skipped_err={n_skipped_err}) "
              f"in {(time.time()-t1)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
