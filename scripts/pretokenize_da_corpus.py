"""Pretokenize the Danish corpus using the SAME code path train.py uses.

Mirrors train.py's data-prep exactly:
    - load_dataset("json", data_files=[...])
    - tokenize_and_chunk(dataset, tokenizer, max_length=2048,
                         morpheme_preprocess=False)  # False for Danish

Then push tokenized dataset to HF via ds.push_to_hub (arrow → parquet upload).
Uses ESPLLM_NUM_PROC to control parallelism (data.py reads this).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make esperanto_lm importable when running from /workspace/scripts
sys.path.insert(0, "/workspace/src")

WORK = Path("/workspace/work")
DEDUP_DIR = WORK / "dedup"
TOKENIZER_PATH = Path("/workspace/tokenizer_da")  # dir with tokenizer.json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push-repo", default="jensjepsen/danish-pretokenized-16k")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    # Import AFTER env is set so ESPLLM_NUM_PROC is picked up
    from esperanto_lm.data import num_proc, tokenize_dataset
    from transformers import PreTrainedTokenizerFast
    from datasets import load_dataset

    print(f"[cfg] ESPLLM_NUM_PROC={os.environ.get('ESPLLM_NUM_PROC', 'auto')}  "
          f"→ num_proc()={num_proc()}", flush=True)

    # Load tokenizer (byte-level BPE saved as single tokenizer.json)
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_PATH / "tokenizer.json"),
        pad_token="<pad>", unk_token="<unk>",
        bos_token="<s>", eos_token="</s>",
    )
    print(f"[tok] vocab_size={tok.vocab_size}  eos_id={tok.eos_token_id}",
          flush=True)

    # Load raw JSONL.gz shards as one dataset
    shards = sorted(str(p) for p in DEDUP_DIR.glob("*.jsonl.gz"))
    print(f"[load] {len(shards)} JSONL.gz shards", flush=True)
    t0 = time.time()
    ds = load_dataset("json", data_files=shards, split="train",
                      num_proc=num_proc())
    print(f"[load] {len(ds):,} docs in {(time.time()-t0)/60:.1f}min",
          flush=True)

    # Tokenize only — chunk_dataset runs at train time (train.py's
    # --pretokenized-dataset flag path). morpheme_preprocess=False for Danish.
    print(f"[tokenize] running tokenize_dataset (train.py's exact code path)",
          flush=True)
    t1 = time.time()
    ds_tok = tokenize_dataset(ds, tok, morpheme_preprocess=False)
    print(f"[tokenize] {len(ds_tok):,} tokenized docs in "
          f"{(time.time()-t1)/60:.1f}min", flush=True)

    if args.no_push:
        return

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN not set")

    print(f"[push] pushing to {args.push_repo}", flush=True)
    ds_tok.push_to_hub(args.push_repo, token=os.environ["HF_TOKEN"])
    print(f"[push] done — https://huggingface.co/datasets/{args.push_repo}",
          flush=True)


if __name__ == "__main__":
    main()
