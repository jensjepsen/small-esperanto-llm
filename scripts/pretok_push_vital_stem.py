"""Pretokenize vital-STEM DA translations and push both raw + tokenized to HF.

Raw:        jensjepsen/danish-vital-stem-da-v1     (page_id, title, text)
Tokenized:  jensjepsen/danish-vital-stem-da-tokenized-v1  (input_ids, attention_mask)
            per-doc, </s> appended (matches jensjepsen/danish-pretokenized-16k format)
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

# Make esperanto_lm importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/data2/wiki_gaps/vital_stem_da_sync.jsonl")
    ap.add_argument("--tokenizer", default="jensjepsen/danish-tokenizer")
    ap.add_argument("--raw-repo", default="jensjepsen/danish-vital-stem-da-v1")
    ap.add_argument("--tok-repo", default="jensjepsen/danish-vital-stem-da-tokenized-v1")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("ESPLLM_NUM_PROC", "8")
    from datasets import Dataset
    from transformers import AutoTokenizer
    from esperanto_lm.data import tokenize_dataset

    # Load jsonl
    rows = []
    for line in open(args.src):
        r = json.loads(line)
        if not r.get("da"): continue
        rows.append({
            "page_id": int(r["page_id"]),
            "title": r.get("title") or "",
            "text": r["da"],
        })
    print(f"[load] {len(rows):,} rows  total chars: {sum(len(r['text']) for r in rows):,}",
          flush=True)

    raw = Dataset.from_list(rows)
    if not args.no_push:
        print(f"[push] raw → {args.raw_repo}", flush=True)
        raw.push_to_hub(args.raw_repo, split="train",
                        commit_message=f"vital-5 STEM DA translations ({len(rows)} articles)")

    # Tokenize (no morpheme, no chunking — matches ropext pipeline)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"[tok] {tok.__class__.__name__}  vocab={tok.vocab_size}  eos={tok.eos_token_id}",
          flush=True)
    tokenized = tokenize_dataset(raw, tok, morpheme_preprocess=False)
    total_tokens = sum(len(x["input_ids"]) for x in tokenized)
    print(f"[tok] tokenized {len(tokenized):,} rows  total tokens: {total_tokens:,}",
          flush=True)

    if not args.no_push:
        print(f"[push] tokenized → {args.tok_repo}", flush=True)
        tokenized.push_to_hub(args.tok_repo, split="train",
                              commit_message=f"vital-5 STEM DA tokenized "
                                             f"({len(rows)} articles, {total_tokens:,} tokens)")

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
