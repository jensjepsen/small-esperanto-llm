"""Pass@k eval for LFM2.5-350M Q4_K_M on GSM8K via llama-cpp-python."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time


GOLD_RE = re.compile(r'####\s*([-\d,\.]+)')
BOXED_RE = re.compile(r'\\boxed\{\s*\$?\s*([\-]?\d+(?:\.\d+)?)\s*\}')
HASH_NUM_RE = re.compile(r'#{2,}\s*\$?([\-]?\d+(?:\.\d+)?)(?![.\d]*\w)')
ANS_RE = re.compile(r'(?:final answer|answer)[:\s\*]+\$?([\-]?\d+(?:\.\d+)?)', re.IGNORECASE)


def gold_num(answer: str) -> str | None:
    m = GOLD_RE.search(answer)
    return m.group(1).replace(',', '') if m else None


def pred_num(text: str) -> str | None:
    t = text.replace(',', '')
    m = BOXED_RE.search(t)
    if m:
        return m.group(1)
    last = None
    for last in HASH_NUM_RE.finditer(t):
        pass
    if last:
        return last.group(1)
    m = ANS_RE.search(t)
    if m:
        return m.group(1)
    nums = re.findall(r'-?\d+(?:\.\d+)?', t[-200:])
    return nums[-1] if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=3, help="pass@k attempts per question")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--out", default="mt/runs/lfm25_gsm8k_pass.jsonl")
    ap.add_argument("--hf-cache", default="/mnt/data/hf_cache")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_cache)
    from llama_cpp import Llama
    from datasets import load_dataset

    llm = Llama(model_path=args.gguf, n_ctx=2048, n_threads=16, verbose=False, seed=0)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    sys_msg = "You are a careful math tutor. Solve step by step, then give the final answer as: #### <number>"

    pass1 = pass_k = 0
    records = []
    t0 = time.perf_counter()
    for i in range(args.n):
        q = ds[i]['question']
        a_gold = gold_num(ds[i]['answer'])
        preds, texts = [], []
        for k in range(args.k):
            out = llm.create_chat_completion(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=k + 1,
            )
            txt = out["choices"][0]["message"]["content"]
            preds.append(pred_num(txt))
            texts.append(txt)
        p1 = preds[0] == a_gold
        pk = any(p == a_gold for p in preds)
        pass1 += p1
        pass_k += pk
        records.append({"i": i, "gold": a_gold, "preds": preds, "texts": texts, "pass1": p1, f"pass{args.k}": pk})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n}  pass@1={pass1}  pass@{args.k}={pass_k}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    dt = time.perf_counter() - t0
    print(f"\n=== {args.n} questions × {args.k} samples in {dt:.0f}s ===")
    print(f"pass@1 (temp={args.temperature}): {pass1}/{args.n} = {100*pass1/args.n:.1f}%")
    print(f"pass@{args.k} (temp={args.temperature}): {pass_k}/{args.n} = {100*pass_k/args.n:.1f}%")

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
