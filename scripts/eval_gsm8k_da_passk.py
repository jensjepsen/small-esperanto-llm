"""GSM8K pass@k eval on Danish. K samples per question (temperature > 0),
correct if any of the K samples matches gold. Streams per-row JSONL and
prints running acc every N questions.

Usage:
    python scripts/eval_gsm8k_da_passk.py --ckpt PATH --out out.jsonl \\
        --n 200 --k 12 --temp 0.8 --report-every 100
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
NUM_RE = re.compile(r"####\s*(-?\d[\d,\.]*)")
LAST_NUM_RE = re.compile(r"(-?\d[\d,]*\.?\d*)")


def extract_num(text: str):
    m = NUM_RE.search(text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = LAST_NUM_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").rstrip(".")
    return None


import math
def norm(s):
    if s is None: return None
    try:
        f = float(s)
        if not math.isfinite(f):
            return s
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError, OverflowError):
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="jensjepsen/danish-gsm8k")
    ap.add_argument("--config", default="sft")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Rows per generate() call. Effective batch on GPU = "
                         "batch_size * k. On 5090/32GB with a 400M model, "
                         "batch_size=4 and k=12 → 48 sequences fits comfortably.")
    args = ap.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for generation with batched prompts
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=dtype).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id]
    if end_id != tok.unk_token_id and end_id is not None:
        eos_ids.append(end_id)

    ds = load_dataset(args.dataset, args.config, split=args.split)
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))
    n = len(ds)
    print(f"  {n} rows, k={args.k}, temp={args.temp}, top_p={args.top_p}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok_any = 0                # pass@k
    n_ok_first = 0              # pass@1 (first sample)
    total_correct_samples = 0   # for maj-vote style stats
    t0 = time.time()

    rows_list = list(ds)
    with open(args.out, "w") as f:
        for batch_start in range(0, n, args.batch_size):
            batch = rows_list[batch_start:batch_start + args.batch_size]
            B = len(batch)

            prompts = [f"{USER} {r['messages'][0]['content']} {ASST}" for r in batch]
            golds = [extract_num(r['messages'][1]['content']) for r in batch]

            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=False, return_token_type_ids=False).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    max_new_tokens=args.max_new,
                    do_sample=True, temperature=args.temp, top_p=args.top_p,
                    num_return_sequences=args.k,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    repetition_penalty=1.05, eos_token_id=eos_ids,
                )
            # out shape: (B * k, plen + max_new). Left-padded → new tokens
            # start at column plen for every row.
            plen = enc["input_ids"].shape[1]

            for row_ix in range(B):
                gold_num = golds[row_ix]
                q = batch[row_ix]['messages'][0]['content']
                gens, preds, corrects = [], [], []
                for sample_ix in range(args.k):
                    seq_ix = row_ix * args.k + sample_ix
                    gen = tok.decode(out[seq_ix, plen:], skip_special_tokens=True).strip()
                    pred = extract_num(gen)
                    ok = norm(pred) == norm(gold_num)
                    gens.append(gen); preds.append(pred); corrects.append(bool(ok))
                n_correct_this_q = sum(corrects)
                n_ok_any += int(n_correct_this_q > 0)
                n_ok_first += int(corrects[0])
                total_correct_samples += n_correct_this_q

                i = batch_start + row_ix + 1
                f.write(json.dumps({
                    "idx": i - 1, "q": q, "gold_num": gold_num,
                    "preds": preds, "corrects": corrects, "gens": gens,
                }, ensure_ascii=False) + "\n")
            f.flush()

            i = batch_start + B
            if (i // args.report_every) != ((i - B) // args.report_every) or i == n:
                el = time.time() - t0
                eta = el * (n - i) / i
                mean_correct = total_correct_samples / (i * args.k)
                print(f"  {i:4d}/{n}  pass@{args.k}={n_ok_any/i:.3f} "
                      f"({n_ok_any}/{i})  pass@1={n_ok_first/i:.3f} "
                      f"({n_ok_first}/{i})  avg_correct/q={mean_correct:.3f}  "
                      f"eta={eta:.0f}s", flush=True)

    print(f"\n=== v5 gsm8k[da] n={n} k={args.k} ===")
    print(f"  pass@{args.k}  = {100*n_ok_any/n:.2f}%  ({n_ok_any}/{n})")
    print(f"  pass@1        = {100*n_ok_first/n:.2f}%  ({n_ok_first}/{n})")
    print(f"  avg_correct/q = {100*total_correct_samples/(n*args.k):.2f}%")


if __name__ == "__main__":
    main()
