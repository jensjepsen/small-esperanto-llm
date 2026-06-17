"""Pass@1 GSM8K eval for LFM2.5-350M on GPU with batched generation."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch


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
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--n", type=int, default=0, help="0 = full test set (1319)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--out", default="mt/runs/lfm25_gsm8k_gpu_pass1.jsonl")
    ap.add_argument("--hf-cache", default="/mnt/data/hf_cache")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_cache)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading {args.model} on cuda fp16…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"            # causal LM needs left padding
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    print(f"  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  GPU mem={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    N = len(ds) if not args.n else min(args.n, len(ds))
    sys_msg = "You are a careful math tutor. Solve step by step, then give the final answer as: #### <number>"

    # Pre-format all prompts via chat template
    prompts = []
    golds = []
    for i in range(N):
        q = ds[i]["question"]
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}]
        prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
        golds.append(gold_num(ds[i]["answer"]))

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    correct = 0
    seen = 0
    t0 = time.perf_counter()
    n_gen_tokens = 0
    with open(out_path, "w") as fout:
        for batch_start in range(0, N, args.batch_size):
            batch_prompts = prompts[batch_start : batch_start + args.batch_size]
            batch_golds = golds[batch_start : batch_start + args.batch_size]
            enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=False).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            in_len = enc["input_ids"].shape[1]
            gen_tokens = out[:, in_len:]
            n_gen_tokens += int((gen_tokens != tok.pad_token_id).sum().item())
            texts = tok.batch_decode(gen_tokens, skip_special_tokens=True)
            for j, (txt, gold) in enumerate(zip(texts, batch_golds)):
                pred = pred_num(txt)
                ok = pred == gold
                correct += ok
                seen += 1
                fout.write(json.dumps({"i": batch_start + j, "gold": gold, "pred": pred, "ok": ok, "text": txt}) + "\n")
            dt = time.perf_counter() - t0
            print(f"  batch {batch_start//args.batch_size + 1}/{(N+args.batch_size-1)//args.batch_size}  "
                  f"{seen}/{N}  pass@1={correct} ({100*correct/seen:.1f}%)  "
                  f"{n_gen_tokens/dt:.0f} tok/s  elapsed={dt:.0f}s", flush=True)

    print(f"\n=== final: pass@1 = {correct}/{N} = {100*correct/N:.2f}% in {dt:.0f}s ({n_gen_tokens/dt:.0f} tok/s) ===")


if __name__ == "__main__":
    main()
