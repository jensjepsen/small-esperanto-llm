"""GPQA-Diamond DA eval: 4-way MC via generation + letter parse.

Reads local JSONL from translate_gpqa_diamond_or.py output (default in schema:
`answers_da[0]` = correct, rest = distractors). Shuffles A/B/C/D per-row with
deterministic seed. Chat-wraps prompt (matches cit_mc style), generates 8
tokens, parses first A/B/C/D letter.

Usage:
    uv run python scripts/eval_gpqa_da.py --ckpt HF_ID_OR_LOCAL \\
        --data data/danish_gpqa_diamond/da_gpqa_diamond.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
LETTER_RE = re.compile(r"\b([ABCD])\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dtype", default="fp32", choices=["fp32","fp16","bf16"])
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-every", type=int, default=25)
    args = ap.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=dtype).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id] + ([end_id] if end_id != tok.unk_token_id else [])

    if args.data.endswith(".jsonl") or args.data.startswith("/") or args.data.startswith("."):
        rows = [json.loads(l) for l in open(args.data)]
    else:
        from datasets import load_dataset
        rows = list(load_dataset(args.data, split="train"))
    print(f"  {len(rows)} rows", flush=True)

    n_ok = 0
    n_parsefail = 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        q = r["question_da"]
        answers = list(r["answers_da"])  # [correct, w1, w2, w3]
        # Shuffle deterministic per orig_idx
        rng = random.Random(args.seed + r["orig_idx"])
        idxs = list(range(4))
        rng.shuffle(idxs)
        letters = "ABCD"
        opts_lines = []
        gold_letter = None
        for slot, orig in enumerate(idxs):
            opts_lines.append(f"{letters[slot]}) {answers[orig]}")
            if orig == 0:  # correct answer
                gold_letter = letters[slot]

        body = (f"{q}\n\n" + "\n".join(opts_lines) +
                "\n\nSvar med bogstavet på det korrekte svar.")
        prompt = f"{USER}{body}{END}{ASST}"
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False,
                  return_token_type_ids=False).input_ids.cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=eos_ids)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        m = LETTER_RE.search(gen)
        if not m:
            n_parsefail += 1
            pred = "?"
        else:
            pred = m.group(1)
        if pred == gold_letter:
            n_ok += 1
        if i % args.report_every == 0 or i == len(rows):
            el = time.time() - t0
            eta = el * (len(rows) - i) / i
            print(f"  {i}/{len(rows)}  acc={n_ok/i:.3f}  parsefail={n_parsefail}  eta={eta:.0f}s",
                  flush=True)

    print(f"\n=== gpqa-diamond-da  n={len(rows)}  acc={100*n_ok/len(rows):.2f}%  "
          f"({n_ok}/{len(rows)})  parsefail={n_parsefail}  random={100/4:.1f}% ===",
          flush=True)


if __name__ == "__main__":
    main()
