"""Standalone A/B/C MC eval on alexandrainst/danish-citizen-tests.

Mirrors the DownstreamEvalCallback._score_citmc logic exactly so
in-training numbers and post-hoc numbers are directly comparable.

Usage:
    python scripts/eval_cit_da_mc.py CKPT [--batch-size 32]
"""
from __future__ import annotations

import argparse
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokenizer", default=None,
                    help="Defaults to --ckpt (post-SFT ckpts embed the chat tokens).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.float16).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id]
    if end_id is not None and end_id != tok.unk_token_id:
        eos_ids.append(end_id)

    ds = load_dataset("alexandrainst/danish-citizen-tests", split="train")
    items = []
    for r in ds:
        gold_letter = r.get("answer")
        if not gold_letter:
            continue
        gold_letter = gold_letter.upper()
        opts = {}
        for ll in ["a", "b", "c", "d"]:
            val = r.get(f"option_{ll}")
            if val:
                opts[ll.upper()] = val
        if len(opts) < 2 or gold_letter not in opts:
            continue
        items.append((r["question"], opts, gold_letter))
    print(f"citmc rows kept: {len(items)}", flush=True)

    prompts = []
    for q, opts, _ in items:
        opts_str = "\n".join(f"{lab}) {opts[lab]}" for lab in sorted(opts))
        body = (f"{q}\n\n{opts_str}\n\n"
                f"Svar med bogstavet på det korrekte svar.")
        prompts.append(f"{USER}{body}{END}{ASST}")

    outs = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=False,
                  return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_new, do_sample=False, num_beams=1,
                pad_token_id=tok.pad_token_id, eos_token_id=eos_ids,
                repetition_penalty=1.1,
            )
        plen = enc["input_ids"].shape[1]
        for row in gen:
            outs.append(tok.decode(row[plen:], skip_special_tokens=True).strip())

    n_ok = 0
    for out, (_, opts, gold) in zip(outs, items):
        present = "".join(sorted(opts))
        m = re.search(rf"\b[{present}]\b", out, re.IGNORECASE)
        if m and m.group(0).upper() == gold:
            n_ok += 1

    print(f"\n=== cit-mc {n_ok}/{len(items)} = {100*n_ok/len(items):.1f}% ===")


if __name__ == "__main__":
    main()
