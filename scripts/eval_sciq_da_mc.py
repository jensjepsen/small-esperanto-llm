"""Standalone A/B/C/D MC eval on jensjepsen/danish-sciq (Danish SciQ).

Uses the 4 built-in choices (da_correct_answer + da_distractor1..3),
shuffles positions per-row with a fixed seed, prompts with the same
'Svar med bogstavet...' shape as citmc / wiki-mc-letters.

Usage:
    python scripts/eval_sciq_da_mc.py CKPT [--batch-size 32] [--seed 42]
"""
from __future__ import annotations

import argparse
import random
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
LABELS = ["A", "B", "C", "D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
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

    ds = load_dataset("jensjepsen/danish-sciq", "default", split="test")
    rng = random.Random(args.seed)
    items = []
    for r in ds:
        q = r["da_question"]
        correct = r["da_correct_answer"]
        distractors = [r["da_distractor1"], r["da_distractor2"], r["da_distractor3"]]
        options = [correct] + distractors
        rng.shuffle(options)
        gold_letter = LABELS[options.index(correct)]
        items.append((q, options, gold_letter))
    print(f"sciq-mc rows: {len(items)}", flush=True)

    prompts = []
    for q, options, _ in items:
        opts_str = "\n".join(f"{LABELS[i]}) {o}" for i, o in enumerate(options))
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
    pattern = re.compile(r"\b[ABCD]\b", re.IGNORECASE)
    for out, (_, _, gold) in zip(outs, items):
        m = pattern.search(out)
        if m and m.group(0).upper() == gold:
            n_ok += 1

    print(f"\n=== sciq-mc {n_ok}/{len(items)} = {100*n_ok/len(items):.1f}% ===")


if __name__ == "__main__":
    main()
