"""Textman-DA eval — ChrF++ vs single reference for summary and rewrite.

Mirrors the callback's scorer (sacrebleu.CHRF(word_order=2) = ChrF++).
Loads validation split of jensjepsen/danish-textman-v1, generates greedy,
scores corpus-level. Returns 0-100 score for each subtype.
"""
from __future__ import annotations
import argparse, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sacrebleu.metrics import CHRF

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"


def load_subtype(subtype):
    ds = load_dataset("jensjepsen/danish-textman-v1", split="validation")
    ds = ds.filter(lambda r: r["subtype"] == subtype)
    items = []
    for r in ds:
        prompt = r["messages"][0]["content"]
        gold = r["messages"][1]["content"]
        items.append((prompt, gold))
    return items


def generate(model, tok, prompts, max_new, bs, eos_ids):
    prev_side, prev_pad = tok.padding_side, tok.pad_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    outs = []
    t0 = time.time()
    try:
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False,
                      return_token_type_ids=False).to(model.device)
            with torch.no_grad():
                g = model.generate(input_ids=enc["input_ids"],
                                   attention_mask=enc["attention_mask"],
                                   max_new_tokens=max_new,
                                   do_sample=False, num_beams=1,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                   eos_token_id=eos_ids,
                                   repetition_penalty=1.1)
            plen = enc["input_ids"].shape[1]
            for row in g:
                outs.append(tok.decode(row[plen:], skip_special_tokens=True).strip())
            print(f"  {i+len(batch)}/{len(prompts)}  ({time.time()-t0:.0f}s)", flush=True)
    finally:
        tok.padding_side, tok.pad_token = prev_side, prev_pad
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--subtype", choices=["textman_summary", "textman_rewrite", "both"],
                    default="both")
    args = ap.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.ckpt)
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=dtype).cuda().eval()

    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id]
    if end_id != tok.unk_token_id:
        eos_ids.append(end_id)

    subs = ["textman_summary", "textman_rewrite"] if args.subtype == "both" else [args.subtype]
    for sub in subs:
        max_new = 200 if sub == "textman_summary" else 512
        print(f"\n=== {sub}  max_new={max_new}  bs={args.batch_size} ===", flush=True)
        items = load_subtype(sub)
        print(f"  n={len(items)}", flush=True)
        prompts = [q for q, _ in items]
        golds = [g for _, g in items]
        outs = generate(model, tok, prompts, max_new, args.batch_size, eos_ids)
        score = CHRF(word_order=2).corpus_score(outs, [golds]).score
        print(f"=== {sub}  ChrF++ = {score:.2f} ===", flush=True)


if __name__ == "__main__":
    main()
