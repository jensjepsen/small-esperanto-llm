"""GPQA-Diamond DA eval: 4-way MC via chat-mc (letter generate + parse) or raw-logp.

Reads local JSONL from translate_gpqa_diamond_or.py output (default in schema:
`answers_da[0]` = correct, rest = distractors). Shuffles A/B/C/D per-row with
deterministic seed for chat-mc; raw-logp scores options directly.

Usage:
    uv run python scripts/eval_gpqa_da.py --ckpt HF_ID_OR_LOCAL \\
        --data data/danish_gpqa_diamond/da_gpqa_diamond.jsonl [--mode raw-logp]
"""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path as _Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from batched_eval import generate_batch, ratchet, score_cont_batch  # noqa: E402

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
LETTER_RE = re.compile(r"\b([ABCD])\b")


def score_cont(model, tok, prompt, cont):
    """Length-normalized log P(cont | prompt)."""
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False,
                     return_token_type_ids=False).input_ids.cuda()
    full_ids = tok(prompt + cont, return_tensors="pt", add_special_tokens=False,
                   return_token_type_ids=False).input_ids.cuda()
    p_len = prompt_ids.shape[1]
    n_cont = full_ids.shape[1] - p_len
    if n_cont <= 0:
        return -float("inf")
    with torch.no_grad():
        logits = model(full_ids).logits
    cont_logits = logits[0, p_len - 1 : -1, :].float()
    cont_targets = full_ids[0, p_len:]
    log_probs = F.log_softmax(cont_logits, dim=-1)
    tok_logp = log_probs.gather(1, cont_targets.unsqueeze(1)).squeeze(1)
    return tok_logp.sum().item() / n_cont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dtype", default="fp32", choices=["fp32","fp16","bf16"])
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-every", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Rows per forward/generate batch. The logp modes "
                         "score 4 sequences per row, batched at this size.")
    ap.add_argument("--mode", default="chat-mc",
                    choices=["chat-mc", "raw-logp", "chat-logp"],
                    help="chat-mc: chat-wrapped letter gen + parse. "
                         "raw-logp: score each option as continuation of "
                         "'question\\nSvar: ', pick highest length-norm log P. "
                         "chat-logp: chat-wrapped MC prompt, score P(letter) for "
                         "A/B/C/D as first-token continuation, argmax.")
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

    LETTERS = "ABCD"

    def shuffled(r):
        """Per-row A/B/C/D permutation, same deterministic seed as before."""
        rng = random.Random(args.seed + r["orig_idx"])
        idxs = list(range(4))
        rng.shuffle(idxs)
        lines, gold = [], None
        for slot, orig in enumerate(idxs):
            lines.append(f"{LETTERS[slot]}) {list(r['answers_da'])[orig]}")
            if orig == 0:
                gold = LETTERS[slot]
        return lines, gold

    t0 = time.time()
    n_parsefail = 0

    if args.mode == "raw-logp":
        pairs = []
        for r in rows:
            base = f"{r['question_da'].strip()}\nSvar: "
            pairs += [(base, a) for a in list(r["answers_da"])]
        lps = score_cont_batch(model, tok, pairs, bs=args.batch_size,
                               progress=ratchet("raw-logp"))
        n_ok = sum(1 for k in range(len(rows))
                   if max(range(4), key=lambda j: lps[4 * k + j]) == 0)
    elif args.mode == "chat-mc":
        prompts, golds = [], []
        for r in rows:
            lines, gold = shuffled(r)
            body = (f"{r['question_da']}\n\n" + "\n".join(lines) +
                    "\n\nSvar med bogstavet på det korrekte svar.")
            prompts.append(f"{USER}{body}{END}{ASST}")
            golds.append(gold)
        gens = generate_batch(model, tok, prompts, args.max_new, eos_ids,
                              bs=args.batch_size, progress=ratchet("chat-mc"))
        n_ok = 0
        for g, gold in zip(gens, golds):
            m = LETTER_RE.search(g)
            if not m:
                n_parsefail += 1
                continue
            n_ok += (m.group(1) == gold)
    else:  # chat-logp
        pairs, golds = [], []
        for r in rows:
            lines, gold = shuffled(r)
            body = (f"{r['question_da']}\n\n" + "\n".join(lines) +
                    "\n\nSvar med bogstavet på det korrekte svar.")
            prompt = f"{USER}{body}{END}{ASST}"
            pairs += [(prompt, lab) for lab in LETTERS]
            golds.append(gold)
        lps = score_cont_batch(model, tok, pairs, bs=args.batch_size,
                               progress=ratchet("chat-logp"))
        n_ok = 0
        for k, gold in enumerate(golds):
            best = max(range(4), key=lambda j: lps[4 * k + j])
            n_ok += (LETTERS[best] == gold)

    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print(f"\n=== gpqa-diamond-da[{args.mode}]  n={len(rows)}  "
          f"acc={100*n_ok/len(rows):.2f}%  ({n_ok}/{len(rows)})  "
          f"parsefail={n_parsefail}  random={100/4:.1f}% ===",
          flush=True)


if __name__ == "__main__":
    main()
