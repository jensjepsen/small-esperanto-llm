"""PIQA-DA logprob MC eval: pick solution with higher log-prob given the prompt.

Source: mrlbenchmarks/global-piqa-nonparallel:dan_latn:test (100 human-authored
Danish PIQA items). Each row has {prompt, solution0, solution1, label}.

For each item, score sol0 and sol1 by conditional log-prob under the model,
length-normalized (sum log p / n_tokens). Higher score = model prefers that
solution. Compare to gold label.

Usage:
    uv run python scripts/eval_piqa_da.py --ckpt PATH [--n 100]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path as _Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from batched_eval import generate_batch, ratchet, score_cont_batch  # noqa: E402

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
LETTER_RE = re.compile(r"\b([AB])\b")


def score_conditional(model, tok, prompt, cont):
    """Sum log-prob of `cont` tokens given `prompt`. Length-normalized."""
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False,
                     return_token_type_ids=False).input_ids.cuda()
    full = prompt + cont
    full_ids = tok(full, return_tensors="pt", add_special_tokens=False,
                   return_token_type_ids=False).input_ids.cuda()
    p_len = prompt_ids.shape[1]
    n_cont = full_ids.shape[1] - p_len
    if n_cont <= 0:
        return -float("inf")
    with torch.no_grad():
        logits = model(full_ids).logits  # [1, T, V]
    # For continuation tokens, use logits[:, p_len-1 : -1] predicting full_ids[:, p_len:]
    cont_logits = logits[0, p_len - 1 : -1, :].float()
    cont_targets = full_ids[0, p_len:]
    log_probs = F.log_softmax(cont_logits, dim=-1)
    tok_logp = log_probs.gather(1, cont_targets.unsqueeze(1)).squeeze(1)
    return tok_logp.sum().item() / n_cont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dtype", default="fp32",
                    choices=["fp32", "fp16", "bf16"],
                    help="fp32 matches HF master weights (default). "
                         "fp16/bf16 downcast at load — faster but lossy.")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Rows per forward/generate batch.")
    ap.add_argument("--mode", default="raw",
                    choices=["raw", "chat-completion", "chat-mc"],
                    help="raw: standard PIQA continuation scoring. "
                         "chat-completion: score sol as assistant reply "
                         "to a completion prompt. chat-mc: score log P(A) vs "
                         "log P(B) after '<|assistant|>' given an A/B prompt.")
    args = ap.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=dtype).cuda().eval()

    ds = load_dataset("mrlbenchmarks/global-piqa-nonparallel", "dan_latn",
                      split="test")
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))
    n = len(ds)
    print(f"  {n} rows", flush=True)

    a_id = tok(" A", add_special_tokens=False).input_ids
    b_id = tok(" B", add_special_tokens=False).input_ids
    # Use single-token IDs when possible for chat-mc scoring
    A_TOKEN = a_id[-1]
    B_TOKEN = b_id[-1]

    rows = [(r["prompt"].strip(), r["solution0"].strip(),
             r["solution1"].strip(), r["label"]) for r in ds]
    t0 = time.time()
    n_parsefail = 0

    if args.mode in ("raw", "chat-completion"):
        pairs = []
        for prompt, s0, s1, _ in rows:
            if args.mode == "raw":
                base = prompt + " "
            else:
                u = (f"Fuldstændiggør sætningen på den mest logiske måde. "
                     f"Skriv KUN den manglende del.\n\n\"{prompt}\"")
                base = f"{USER}{u}{END}{ASST}"
            pairs += [(base, s0), (base, s1)]
        lps = score_cont_batch(model, tok, pairs, bs=args.batch_size,
                               progress=ratchet(args.mode))
        preds = [0 if lps[2 * k] > lps[2 * k + 1] else 1
                 for k in range(len(rows))]
    else:  # chat-mc
        prompts = []
        for prompt, s0, s1, _ in rows:
            q = f"Hvilken fortsættelse passer bedst? \"{prompt}\""
            u = (f"{q}\n\n"
                 f"A) {s0}\nB) {s1}\n\n"
                 f"Svar med bogstavet på det korrekte svar.")
            prompts.append(f"{USER}{u}{END}{ASST}")
        end_id = tok.convert_tokens_to_ids(END)
        eos_ids = [tok.eos_token_id] + (
            [end_id] if end_id != tok.unk_token_id else [])
        gens = generate_batch(model, tok, prompts, 8, eos_ids,
                              bs=args.batch_size, progress=ratchet("chat-mc"))
        preds = []
        for g in gens:
            m = LETTER_RE.search(g)
            if not m:
                n_parsefail += 1
                preds.append(-1)
            else:
                preds.append(0 if m.group(1) == "A" else 1)

    n_ok = sum(1 for pr, (_, _, _, gold) in zip(preds, rows) if pr == gold)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print(f"\n=== piqa-da[{args.mode}]  n={n}  acc={100*n_ok/n:.2f}%  ({n_ok}/{n})  parsefail={n_parsefail} ===",
          flush=True)


if __name__ == "__main__":
    main()
