"""Cloze log-prob probe on danish-citizen-tests. Scores each MC choice as
NLL of (question + \" \" + choice), picks the min-NLL choice, reports accuracy.

Runs two models back-to-back so we can compare a base vs a checkpoint on
identical rows.

Usage:
    python scripts/probe_citgen_logprob.py <ckpt_path_or_hf> [<ckpt_path_or_hf> ...]
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = "Spørgsmål: {q}\nSvar: {c}"


def load_ds(split="train", limit=None):
    # danish-citizen-tests has only a 'train' split (720 rows), no test.
    # Schema: question, option_a, option_b, option_c (variable, some None),
    # answer (capital A/B/C).
    ds = load_dataset("alexandrainst/danish-citizen-tests", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for r in ds:
        opts_raw = [r.get(f"option_{k}") for k in "abc"]
        opts = [o.strip() for o in opts_raw if o]
        gold_letter = (r.get("answer") or "").strip().lower()
        gold_idx = "abc".find(gold_letter)
        if gold_idx < 0 or gold_idx >= len(opts):
            continue
        rows.append({
            "q": r["question"].strip(),
            "opts": opts,
            "gold": gold_idx,
        })
    return rows


@torch.no_grad()
def choice_nll(model, tok, prompt: str, choice: str, device):
    """NLL of the choice tokens conditioned on the prompt tokens.
    Returns mean per-token NLL over the choice span.
    """
    # Encode prompt + " " (space separator) with no special tokens.
    prompt_ids = tok(prompt + " ", add_special_tokens=False)["input_ids"]
    full_ids = tok(prompt + " " + choice, add_special_tokens=False)["input_ids"]
    choice_start = len(prompt_ids)
    if choice_start >= len(full_ids):
        return float("inf")

    inp = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = model(input_ids=inp).logits[0]         # [L, V]
    # Loss position i predicts token i+1. So loss over choice tokens
    # (indices choice_start..L-1) uses logits at positions choice_start-1..L-2.
    shift_logits = logits[choice_start - 1: -1]     # [choice_len, V]
    shift_labels = torch.tensor(full_ids[choice_start:], dtype=torch.long, device=device)
    if shift_logits.shape[0] != shift_labels.shape[0]:
        return float("inf")
    losses = F.cross_entropy(shift_logits, shift_labels, reduction="none")
    return losses.mean().item()


def eval_model(model_ref: str, rows, batch_desc: str):
    print(f"\n── {batch_desc}  ({model_ref}) ────────────────────────────")
    tok_ref = model_ref
    # If it's a local path with a tokenizer, use it; otherwise fall back to shared one.
    try:
        tok = AutoTokenizer.from_pretrained(tok_ref)
    except Exception:
        tok = AutoTokenizer.from_pretrained("jensjepsen/danish-tokenizer")
    model = AutoModelForCausalLM.from_pretrained(model_ref, torch_dtype=torch.float16).cuda().eval()
    device = next(model.parameters()).device

    correct = 0
    for i, r in enumerate(rows):
        prompt = f"Spørgsmål: {r['q']}\nSvar:"
        nlls = [choice_nll(model, tok, prompt, c, device) for c in r["opts"]]
        pred = int(min(range(len(nlls)), key=lambda k: nlls[k]))
        if pred == r["gold"]:
            correct += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}  acc={100*correct/(i+1):.1f}%", flush=True)

    acc = correct / len(rows)
    print(f"  final: {correct}/{len(rows)} = {100*acc:.2f}%", flush=True)
    del model
    torch.cuda.empty_cache()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="Checkpoint paths or HF repos")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = load_ds(args.split, args.limit)
    print(f"loaded {len(rows)} valid MC rows from citizen-tests {args.split}")

    results = {}
    for m in args.models:
        results[m] = eval_model(m, rows, f"model: {m}")

    print(f"\n═══ SUMMARY (n={len(rows)}) ═══")
    for m, acc in results.items():
        print(f"  {acc*100:6.2f}%  {m}")


if __name__ == "__main__":
    main()
