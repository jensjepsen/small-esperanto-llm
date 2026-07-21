"""Log-prob multiple-choice eval on Danish ARC + HellaSwag.

Runs quickly (~5 min for both on a 400M model on a single modern GPU).
Appends results to a CSV so you can plot a trajectory across checkpoints.

Usage:
    uv run python scripts/eval_da_mc.py --ckpt /path/to/checkpoint \
        --step 25000 [--csv results/da_pretrain_eval.csv] [--hs-samples 1000]
"""
import argparse
import csv
import os
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def score_choice(model, tok, prompt: str, choice: str) -> tuple[float, int]:
    """Return (sum_logprob, num_tokens) of `choice` given `prompt`."""
    prompt_ids = tok(prompt, return_tensors="pt").input_ids
    full_ids = tok(prompt + " " + choice, return_tensors="pt").input_ids.cuda()
    plen = prompt_ids.shape[1]
    with torch.no_grad():
        logits = model(full_ids).logits
    lp = F.log_softmax(logits[0, plen - 1:-1, :], dim=-1)
    tgt = full_ids[0, plen:]
    return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item(), tgt.numel()


def eval_arc(model, tok, verbose=True):
    ds = load_dataset("alexandrainst/m_arc", "da", split="test")
    n = len(ds)
    correct = 0
    t0 = time.time()
    for i, row in enumerate(ds):
        prompt = f"Spørgsmål: {row['instruction']}\nSvar:"
        opts = {k: row[f"option_{k.lower()}"] for k in "ABCDE"
                if row[f"option_{k.lower()}"] is not None}
        scores = {}
        for k, text in opts.items():
            lp, ntok = score_choice(model, tok, prompt, text)
            scores[k] = lp / max(ntok, 1)
        pred = max(scores, key=scores.get)
        if pred == row["answer"]:
            correct += 1
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  ARC {i+1}/{n} acc={correct/(i+1):.3f} eta={elapsed*(n-i-1)/(i+1):.0f}s")
    return correct / n, n


def eval_hellaswag(model, tok, k_samples=1000, seed=42, verbose=True):
    ds = load_dataset("alexandrainst/m_hellaswag", "da", split="val")
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:k_samples]
    n = len(idxs)
    correct = 0
    t0 = time.time()
    for i, idx in enumerate(idxs):
        row = ds[idx]
        ctx = row.get("instruction") or row.get("ctx") or row.get("context") or ""
        opts = {}
        for L in "ABCD":
            v = row.get(f"option_{L.lower()}")
            if v is not None:
                opts[L] = v
        if not opts and "endings" in row and row["endings"]:
            for j, e in enumerate(row["endings"]):
                opts["ABCD"[j]] = e
        if not opts:
            continue
        gold = row.get("answer") or row.get("label") or row.get("gold")
        if isinstance(gold, int):
            gold = "ABCD"[gold]
        elif isinstance(gold, str) and gold.isdigit():
            gold = "ABCD"[int(gold)]
        scores = {}
        for k, text in opts.items():
            lp, ntok = score_choice(model, tok, ctx, text)
            scores[k] = lp / max(ntok, 1)
        pred = max(scores, key=scores.get)
        if pred == gold:
            correct += 1
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  HS  {i+1}/{n} acc={correct/(i+1):.3f} eta={elapsed*(n-i-1)/(i+1):.0f}s")
    return correct / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to HF checkpoint directory")
    ap.add_argument("--step", type=int, required=True, help="Training step (for CSV)")
    ap.add_argument("--csv", default="/mnt/data2/da_pretrain_eval.csv",
                    help="CSV file to append results to")
    ap.add_argument("--tokenizer", default="jensjepsen/danish-tokenizer")
    ap.add_argument("--hs-samples", type=int, default=1000)
    ap.add_argument("--skip-arc", action="store_true")
    ap.add_argument("--skip-hs", action="store_true")
    args = ap.parse_args()

    print(f"Loading model from {args.ckpt}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    row = {"step": args.step, "timestamp": datetime.utcnow().isoformat(timespec="seconds")}

    if not args.skip_arc:
        print(f"\n=== ARC[da] test (full) ===")
        arc_acc, arc_n = eval_arc(model, tok)
        print(f"ARC[da]: {arc_acc:.4f} ({int(arc_acc*arc_n)}/{arc_n})")
        row["arc_da_acc"] = round(arc_acc, 4)
        row["arc_da_n"] = arc_n

    if not args.skip_hs:
        print(f"\n=== HellaSwag[da] val (n={args.hs_samples}) ===")
        hs_acc, hs_n = eval_hellaswag(model, tok, k_samples=args.hs_samples)
        print(f"HellaSwag[da]: {hs_acc:.4f} ({int(hs_acc*hs_n)}/{hs_n})")
        row["hs_da_acc"] = round(hs_acc, 4)
        row["hs_da_n"] = hs_n

    # Append to CSV (create with header if missing)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True) if os.path.dirname(args.csv) else None
    fieldnames = ["step", "timestamp", "arc_da_acc", "arc_da_n", "hs_da_acc", "hs_da_n"]
    exists = os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nAppended to {args.csv}")


if __name__ == "__main__":
    main()
