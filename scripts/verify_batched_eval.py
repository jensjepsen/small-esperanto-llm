"""Assert the batched eval helpers match the per-item versions they replace.

Batching is only a speedup if it is numerically identical -- a padding-side
mistake shifts scores silently and every number produced afterwards is wrong
in a way no summary statistic reveals. This scores the same items both ways
and compares.

  score_cont_batch  vs the original single-sequence log-prob loop (exact to
                    float tolerance; same maths, one row at a time)
  generate_batch    vs single-prompt greedy generate (string equality; left
                    padding must make batch position irrelevant)

Usage:
  python scripts/verify_batched_eval.py --ckpt <path> [--n 24] [--bs 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batched_eval import generate_batch, score_cont_batch  # noqa: E402

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"


def score_cont_single(model, tok, prompt, cont):
    """The implementation that was in eval_arc_da/eval_gpqa_da/eval_piqa_da."""
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False,
                return_token_type_ids=False).input_ids.to(model.device)
    f_ids = tok(prompt + cont, return_tensors="pt", add_special_tokens=False,
                return_token_type_ids=False).input_ids.to(model.device)
    p_len = p_ids.shape[1]
    n_cont = f_ids.shape[1] - p_len
    if n_cont <= 0:
        return -float("inf")
    with torch.no_grad():
        logits = model(f_ids).logits
    cl = logits[0, p_len - 1:-1, :].float()
    ct = f_ids[0, p_len:]
    lp = F.log_softmax(cl, dim=-1)
    return lp.gather(1, ct.unsqueeze(1)).squeeze(1).sum().item() / n_cont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"],
                    help="fp32 isolates real indexing bugs from bf16 "
                         "batch-shape nondeterminism: reduction order changes "
                         "with batch width, so bf16 diffs of ~1e-1 on a "
                         "near-tie are expected and mean nothing.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.float32 if args.dtype == "fp32"
        else torch.bfloat16).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)
    eos = [tok.eos_token_id] + ([end_id] if end_id != tok.unk_token_id else [])

    ds = load_dataset("jensjepsen/danish-arc", "arc_easy", split="test")
    ds = ds.select(range(args.n))

    import ast as _ast
    pairs, prompts = [], []
    for r in ds:
        ch = r["choices"]
        if isinstance(ch, str):
            ch = _ast.literal_eval(ch)
        base = f"{r['question'].strip()}\nSvar: "
        pairs += [(base, c["text"]) for c in ch]
        opts = "\n".join(f"{c['label']}) {c['text']}" for c in ch)
        prompts.append(f"{USER}{r['question']}\n\n{opts}\n\n"
                       f"Svar med bogstavet på det korrekte svar.{END}{ASST}")

    print(f"scoring {len(pairs)} (prompt,cont) pairs both ways...", flush=True)
    batched = score_cont_batch(model, tok, pairs, bs=args.bs)
    single = [score_cont_single(model, tok, p, c) for p, c in pairs]
    import math
    diffs = [0.0 if (math.isinf(a) and math.isinf(b)) else abs(a - b)
             for a, b in zip(batched, single)]
    n_inf = sum(1 for a in batched if math.isinf(a))
    if n_inf:
        print(f"  ({n_inf} pairs scored -inf on both sides: empty "
              f"continuation after tokenization)")
    worst = max(diffs)
    print(f"  max |batched - single| = {worst:.2e}   mean = "
          f"{sum(diffs)/len(diffs):.2e}")
    # argmax per 4-option group is what actually decides accuracy
    flips = 0
    for k in range(0, len(pairs), 4):
        gb = max(range(4), key=lambda j: batched[k + j])
        gs = max(range(4), key=lambda j: single[k + j])
        flips += (gb != gs)
    print(f"  argmax flips: {flips}/{len(pairs)//4}")

    print(f"\ngenerating {len(prompts)} prompts both ways...", flush=True)
    gb = generate_batch(model, tok, prompts, 8, eos, bs=args.bs)
    gs = generate_batch(model, tok, prompts, 8, eos, bs=1)
    mism = [(i, a, b) for i, (a, b) in enumerate(zip(gb, gs)) if a != b]
    print(f"  mismatches: {len(mism)}/{len(prompts)}")
    for i, a, b in mism[:5]:
        print(f"    [{i}] batched={a!r}  single={b!r}")

    ok = worst < args.tol and flips == 0 and not mism
    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
