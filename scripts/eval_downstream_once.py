"""Run the training-loop downstream evals on a checkpoint, outside training.

The same callback the trainer uses, so a baseline measured here is directly
comparable to the in-training curve rather than being a different
implementation of nominally the same metric. Exists because in-loop numbers
are meaningless without a base reading, and repeatedly re-deriving one by hand
invites the two from drifting apart.

Usage:
  python scripts/eval_downstream_once.py --ckpt <repo-or-path> --evals ifeval icl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esperanto_lm.downstream_eval_callback import (  # noqa: E402
    DownstreamEvalCallback,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--evals", nargs="+",
                    default=["gsm8k", "citgen", "sciq", "ifeval", "icl"])
    ap.add_argument("--n", type=int, default=200,
                    help="rows per eval; match the training run's "
                         "--downstream-n or the numbers are not comparable")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=getattr(torch, args.dtype)).cuda().eval()
    cb = DownstreamEvalCallback(tokenizer=tok, evals=tuple(args.evals),
                               n_per_eval=args.n, batch_size=args.batch_size)

    print(f"{args.ckpt}   n={args.n} per eval\n", flush=True)
    for name in args.evals:
        scorer = getattr(cb, f"_score_{name}", None)
        if scorer is None:
            print(f"  {name:<10} (no scorer)")
            continue
        try:
            v = scorer(model)
            print(f"  {name:<10} {100 * v:.1f}%", flush=True)
        except Exception as e:
            print(f"  {name:<10} FAILED {type(e).__name__}: {str(e)[:90]}",
                  flush=True)


if __name__ == "__main__":
    main()
