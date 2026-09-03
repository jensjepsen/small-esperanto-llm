"""Re-score a published checkpoint's tool evals against a chosen gold corpus.

WHY THIS EXISTS. `TOOL_REPO` used to be hardcoded to danish-tool-dialogues-v1,
so every tool number ever reported was measured against v1's gold -- and v1
translated argument values PER ROW, leaving ~40% of mixed-slot values in
English and ~60% in Danish, decided by translation-batch luck rather than by
anything visible in the prompt. On those slots no model can be reliably right:
it learns the majority form and is marked wrong whenever gold holds the other.

So v35's headline 81.2 / 82.5 argF1 was measured against a partly unanswerable
test, and a v36 trained on the CANONICAL v2 corpus scored 9pp worse against
that same test purely for being self-consistent. Comparing the two runs needs
one yardstick applied to both, offline, with no training.

This drives the real DownstreamEvalCallback rather than reimplementing its
scoring: same prompt construction, same <|tool_call|> regex and raw_decode
fallback, same graded pair-F1 with a wrong tool scoring zero. A reimplementation
would be one silent divergence away from producing numbers that cannot be
compared with the in-training ones -- which is the exact failure this script
exists to clean up after.

Usage:
  python scripts/rescore_tool_evals.py \
      --model jensjepsen/danish-lm-400m-sft-v35 \
      --subfolder step-45312-agg-0.408 \
      --gold jensjepsen/danish-tool-dialogues-v1 jensjepsen/danish-tool-dialogues-v2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esperanto_lm.downstream_eval_callback import (  # noqa: E402
    DownstreamEvalCallback,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--gold", nargs="+", required=True,
                    help="one or more tool-dialogue repos to score against")
    ap.add_argument("--splits", nargs="+",
                    default=["tool_seen", "tool_unseen"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="float16",
                    help="fp16 on Pascal: bf16 is emulated below sm_80")
    args = ap.parse_args()

    kw = {"subfolder": args.subfolder} if args.subfolder else {}
    print(f"loading {args.model} {args.subfolder or ''}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), **kw)
    model = model.cuda().eval()

    results = {}
    for gold in args.gold:
        print(f"\n=== gold: {gold} ===", flush=True)
        cb = DownstreamEvalCallback(
            tok, evals=tuple(args.splits), batch_size=args.batch_size)
        # per-instance override; _tool_items reads self.TOOL_REPO
        cb.TOOL_REPO = gold
        cb._cache = {}
        for name in args.splits:
            # _tool_score prints only when a whole split finishes, which on a
            # 1080 Ti is 20-40 minutes of silence per gold. Say what is about
            # to run and how big it is, so a slow pass is distinguishable from
            # a wedged one without resorting to nvidia-smi.
            n = len(cb._get(name))
            print(f"  {name}: {n} prompts, bs={args.batch_size}, "
                  f"max_new={cb.TOOL_MAX_NEW} — generating...", flush=True)
            t0 = time.time()
            score = cb._tool_score(model, name)
            print(f"  {name}: done in {time.time()-t0:.0f}s", flush=True)
            extra = {k: v for k, v in cb._extra_metrics.items() if name in k}
            results[(gold, name)] = (score, dict(extra))
            cb._extra_metrics = {}

    print("\n" + "=" * 74)
    print(f"{'gold':<44} {'split':<13} {'argF1':>7}")
    print("=" * 74)
    for (gold, name), (score, _extra) in results.items():
        print(f"{gold.split('/')[-1]:<44} {name:<13} {100*score:>6.1f}%")
    if len(args.gold) == 2:
        a, b = args.gold
        print("\ndelta (second gold minus first):")
        for name in args.splits:
            d = 100 * (results[(b, name)][0] - results[(a, name)][0])
            print(f"   {name:<13} {d:+.1f} pp")


if __name__ == "__main__":
    main()
