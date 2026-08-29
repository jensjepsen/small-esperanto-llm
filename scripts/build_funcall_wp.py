"""Build a tool-call SFT dataset from the diverse word-problem pipeline.

Produces one JSONL record per problem with the shape::

    {
      "messages": [
        {"role": "user",      "content": "Petro dividas..."},
        {"role": "assistant", "content": "ni dividu... <|tool_call|>3+5<|/tool_call|>"},
        {"role": "tool",      "content": "<|tool_result|>8<|/tool_result|>"},
        {"role": "assistant", "content": "do 8 partoj entute. ... <|tool_call|>40/8<|/tool_call|>"},
        {"role": "tool",      "content": "<|tool_result|>5<|/tool_result|>"},
        ...
        {"role": "assistant", "content": "#### 25"}
      ],
      "type":          "ratio-diverse",
      "strategy":      "parts",
      "answer":        "25",
      "math_language": "verbose",
      "wrapper_tone":  "klariga",
      "question_form": "direct",
      "n_tool_calls":  3
    }

Tool calls use the special tokens declared in `esperanto_lm.funcall.tokens`,
so the dataset is directly trainable with the existing tool-aware
`train_sft.py` (which knows the chat + tool turn structure).

Pipeline:
  1. `render_diverse(type, rng)` from word_problems_diverse  picks the
     diverse-wrapped EO question, math language, wrapper tone, and the
     concrete `(problem, strategy)` pair.
  2. `render_funcall_for(type, p, strategy)` from word_problems_procedural
     emits the multi-turn assistant/tool sequence with hidden results.

Usage::

    uv run python scripts/build_funcall_wp.py \\
        --types ratio,percent,inverse-rate,consecutive,coin,age,mixture,distance \\
        --n 5000 --out data/word_problems/funcall_40k.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from word_problems_diverse import DIVERSE, render_diverse
from word_problems_procedural import render_funcall_for


def build_record(type_name: str, rng: random.Random) -> dict | None:
    row = render_diverse(type_name, rng)
    if row is None:
        return None
    p = row.pop("_problem")
    base_type = row.pop("_base_type")
    strat = row["strategy"]
    try:
        chain_turns = render_funcall_for(base_type, p, strat)
    except (KeyError, ValueError):
        return None
    messages = [{"role": "user", "content": row["question_eo"]}, *chain_turns]
    n_calls = sum(1 for t in chain_turns if t["role"] == "tool")
    return {
        "messages": messages,
        "type": row["type"],
        "strategy": strat,
        "answer": str(row["answer"]),
        "math_language": row["math_language"],
        "wrapper_tone": row["wrapper_tone"],
        "question_form": row["question_form"],
        "n_tool_calls": n_calls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types",
                    default="ratio,percent,inverse-rate,consecutive,coin,age,mixture,distance")
    ap.add_argument("--n", type=int, default=5000, help="problems per type")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts-mult", type=int, default=5,
                    help="cap attempts per type at n * this")
    args = ap.parse_args()

    types = args.types.split(",")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    t0 = time.time()
    n_written = 0
    n_skipped = 0
    per_type_written = {}
    with args.out.open("w") as f:
        for t in types:
            if t not in DIVERSE:
                print(f"skip unknown type {t!r}", flush=True)
                continue
            written = 0
            attempts = 0
            cap = args.n * args.max_attempts_mult
            while written < args.n and attempts < cap:
                attempts += 1
                row = build_record(t, rng)
                if row is None:
                    n_skipped += 1
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                n_written += 1
            per_type_written[t] = written
            rate = written / max(0.1, time.time() - t0)
            print(f"  {t:14s}  wrote {written:>6d}/{args.n}  "
                  f"(attempts={attempts}, cum-rate={rate:.0f}/s)", flush=True)
    elapsed = time.time() - t0
    print(f"\ntotal: {n_written} records → {args.out}  "
          f"({n_skipped} skipped, {elapsed:.0f}s, {n_written/max(0.1,elapsed):.0f}/s)")
    print("per type:", per_type_written)


if __name__ == "__main__":
    main()
