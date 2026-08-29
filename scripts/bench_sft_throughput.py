"""Measure SFT training throughput across configs, in tokens/s and MFU.

Wall-clock per step is not comparable across batch sizes -- a bigger batch
does proportionally more work per step. Tokens/s is, and MFU says how much of
the GPU is actually being used. An earlier estimate here assumed the job was
compute-bound because utilisation read 99%; it was running at 8% MFU. GPU
utilisation reports that a kernel was resident, not that the SMs were busy.

Runs real rows from the real mix through real training steps (forward,
backward, optimizer) rather than synthetic tensors, so packing efficiency and
the short-row penalty are included.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorWithFlattening)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sft import format_conversation  # noqa: E402

H100_BF16_PEAK = 989e12   # dense, no sparsity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="jensjepsen/danish-lm-400m-sft-v31-avg-top3")
    ap.add_argument("--datasets", nargs="+",
                    default=["jensjepsen/danish-ner-sft-v1:sft",
                             "jensjepsen/danish-icl-schema-format-v3:sft"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 128])
    ap.add_argument("--attn", nargs="+", default=["flash_attention_2"])
    ap.add_argument("--compile", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--optim", nargs="+", default=["adamw_torch_fused"])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rows", type=int, default=4000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    rows = []
    for spec in args.datasets:
        repo, _, cfg = spec.partition(":")
        d = load_dataset(repo, cfg or None, split="train")
        d = d.select(range(min(args.rows // len(args.datasets), len(d))))
        for r in d:
            ids = tok(format_conversation(r["messages"]),
                      add_special_tokens=False)["input_ids"][:3072]
            rows.append({"input_ids": ids, "labels": ids})
    ntok = sum(len(r["input_ids"]) for r in rows)
    print(f"{len(rows)} rows, {ntok:,} tokens, mean {ntok/len(rows):.0f}/row\n",
          flush=True)

    coll = DataCollatorWithFlattening()
    print(f"{'attn':<20}{'bs':>5}{'compile':>9}{'optim':>20}"
          f"{'tok/s':>11}{'MFU':>7}{'s/step':>9}")
    results = []
    for attn in args.attn:
        for comp in args.compile:
            for opt in args.optim:
                for bs in args.batch_sizes:
                    try:
                        r = bench(args, tok, rows, coll, attn, bool(comp), opt, bs)
                    except torch.cuda.OutOfMemoryError:
                        print(f"{attn:<20}{bs:>5}{comp:>9}{opt:>20}{'OOM':>11}")
                        torch.cuda.empty_cache()
                        continue
                    results.append(r)
                    print(f"{attn:<20}{bs:>5}{comp:>9}{opt:>20}"
                          f"{r['tok_s']:>11,.0f}{100*r['mfu']:>6.0f}%"
                          f"{r['s_step']:>9.3f}", flush=True)
    best = max(results, key=lambda r: r["tok_s"]) if results else None
    if best:
        base = min(results, key=lambda r: r["tok_s"])
        print(f"\nbest {best['tok_s']:,.0f} tok/s at bs={best['bs']} "
              f"compile={best['compile']} optim={best['optim']} "
              f"({best['tok_s']/base['tok_s']:.2f}x the slowest cell)")
        print(json.dumps(results, indent=2)[:0] or "", end="")


def bench(args, tok, rows, coll, attn, comp, opt, bs):
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(
        args.ckpt, attn_implementation=attn, dtype=torch.bfloat16).cuda()
    m.gradient_checkpointing_disable()
    m.train()
    if comp:
        m = torch.compile(m)
    P = sum(p.numel() for p in m.parameters())
    if opt == "adamw_torch_fused":
        o = torch.optim.AdamW(m.parameters(), lr=1e-5, fused=True)
    else:
        import bitsandbytes as bnb
        o = bnb.optim.AdamW8bit(m.parameters(), lr=1e-5)

    batches, i = [], 0
    while len(batches) < args.steps + args.warmup and i + bs <= len(rows):
        batches.append(coll(rows[i:i + bs]))
        i += bs
    tot_tok = 0
    for n, b in enumerate(batches):
        if n == args.warmup:
            torch.cuda.synchronize()
            t0 = time.time()
            tot_tok = 0
        b = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in b.items()}
        out = m(**b)
        out.loss.backward()
        o.step()
        o.zero_grad(set_to_none=True)
        if n >= args.warmup:
            tot_tok += b["input_ids"].numel()
    torch.cuda.synchronize()
    dt = time.time() - t0
    steps = len(batches) - args.warmup
    tok_s = tot_tok / dt
    del m, o
    torch.cuda.empty_cache()
    return {"attn": attn, "bs": bs, "compile": comp, "optim": opt,
            "tok_s": tok_s, "s_step": dt / steps,
            "mfu": 6 * P * tok_s / H100_BF16_PEAK}


if __name__ == "__main__":
    main()
