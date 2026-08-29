"""Probe an SFT checkpoint against `jensjepsen/danish-instruction-following-v2:default:eval`.

Loads N eval rows, samples model output for each, re-runs the constraint checks,
prints per-constraint pass rates + a handful of failing samples.

Usage:
    python scripts/probe_if_v5.py --checkpoint /root/runs/sft/da_v5_mix9if2/checkpoint-7740 \
                                  --tokenizer jensjepsen/danish-tokenizer \
                                  --n 60
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import if_constraints as ifc  # noqa: E402

BY_NAME = {c.name: c for c in ifc.ALL}


def build_check_combo(names: list[str], params_json: str) -> list[dict]:
    params_by_name = json.loads(params_json)
    out = []
    for n in names:
        c = BY_NAME.get(n)
        if c is None:
            continue
        p = params_by_name.get(n, {})
        out.append({"name": n, "params": p, "render": "", "_check": c.check})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="jensjepsen/danish-tokenizer")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-fails", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    print(f"loading tokenizer {args.tokenizer}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"loading model {args.checkpoint}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16
    ).cuda().eval()

    print("loading eval split", flush=True)
    ds = load_dataset("jensjepsen/danish-instruction-following-v2",
                      "default", split="eval")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))
    print(f"probing {len(ds)} rows", flush=True)

    per_constraint_total: dict[str, int] = defaultdict(int)
    per_constraint_pass: dict[str, int] = defaultdict(int)
    all_pass = 0
    failed_samples = []

    end_id = tok.convert_tokens_to_ids("<|end|>")
    stop_ids = [tok.eos_token_id, end_id] if end_id is not None else [tok.eos_token_id]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
    else:
        tok.padding_side = "left"

    rows = list(ds)
    for bs_start in range(0, len(rows), args.batch_size):
        batch = rows[bs_start:bs_start + args.batch_size]
        prompts = [f"<|user|> {r['messages'][0]['content']} <|assistant|>" for r in batch]
        ids = tok(prompts, return_tensors="pt", return_token_type_ids=False,
                  padding=True).to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temp > 0),
                temperature=args.temp if args.temp > 0 else 1.0,
                pad_token_id=tok.eos_token_id,
                eos_token_id=stop_ids,
            )
        gens = []
        input_len = ids["input_ids"].shape[1]
        for b_ix in range(out.shape[0]):
            gen = tok.decode(out[b_ix, input_len:], skip_special_tokens=True)
            gen = gen.replace("<|end|>", "").strip()
            gens.append(gen)

        for row, gen in zip(batch, gens):
            combo = build_check_combo(row["constraints"], row["params"])
            ok, failures = ifc.verify_all(gen, combo)
            for r in combo:
                per_constraint_total[r["name"]] += 1
                if r["name"] not in {f.split(":")[0] for f in failures}:
                    per_constraint_pass[r["name"]] += 1
            if ok:
                all_pass += 1
            elif len(failed_samples) < args.show_fails:
                failed_samples.append({
                    "prompt": row["messages"][0]["content"],
                    "gen": gen,
                    "constraints": row["constraints"],
                    "failed": failures,
                })

        done = bs_start + len(batch)
        print(f"  {done}/{len(rows)}  all-pass={all_pass}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"all-constraints-pass: {all_pass}/{len(ds)} = {100 * all_pass / len(ds):.1f}%")
    print("\nper-constraint (pass/total = rate):")
    for name in sorted(per_constraint_total, key=lambda n: per_constraint_pass[n] / per_constraint_total[n]):
        tot = per_constraint_total[name]
        pas = per_constraint_pass[name]
        print(f"  {100 * pas / tot:5.1f}%  {pas:3d}/{tot:3d}  {name}")

    print("\n=== FAILING SAMPLES ===")
    for s in failed_samples:
        print("-" * 72)
        print("PROMPT:")
        print(s["prompt"])
        print("CONSTRAINTS:", s["constraints"])
        print("FAILED:", s["failed"])
        print("GEN:")
        print(s["gen"])
        print()


if __name__ == "__main__":
    main()
