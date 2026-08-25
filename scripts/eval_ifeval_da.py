"""Danish IFEval — runs the standard Google IFEval verifiers against
danish-foundation-models/ifeval-da.

Reports two metrics per constraint level (loose / strict):
  - prompt-level accuracy (all constraints in a prompt pass)
  - instruction-level accuracy (individual constraints)

Usage:
    python scripts/eval_ifeval_da.py CKPT [--subfolder SUB] [--batch-size 16]
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
from ifeval_google import instructions_registry as reg  # noqa: E402

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"


def build_instructions(row):
    """Instantiate the constraint check classes for this row."""
    instructions_list = []
    for i, name in enumerate(row["instruction_id_list"]):
        cls = reg.INSTRUCTION_DICT.get(name)
        if cls is None:
            continue
        inst = cls(name)
        raw_kwargs = row["kwargs"][i] if i < len(row["kwargs"]) else {}
        kwargs = {k: v for k, v in raw_kwargs.items() if v is not None}
        try:
            inst.build_description(**kwargs)
        except TypeError:
            # some kwarg names differ; try with none
            try:
                inst.build_description()
            except Exception:
                # verifier can't handle these kwargs; skip
                continue
        except KeyError:
            # e.g. language:response_language for a language not in
            # google's LANGUAGES table (Faroese, Norwegian variants, …)
            continue
        instructions_list.append((name, inst))
    return instructions_list


def score_row(response: str, insts):
    """Return (all_follow_strict, all_follow_loose, per-inst list)."""
    strict_flags = []
    loose_flags = []
    for _, inst in insts:
        try:
            strict_flags.append(inst.check_following(response))
        except Exception:
            strict_flags.append(False)
        # loose = try 3 variants of the response (trimmed / uppercased / etc.)
        variants = [response, response.strip(),
                    "\n".join(response.split("\n")[1:]).strip() if "\n" in response else response,
                    response.replace("*", "")]
        loose = False
        for v in variants:
            try:
                if inst.check_following(v):
                    loose = True
                    break
            except Exception:
                pass
        loose_flags.append(loose)
    return strict_flags, loose_flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--repetition-penalty", type=float, default=1.1,
                    help="HF repetition_penalty for generate(). Default 1.1 "
                         "mirrors the training/callback stack. Try 1.0 to "
                         "unlock keyword-frequency / repeat_prompt constraints "
                         "which fight the default penalty.")
    ap.add_argument("--dump-jsonl", default=None,
                    help="If set, per-row (prompt, kwargs, gen, verdicts) go here.")
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32,
             "bf16": torch.bfloat16}[args.dtype]

    tok_kw = {"subfolder": args.subfolder} if args.subfolder else {}
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.ckpt, **tok_kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    print(f"loading {args.ckpt}" + (f"/{args.subfolder}" if args.subfolder else ""), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=dtype, **tok_kw).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id] + ([end_id] if end_id != tok.unk_token_id else [])

    ds = load_dataset("danish-foundation-models/ifeval-da", split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"rows: {len(ds)}", flush=True)

    prompts = [f"{USER}{r['prompt']}{END}{ASST}" for r in ds]
    all_insts = [build_instructions(r) for r in ds]

    # Streaming: generate + score per batch, print running stats.
    prompt_strict_ok = 0
    prompt_loose_ok = 0
    inst_strict_ok = 0
    inst_loose_ok = 0
    inst_strict_tot = 0
    prompts_scored = 0
    per_constraint = defaultdict(lambda: [0, 0])
    per_subtype = defaultdict(lambda: [0, 0])
    dump_fh = None
    if args.dump_jsonl:
        Path(args.dump_jsonl).parent.mkdir(parents=True, exist_ok=True)
        dump_fh = open(args.dump_jsonl, "w")

    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=False,
                  return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False, num_beams=1,
                pad_token_id=tok.pad_token_id, eos_token_id=eos_ids,
                repetition_penalty=args.repetition_penalty,
            )
        plen = enc["input_ids"].shape[1]
        # Score each row in the batch immediately
        for j, row in enumerate(gen):
            row_ix = i + j
            resp = tok.decode(row[plen:], skip_special_tokens=True).strip()
            insts = all_insts[row_ix]
            if not insts:
                continue
            s, l = score_row(resp, insts)
            prompt_strict_ok += all(s)
            prompt_loose_ok += all(l)
            inst_strict_ok += sum(s)
            inst_loose_ok += sum(l)
            inst_strict_tot += len(s)
            prompts_scored += 1
            for (name, _), ok_s in zip(insts, s):
                fam = name.split(":")[0]
                per_constraint[fam][0] += ok_s
                per_constraint[fam][1] += 1
                per_subtype[name][0] += ok_s
                per_subtype[name][1] += 1
            if dump_fh is not None:
                src = ds[row_ix]
                dump_fh.write(json.dumps({
                    "key": src.get("key"),
                    "prompt": src["prompt"],
                    "instruction_id_list": src["instruction_id_list"],
                    "kwargs": src["kwargs"],
                    "generation": resp,
                    "strict_flags": s,
                    "loose_flags": l,
                    "prompt_strict": all(s),
                    "prompt_loose": all(l),
                }, ensure_ascii=False) + "\n")
                dump_fh.flush()
        done = i + len(batch)
        if prompts_scored:
            print(f"  {done}/{len(prompts)}  "
                  f"prompt-strict={100*prompt_strict_ok/prompts_scored:.1f}%  "
                  f"prompt-loose={100*prompt_loose_ok/prompts_scored:.1f}%  "
                  f"inst-strict={100*inst_strict_ok/inst_strict_tot:.1f}%  "
                  f"inst-loose={100*inst_loose_ok/inst_strict_tot:.1f}%",
                  flush=True)

    if dump_fh is not None:
        dump_fh.close()
        print(f"per-row dump → {args.dump_jsonl}", flush=True)

    n = prompts_scored
    print(f"\n=== ifeval-da  n={n} ===")
    print(f"  prompt-strict:  {prompt_strict_ok}/{n} = {100*prompt_strict_ok/n:.1f}%")
    print(f"  prompt-loose:   {prompt_loose_ok}/{n} = {100*prompt_loose_ok/n:.1f}%")
    print(f"  inst-strict:    {inst_strict_ok}/{inst_strict_tot} = {100*inst_strict_ok/inst_strict_tot:.1f}%")
    print(f"  inst-loose:     {inst_loose_ok}/{inst_strict_tot} = {100*inst_loose_ok/inst_strict_tot:.1f}%")
    print(f"\nPer-family (strict):")
    for fam in sorted(per_constraint, key=lambda f: -per_constraint[f][1]):
        p, t = per_constraint[fam]
        print(f"  {100*p/t:5.1f}%  {p:3d}/{t:3d}  {fam}")
    print(f"\nPer-subtype (strict):")
    for name in sorted(per_subtype, key=lambda n: -per_subtype[n][1]):
        p, t = per_subtype[name]
        print(f"  {100*p/t:5.1f}%  {p:3d}/{t:3d}  {name}")


if __name__ == "__main__":
    main()
