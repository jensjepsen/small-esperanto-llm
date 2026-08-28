"""Exact-match eval for JSON schema induction.

eval_loss cannot separate the two outcomes this data exists to distinguish.
A model that memorised the 113 training schemas and one that learned to read
a schema off the demonstrations both score well on seen schemas; only the
unseen-schema split tells them apart. So the number that matters is the GAP:

  val   train schemas, passages held out   -> did it learn the task
  eval  schemas never seen in training     -> did it learn to INDUCE

Reported separately, and split again by whether the field names were replaced
with meaning-free symbols. Plain-key rows can be solved from field-name
semantics ("navn" plausibly holds a name); only symbol rows force induction.

Scoring is exact match on the parsed object, since gold is exact by
construction. keys-only match is reported alongside to separate "wrong
schema" from "right schema, wrong values".
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
DATASET = "jensjepsen/danish-icl-json-v1"


def parse(text: str):
    """First balanced {...} in the completion, or None."""
    s = text.find("{")
    if s < 0:
        return None
    depth, instr, esc = 0, False, False
    for i in range(s, len(text)):
        c = text[i]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    o = json.loads(text[s:i + 1])
                    return o if isinstance(o, dict) else None
                except Exception:
                    return None
    return None


def norm(v):
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v).strip().lower()
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def same(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    return all(norm(a[k]) == norm(b[k]) for k in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "eval"])
    ap.add_argument("--n", type=int, default=0, help="0 = full split")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16).cuda().eval()
    eid = tok.convert_tokens_to_ids(END)
    eos = [tok.eos_token_id] + ([eid] if eid != tok.unk_token_id else [])

    dump = []
    print(f"{args.ckpt}\n")
    for split in args.splits:
        ds = load_dataset(DATASET, "default", split=split)
        if args.n:
            ds = ds.select(range(min(args.n, len(ds))))
        rows = list(ds)
        stats = defaultdict(lambda: [0, 0])       # bucket -> [hit, total]
        keyhit = 0
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i:i + args.batch_size]
            prompts = [f"{USER}{r['messages'][0]['content']}{END}{ASST}"
                       for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=False,
                      return_token_type_ids=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=args.max_new,
                                   do_sample=False,
                                   pad_token_id=tok.pad_token_id,
                                   eos_token_id=eos)
            pl = enc["input_ids"].shape[1]
            outs = [tok.decode(x[pl:], skip_special_tokens=True).strip()
                    for x in g]
            for r, o in zip(chunk, outs):
                gold = json.loads(r["messages"][1]["content"])
                pred = parse(o)
                ok = pred is not None and same(pred, gold)
                kk = pred is not None and set(pred) == set(gold)
                keyhit += kk
                sym = "symbol" if r["symbols"] != "none" else "plain"
                for b in ("all", sym, f"shots{r['shots']}"):
                    stats[b][0] += ok
                    stats[b][1] += 1
                if args.dump and len(dump) < 400:
                    dump.append({"split": split, "symbols": r["symbols"],
                                 "shots": r["shots"], "schema": r["schema"],
                                 "gold": r["messages"][1]["content"],
                                 "pred": o[:400], "exact": ok})
            done = min(i + args.batch_size, len(rows))
            print(f"  {split} {done}/{len(rows)}  "
                  f"exact={100*stats['all'][0]/max(1,stats['all'][1]):.1f}%",
                  flush=True)

        h, n = stats["all"]
        print(f"\n{split.upper()}  n={n}")
        print(f"  exact match      {h}/{n}  {100*h/n:.1f}%")
        print(f"  key-set match    {keyhit}/{n}  {100*keyhit/n:.1f}%")
        for b in sorted(stats):
            if b == "all":
                continue
            hh, nn = stats[b]
            print(f"    {b:<10} {hh}/{nn}  {100*hh/max(1,nn):.1f}%")
        print(flush=True)

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
        print(f"-> {args.dump}")


if __name__ == "__main__":
    main()
