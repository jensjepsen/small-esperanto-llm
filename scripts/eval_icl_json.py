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
from pathlib import Path
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_icl_json import canon  # noqa: E402

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
DATASET = "jensjepsen/danish-icl-json-v2"


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


def _keys(r):
    """Key names this row's answer uses. canon() needs them to build the
    per-format regex; taking them from GOLD (not from the prediction) means a
    model inventing a key simply fails to parse, which is correct."""
    g = r["messages"][1]["content"]
    if r.get("format", "json") == "json":
        try:
            return set(json.loads(g[g.find("{"):g.rfind("}") + 1]))
        except Exception:
            return set()
    pats = {"kv_colon": r"^\s*([^\s:]+)\s*:", "kv_eq": r"^\s*([^\s=]+)\s*=",
            "kv_bracket": r"^\s*\[([^\]]+)\]", "kv_arrow": r"->\s*(\S+)\s*$",
            "numbered": r"^\s*\d+\.\s*([^\s:]+)\s*:", "tsv": r"^([^\t]+)\t",
            "tagged": r"<([^/>]+)>"}
    return set(re.findall(pats[r["format"]], g, re.M))


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
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--splits", nargs="+",
                    default=["val", "eval_schema", "eval_format", "eval_both"])
    ap.add_argument("--n", type=int, default=0, help="0 = full split")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--dump", default=None,
                    help="JSONL prefix; writes <prefix>.<split>.jsonl with one "
                         "record per row including the full prompt, so a wrong "
                         "answer can be judged against the demonstrations it "
                         "was given")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16).cuda().eval()
    eid = tok.convert_tokens_to_ids(END)
    eos = [tok.eos_token_id] + ([eid] if eid != tok.unk_token_id else [])

    print(f"{args.ckpt}\n")
    for split in args.splits:
        jf = open(f"{args.dump}.{split}.jsonl", "w") if args.dump else None
        ds = load_dataset(args.dataset, "default", split=split)
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
                # v2 rows carry a format; v1 rows are all json
                fmt = r.get("format", "json")
                gold = canon(r["messages"][1]["content"], fmt, _keys(r))
                pred = canon(o, fmt, _keys(r))
                ok = pred is not None and gold is not None and pred == gold
                kk = (pred is not None and gold is not None
                      and set(pred) == set(gold))
                keyhit += kk
                sym = "symbol" if r["symbols"] != "none" else "plain"
                for b in ("all", sym, f"shots{r['shots']}",
                          f"fmt:{fmt}"):
                    stats[b][0] += ok
                    stats[b][1] += 1
                if jf:
                    # every row, not a 400-row prefix, and the prompt is
                    # included: without the demonstrations there is no way to
                    # tell a model error from an unanswerable row
                    jf.write(json.dumps({
                        "split": split, "schema": r["schema"],
                        "symbols": r["symbols"], "shots": r["shots"],
                        "task_type": r["task_type"], "domain": r["domain"],
                        "format": fmt,
                        "prompt": r["messages"][0]["content"],
                        "gold": r["messages"][1]["content"],
                        "pred": o, "exact": bool(ok), "keys_ok": bool(kk),
                    }, ensure_ascii=False) + "\n")
            # Running numbers per batch, not just a final line: these evals
            # take minutes per checkpoint and a partial result is worth
            # seeing. symbol/plain are broken out because plain-key rows are
            # partly solvable from field-name semantics, so a headline that
            # mixes them can look healthy while induction is flat.
            done = min(i + args.batch_size, len(rows))
            def pct(b_):
                h_, n_ = stats[b_]
                return f"{100*h_/n_:.1f}%" if n_ else "  -  "
            print(f"  [{split}] {done:>4}/{len(rows)}  "
                  f"exact={pct('all')}  symbol={pct('symbol')}  "
                  f"plain={pct('plain')}  keys={100*keyhit/max(1,done):.1f}%",
                  flush=True)
            if jf:
                jf.flush()      # partial dump survives a killed run

        h, n = stats["all"]
        print(f"\n{split.upper()}  n={n}")
        print(f"  exact match      {h}/{n}  {100*h/n:.1f}%")
        print(f"  key-set match    {keyhit}/{n}  {100*keyhit/n:.1f}%")
        for b in sorted(stats):
            if b == "all":
                continue
            hh, nn = stats[b]
            print(f"    {b:<10} {hh}/{nn}  {100*hh/max(1,nn):.1f}%")
        if jf:
            jf.close()
            print(f"  -> {args.dump}.{split}.jsonl")
        print(flush=True)


if __name__ == "__main__":
    main()
