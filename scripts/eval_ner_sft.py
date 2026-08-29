"""Score danish-ner-sft-v1 predictions, key-value and span-wrap alike.

Three numbers per cell, because they fail independently:

  EXACT      the parsed answer equals the gold answer
  ENTITY F1  micro F1 over (type, span) pairs -- exact match is harsh when a
             passage carries several entities and one is missed
  FAITHFUL   span-wrap only: does stripping the tags recover the passage.
             This is the property that makes span-wrap worth having, and it
             is the one every checkpoint measured so far has failed -- tags
             get opened and not closed.

Broken out per format and per prompt mode, since an average over eleven
formats and three modes hides which ones work.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_icl_schema_format import canon, NULL, SYMBOLS  # noqa: E402
from gen_ner_sft import SPAN_WRAPS, parse_spans  # noqa: E402

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
DATASET = "jensjepsen/danish-ner-sft-v1"


def keyset(r):
    """Keys this row uses, from metadata -- a per-format regex over the
    rendered answer rots every time a format is added."""
    if r["symbols"] == "none":
        return set(r["types"].split("|"))
    return set(SYMBOLS[r["symbols"]][:r["n_types"]])


def pairs(d):
    return {(k, v) for k, vs in (d or {}).items() for v in vs if v != NULL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+", default=["eval", "eval_format"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=320)
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

    print(f"{args.ckpt}\n", flush=True)
    for split in args.splits:
        ds = load_dataset(DATASET, "default", split=split)
        if args.n:
            ds = ds.select(range(min(args.n, len(ds))))
        rows = list(ds)
        st = defaultdict(lambda: [0, 0, 0, 0, 0])   # hit, n, tp, fp, fn
        faith = defaultdict(lambda: [0, 0])
        dump = []
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
                fmt, ks = r["format"], keyset(r)
                gold_s = r["messages"][1]["content"]
                passage = r["messages"][0]["content"].rsplit(
                    "Tekst:\n", 1)[1].split("\nSvar:")[0]
                if fmt in SPAN_WRAPS:
                    G, _ = parse_spans(gold_s, ks, fmt)
                    P, stripped = parse_spans(o, ks, fmt)
                    faith[fmt][1] += 1
                    faith[fmt][0] += (stripped.strip() == passage.strip())
                    faith["ALL"][1] += 1
                    faith["ALL"][0] += (stripped.strip() == passage.strip())
                else:
                    G = canon(gold_s, fmt, ks)
                    P = canon(o, fmt, ks)
                gp, pp = pairs(G), pairs(P)
                ok = P is not None and gp == pp
                for b in ("ALL", f"fmt:{fmt}", f"mode:{r['mode']}"):
                    s = st[b]
                    s[0] += ok; s[1] += 1
                    s[2] += len(gp & pp); s[3] += len(pp - gp); s[4] += len(gp - pp)
                if args.dump and len(dump) < 400:
                    dump.append({"split": split, "format": fmt,
                                 "mode": r["mode"], "symbols": r["symbols"],
                                 "passage": passage, "gold": gold_s, "pred": o})
            d = min(i + args.batch_size, len(rows))
            a = st["ALL"]
            print(f"  [{split}] {d}/{len(rows)}  exact={100*a[0]/a[1]:.1f}%",
                  flush=True)

        def f1(s):
            tp, fp, fn = s[2], s[3], s[4]
            p = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            return 100 * (2 * p * rc / (p + rc) if p + rc else 0.0)

        a = st["ALL"]
        print(f"\n{split.upper()}  n={a[1]}")
        print(f"  exact {100*a[0]/a[1]:.1f}%   entity-F1 {f1(a):.1f}"
              + (f"   faithful {100*faith['ALL'][0]/faith['ALL'][1]:.1f}%"
                 f" (n={faith['ALL'][1]})" if faith["ALL"][1] else ""))
        for b in sorted(k for k in st if k != "ALL"):
            s = st[b]
            extra = ""
            key = b.split(":", 1)[1]
            if key in faith and faith[key][1]:
                extra = f"  faithful {100*faith[key][0]/faith[key][1]:.0f}%"
            print(f"    {b:<22} n={s[1]:<5} exact {100*s[0]/s[1]:>5.1f}%"
                  f"  F1 {f1(s):>5.1f}{extra}")
        print(flush=True)
        if args.dump:
            Path(f"{args.dump}.{split}.json").write_text(
                json.dumps(dump, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
