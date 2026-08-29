"""Score danish-ner-sft-v1 predictions, key-value and span-wrap alike.

Three numbers per cell, because they fail independently:

  EXACT      the parsed answer equals the gold answer
  ENTITY F1  micro F1 over (type, span) pairs -- exact match is harsh when a
             passage carries several entities and one is missed
  STRIP-OK   span-wrap only: stripping the tags recovers the passage.
  TAGGED     span-wrap only: at least one well-formed pair was emitted.
  FAITHFUL   both of the above, over rows whose GOLD carries at least one tag.

The three are separated because STRIP-OK alone is trivially satisfiable: a
model that returns the passage untouched strips back to it perfectly while
extracting nothing. That inflated an earlier reading of the held-out brace
wrapper to 67% "faithful" when 82 of its 131 predictions were bare passages.

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
            # [faithful, n_gold_tagged, strip_ok, tagged, bare_passage]
        faith = defaultdict(lambda: [0, 0, 0, 0, 0])
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
                    # only rows the model was actually asked to tag can
                    # distinguish tagging from doing nothing
                    if pairs(G):
                        strip_ok = stripped.strip() == passage.strip()
                        tagged = bool(pairs(P))
                        bare = o.strip() == passage.strip()
                        for b_ in (fmt, "ALL"):
                            f = faith[b_]
                            f[1] += 1
                            f[2] += strip_ok
                            f[3] += tagged
                            f[0] += (strip_ok and tagged)
                            f[4] += bare
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

        def fstr(b_):
            f = faith.get(b_)
            if not f or not f[1]:
                return ""
            return (f"  faithful {100*f[0]/f[1]:>4.0f}%"
                    f" [strip {100*f[2]/f[1]:>3.0f}%,"
                    f" tagged {100*f[3]/f[1]:>3.0f}%,"
                    f" bare {100*f[4]/f[1]:>3.0f}%]")

        a = st["ALL"]
        print(f"\n{split.upper()}  n={a[1]}")
        print(f"  exact {100*a[0]/a[1]:.1f}%   entity-F1 {f1(a):.1f}"
              + fstr("ALL") + (f" (span rows n={faith['ALL'][1]})"
                               if faith["ALL"][1] else ""))
        for b in sorted(k for k in st if k != "ALL"):
            s = st[b]
            print(f"    {b:<22} n={s[1]:<5} exact {100*s[0]/s[1]:>5.1f}%"
                  f"  F1 {f1(s):>5.1f}{fstr(b.split(':', 1)[1])}")
        print(flush=True)
        if args.dump:
            Path(f"{args.dump}.{split}.json").write_text(
                json.dumps(dump, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
