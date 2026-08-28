"""Does the learned output format survive a task the model never trained on?

The ICL training set is entirely danish-json-grpo-v1: short product /
recipe / API-error passages, extract-rewrite-fill_template. This probe keeps
the prompt SHAPE and the output FORMAT identical to training and swaps the
task and the domain for dane_plus NER — news prose, entity extraction.

That isolates one thing. If format ability is general, a trained format
(`tagged`, `kv_colon`) should work here; if it is welded to the training
distribution, it will not. `bracket_pair` is carried along as a control: it
was held out of training and scored 0.0% in-distribution, so it should score
0 here too, and if it does not the comparison is confounded.

Context for the numbers: at the start of this work the v31 base produced 0
grounded spans and 0 valid `type: enhed` lines in 60 samples on exactly this
dane_plus subset.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_icl_json import canon, render, NULL  # noqa: E402

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
CANON = {"PERSON": "person", "PER": "person", "ORGANIZATION": "org",
         "ORG": "org", "GPE": "sted", "LOCATION": "sted", "LOC": "sted",
         "FACILITY": "sted"}
FIELDS = ["person", "org", "sted"]
KEYMAPS = {
    "plain": {"person": "person", "org": "organisation", "sted": "sted"},
    "symbol": {"person": "alfa", "org": "beta", "sted": "gamma"},
}


def load_rows(split, maxlen=200):
    out = []
    for r in load_dataset("KennethEnevoldsen/dane_plus", split=split):
        t = (r.get("text") or "").strip()
        if not t or len(t) > maxlen:
            continue
        g = {f: [] for f in FIELDS}
        for e in sorted(r["ents"] or [], key=lambda e: e["start"]):
            lab = CANON.get(str(e.get("label", "")).upper())
            s = t[e["start"]:e["end"]].strip()
            if lab and s and s not in g[lab]:
                g[lab].append(s)
        out.append({"text": t, "gold": g,
                    "surf": {v.lower() for vs in g.values() for v in vs}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--formats", nargs="+",
                    default=["tagged", "kv_colon", "bracket_pair"])
    ap.add_argument("--keymap", default="plain", choices=list(KEYMAPS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    km = KEYMAPS[args.keymap]
    keys = set(km.values())
    dev, test = load_rows("dev"), load_rows("test")
    rng = random.Random(5)
    targets = [r for r in test if any(r["gold"].values())]
    targets = rng.sample(targets, min(args.n, len(targets)))
    pool = [r for r in dev if sum(len(v) for v in r["gold"].values()) >= 2]

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16).cuda().eval()
    eid = tok.convert_tokens_to_ids(END)
    eos = [tok.eos_token_id] + ([eid] if eid != tok.unk_token_id else [])

    print(f"{args.ckpt}\ndane_plus NER as an ICL task | keymap={args.keymap} "
          f"| {args.shots}-shot | n={len(targets)}\n", flush=True)
    dump = []
    for fmt in args.formats:
        parsed = exact = 0
        tp = fp = fn = 0
        prompts, metas = [], []
        for t in targets:
            # anti-leak, same discipline as the training data: an exemplar
            # must not hand over an entity the target needs
            cands = [d for d in pool if not (d["surf"] & t["surf"])
                     and not any(s in t["text"].lower() for s in d["surf"])]
            picks = rng.sample(cands, min(args.shots, len(cands)))
            parts = ["Eksempler:"]
            for d in picks:
                parts.append(f'Tekst:\n{d["text"]}\n'
                             f'Svar: {render(d["gold"], FIELDS, km, fmt)}')
            parts.append(f'Tekst:\n{t["text"]}\nSvar:')
            prompts.append(f"{USER}" + "\n\n".join(parts) + f"{END}{ASST}")
            metas.append(t)

        for i in range(0, len(prompts), args.batch_size):
            enc = tok(prompts[i:i + args.batch_size], return_tensors="pt",
                      padding=True, add_special_tokens=False,
                      return_token_type_ids=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=args.max_new,
                                   do_sample=False,
                                   pad_token_id=tok.pad_token_id,
                                   eos_token_id=eos)
            pl = enc["input_ids"].shape[1]
            outs = [tok.decode(x[pl:], skip_special_tokens=True).strip()
                    for x in g]
            for t, o in zip(metas[i:i + args.batch_size], outs):
                gold = canon(render(t["gold"], FIELDS, km, fmt), fmt, keys)
                pred = canon(o, fmt, keys)
                if pred is not None:
                    parsed += 1
                    exact += (pred == gold)
                    P = {(k, v) for k, vs in pred.items() for v in vs
                         if v != NULL}
                    G = {(k, v) for k, vs in gold.items() for v in vs
                         if v != NULL}
                    tp += len(P & G); fp += len(P - G); fn += len(G - P)
                else:
                    fn += sum(1 for vs in gold.values() for v in vs
                              if v != NULL)
                if args.dump and len(dump) < 300:
                    dump.append({"format": fmt, "text": t["text"],
                                 "gold": render(t["gold"], FIELDS, km, fmt),
                                 "pred": o})
            done = min(i + args.batch_size, len(prompts))
            print(f"  [{fmt}] {done}/{len(prompts)}  "
                  f"parsed={100*parsed/done:.0f}%  exact={100*exact/done:.0f}%",
                  flush=True)

        n = len(prompts)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        print(f"\n{fmt}:  parses {parsed}/{n} ({100*parsed/n:.0f}%)   "
              f"exact-set {exact}/{n} ({100*exact/n:.0f}%)   "
              f"entity P/R/F1 {100*pr:.1f}/{100*rc:.1f}/{100*f1:.1f}\n",
              flush=True)

    if args.dump:
        Path(args.dump).write_text(json.dumps(dump, ensure_ascii=False,
                                              indent=2))
        print(f"-> {args.dump}")


if __name__ == "__main__":
    main()
