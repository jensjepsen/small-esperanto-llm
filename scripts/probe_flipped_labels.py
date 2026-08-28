"""Do the demonstrations override the model's prior about what a field means?

Wei et al. (symbol tuning, arXiv:2305.08298) report that symbol-tuned models
get better at following flipped labels presented in context -- using
in-context information to override prior knowledge. This tests that claim
directly on a model we symbol-tuned.

Method: take a schema whose field names carry real meaning (`navn`, `by`,
`land`) and DERANGE the mapping in every demonstration, so `navn` holds the
city and `by` holds the person. No field maps to itself. The target is
rendered under the same deranged mapping.

Two hypotheses then compete, and every prediction can be scored against both:
  FOLLOWS-DEMOS  matches the deranged gold  -> in-context mapping won
  FOLLOWS-PRIOR  matches the identity gold  -> semantic prior won
A model that ignores the demonstrations and puts the name under `navn` scores
FOLLOWS-PRIOR. Both golds contain the same values, only the key assignment
differs, so this isolates label-binding from extraction ability.

Only plain (meaningful) field names are used -- with `alfa`/`kat_a` there is
no prior to flip, so the question is undefined.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_icl_json import canon, render  # noqa: E402

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
SOURCE = "jensjepsen/danish-json-grpo-v1"


def derange(fields, rng):
    """Cyclic shift: no field keeps its own name, and the permutation is the
    same for every row in a trial so the demonstrations are self-consistent."""
    k = rng.randrange(1, len(fields))
    return {f: fields[(i + k) % len(fields)] for i, f in enumerate(fields)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--format", default="kv_colon")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    ds = load_dataset(SOURCE, split="train")
    groups = defaultdict(list)
    for r in ds:
        if len((r["passage"] or "").strip()) < 15:
            continue
        try:
            g = json.loads(r["gold_values"])
        except Exception:
            continue
        # every field must hold a non-empty scalar, or a deranged assignment
        # is indistinguishable from the identity one
        fs = list(r["fields"])
        if not (2 <= len(fs) <= 4):
            continue
        if any(not isinstance(g.get(f), (str, int, float))
               or isinstance(g.get(f), bool) or g.get(f) in (None, "")
               for f in fs):
            continue
        groups[tuple(fs)].append(r)
    groups = {k: v for k, v in groups.items() if len(v) >= args.shots + 1}
    print(f"{len(groups)} usable schemas", flush=True)

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16).cuda().eval()
    eid = tok.convert_tokens_to_ids(END)
    eos = [tok.eos_token_id] + ([eid] if eid != tok.unk_token_id else [])

    rng = random.Random(11)
    keys_ = sorted(groups)
    prompts, golds, priors = [], [], []
    for _ in range(args.n):
        fields = list(rng.choice(keys_))
        picks = rng.sample(groups[tuple(fields)], args.shots + 1)
        km = derange(fields, rng)
        ident = {f: f for f in fields}
        parts = ["Eksempler:"]
        for d in picks[:-1]:
            gv = json.loads(d["gold_values"])
            parts.append(f'Tekst:\n{d["passage"].strip()}\n'
                         f'Svar: {render(gv, fields, km, args.format)}')
        t = picks[-1]
        parts.append(f'Tekst:\n{t["passage"].strip()}\nSvar:')
        tv = json.loads(t["gold_values"])
        prompts.append(f"{USER}" + "\n\n".join(parts) + f"{END}{ASST}")
        golds.append(render(tv, fields, km, args.format))
        priors.append(render(tv, fields, ident, args.format))

    keyset = set(k for f in keys_ for k in f)
    demo_w = prior_w = neither = unparsed = 0
    dump = []
    for i in range(0, len(prompts), args.batch_size):
        enc = tok(prompts[i:i + args.batch_size], return_tensors="pt",
                  padding=True, add_special_tokens=False,
                  return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=args.max_new,
                               do_sample=False, pad_token_id=tok.pad_token_id,
                               eos_token_id=eos)
        pl = enc["input_ids"].shape[1]
        outs = [tok.decode(x[pl:], skip_special_tokens=True).strip()
                for x in g]
        for j, o in enumerate(outs):
            k = i + j
            p = canon(o, args.format, keyset)
            gd = canon(golds[k], args.format, keyset)
            pr = canon(priors[k], args.format, keyset)
            if p is None:
                unparsed += 1
            elif p == gd:
                demo_w += 1
            elif p == pr:
                prior_w += 1
            else:
                neither += 1
            if args.dump and len(dump) < 200:
                dump.append({"prompt": prompts[k], "deranged_gold": golds[k],
                             "prior_gold": priors[k], "pred": o})
        done = min(i + args.batch_size, len(prompts))
        print(f"  {done}/{len(prompts)}  demos={demo_w} prior={prior_w} "
              f"neither={neither} unparsed={unparsed}", flush=True)

    n = len(prompts)
    print(f"\n{args.ckpt}  format={args.format}  {args.shots}-shot  n={n}")
    print(f"  FOLLOWS-DEMOS (deranged mapping) {demo_w:>4}  {100*demo_w/n:.1f}%")
    print(f"  FOLLOWS-PRIOR (field semantics)  {prior_w:>4}  {100*prior_w/n:.1f}%")
    print(f"  neither                          {neither:>4}  {100*neither/n:.1f}%")
    print(f"  unparseable                      {unparsed:>4}  {100*unparsed/n:.1f}%")
    if args.dump:
        Path(args.dump).write_text(json.dumps(dump, ensure_ascii=False,
                                              indent=2))
        print(f"-> {args.dump}")


if __name__ == "__main__":
    main()
