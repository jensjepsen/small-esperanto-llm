"""How much does each additional demonstration buy?

Sweeps shot count on SEEN and UNSEEN schemas, with meaningful and with
meaning-free keys. Training used 1-5 shots, so 6 and 8 also probe past the
trained range.

The 0-shot cell is the interesting control rather than a throwaway. With no
demonstrations the prompt carries no schema at all -- just a passage and
"Svar:". So:

  0-shot, UNSEEN schema   should be ~0. Anything else means leakage.
  0-shot, SEEN schema     is a MEMORISATION probe: a model that has bound
                          "this kind of passage -> these field names" during
                          training can answer without being shown, and that
                          would inflate every seen-schema number elsewhere.
  0-shot, SYMBOL keys     is unanswerable by construction -- nothing can
                          reveal that `alfa` means the person -- so it is the
                          floor the other cells are read against.

Seen/unseen is decided by the same hash the generator used, so the partition
matches the trained model's actual experience.
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
from gen_icl_schema_format import canon, render, is_heldout, SYMBOLS  # noqa: E402

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
SOURCE = "jensjepsen/danish-json-grpo-v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--format", default="kv_colon")
    ap.add_argument("--shots", nargs="+", type=int,
                    default=[0, 1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--heldout-frac", type=float, default=0.2,
                    help="must match the value the training data was built with")
    args = ap.parse_args()

    ds = load_dataset(SOURCE, split="train")
    groups = defaultdict(list)
    for r in ds:
        if len((r["passage"] or "").strip()) < 15:
            continue
        try:
            json.loads(r["gold_values"])
        except Exception:
            continue
        groups[tuple(r["fields"])].append(r)
    big = {k: v for k, v in groups.items()
           if len(v) >= max(args.shots) + 1 and 2 <= len(k) <= 5}
    seen = [k for k in big if not is_heldout(k, args.heldout_frac)]
    unseen = [k for k in big if is_heldout(k, args.heldout_frac)]
    print(f"{len(seen)} seen schemas, {len(unseen)} unseen "
          f"(>= {max(args.shots)+1} rows each)\n", flush=True)

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16).cuda().eval()
    eid = tok.convert_tokens_to_ids(END)
    eos = [tok.eos_token_id] + ([eid] if eid != tok.unk_token_id else [])

    def run(pool, keymode, shots, rng):
        prompts, golds, keysets = [], [], []
        for _ in range(args.n):
            fields = list(rng.choice(pool))
            picks = rng.sample(big[tuple(fields)], shots + 1)
            if keymode == "plain":
                km = {f: f for f in fields}
            else:
                syms = SYMBOLS["greek"][:len(fields)]
                sh = list(syms)
                rng.shuffle(sh)
                km = dict(zip(fields, sh))
            parts = ["Eksempler:"] if shots else []
            for d in picks[:-1]:
                parts.append(
                    f'Tekst:\n{d["passage"].strip()}\nSvar: '
                    f'{render(json.loads(d["gold_values"]), fields, km, args.format)}')
            t = picks[-1]
            parts.append(f'Tekst:\n{t["passage"].strip()}\nSvar:')
            prompts.append(f"{USER}" + "\n\n".join(parts) + f"{END}{ASST}")
            golds.append(render(json.loads(t["gold_values"]), fields, km,
                                args.format))
            keysets.append(set(km.values()))
        ok = 0
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
            for j, x in enumerate(g):
                k = i + j
                o = tok.decode(x[pl:], skip_special_tokens=True).strip()
                p = canon(o, args.format, keysets[k])
                ok += (p is not None
                       and p == canon(golds[k], args.format, keysets[k]))
        return 100 * ok / len(prompts)

    print(f"exact match, format={args.format}, n={args.n} per cell\n")
    print(f"{'shots':<7}{'seen/plain':>12}{'seen/symbol':>13}"
          f"{'unseen/plain':>14}{'unseen/symbol':>15}")
    for s in args.shots:
        cells = []
        for pool, keymode in ((seen, "plain"), (seen, "symbol"),
                              (unseen, "plain"), (unseen, "symbol")):
            # same seed per cell so every cell sees the same schema draws
            cells.append(run(pool, keymode, s, random.Random(1000 + s)))
        print(f"{s:<7}" + "".join(f"{c:>12.1f}%" if i == 0 else
                                  f"{c:>12.1f}%" for i, c in enumerate(cells)),
              flush=True)


if __name__ == "__main__":
    main()
