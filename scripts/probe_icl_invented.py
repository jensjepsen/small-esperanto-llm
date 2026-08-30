"""Can the model induce a format/symbol class it has never seen, from
demonstrations alone?

Every format in danish-icl-schema-format and danish-ner-sft is drawn from one
fixed table of 14, and every symbol scheme from one table of 4. Both eval
splits hold formats OUT of training, but they hold out *members of the same
table* -- so "unseen format" has so far meant "unseen delimiter from a family
the model has seen". This asks the harder question with delimiters and symbol
classes that appear in no dataset at all.

Conditions (each 4-shot, greedy, same underlying rows):

  control     `key: value`            trained format, real key names
  tilde       `key ~> value`          delimiter in no training set
  arrowback   `value <= key`          novel delimiter AND inverted order --
                                      separates "new token" from "new grammar"
  emoji       `🔵: value`             trained delimiter, key is an emoji --
                                      a symbol CLASS outside greek/kat/fnum/foo
  wrapped     `((key)) value`         doubled bracket, novel

Rows come from danish-json-grpo-v1 (passage + fields + gold_values), the source
the ICL sets are generated from, so gold is exact rather than judged. Scoring
is per-condition exact match on the parsed key->value map, plus key-set match
to separate "wrong format" from "wrong values" -- the same split that showed
the model reaches 93-96% keys while exact sits far lower.

Usage:
  python scripts/probe_icl_invented.py --ckpt <path> [--n 60] [--shots 4]
"""
from __future__ import annotations

import argparse
import json
import random
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
EMOJI = ["🔵", "🟢", "🔴", "🟡", "🟣", "⚫", "⚪", "🟠"]


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


# render(key, value) -> line   /   parse(text, keys) -> {key: value}
def _mk(render, pattern, swap=False):
    def parse(text, keys):
        # MULTILINE: the answer is one key/value per LINE, so ^ and $ must
        # anchor per line. Without it finditer matches only the whole string
        # and every condition silently scores 0% -- which reads as "the model
        # cannot do this" rather than "the probe is broken".
        out = {}
        for m in re.finditer(pattern, text, re.MULTILINE):
            a, b = m.group(1).strip(), m.group(2).strip()
            k, v = (b, a) if swap else (a, b)
            if k in keys and k not in out:
                out[k] = norm(v)
        return out
    return render, parse


CONDS = {
    "control":   _mk(lambda k, v: f"{k}: {v}",      r"^\s*([^\n:]+?)\s*:\s*(.+)$"),
    "tilde":     _mk(lambda k, v: f"{k} ~> {v}",    r"^\s*(.+?)\s*~>\s*(.+)$"),
    "arrowback": _mk(lambda k, v: f"{v} <= {k}",    r"^\s*(.+?)\s*<=\s*(.+)$", swap=True),
    "emoji":     _mk(lambda k, v: f"{k}: {v}",      r"^\s*([^\n:]+?)\s*:\s*(.+)$"),
    "wrapped":   _mk(lambda k, v: f"(({k})) {v}",   r"^\s*\(\((.+?)\)\)\s*(.+)$"),
}


def build(rows, cond, shots, rng, emoji_keys):
    render, _ = CONDS[cond]
    demos, target = rows[:shots], rows[shots]

    def block(r):
        km = r["_keymap"]
        lines = [render(km[k], norm(r["gold"][k]))
                 for k in r["fields"] if k in r["gold"]]
        return f"Tekst:\n{r['passage']}\nSvar:\n" + "\n".join(lines)

    body = "Eksempler:\n\n" + "\n\n".join(block(d) for d in demos)
    body += f"\n\nTekst:\n{target['passage']}\nSvar:\n"
    return body, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
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

    ds = load_dataset("jensjepsen/danish-json-grpo-v1", split="train")
    pool = []
    for r in ds:
        p = (r.get("passage") or "").strip()
        if len(p) < 20:
            continue
        try:
            g = json.loads(r["gold_values"])
        except Exception:
            continue
        if not isinstance(g, dict):
            continue
        f = [k for k in r["fields"] if isinstance(g.get(k), (str, int, float))]
        if len(f) < 2:
            continue
        pool.append({"passage": p, "fields": f[:4],
                     "gold": {k: g[k] for k in f[:4]}})
    # group by identical field-set so demonstrations and query share a schema --
    # otherwise the model has no schema to induce, only a format
    by_schema = {}
    for r in pool:
        by_schema.setdefault(tuple(r["fields"]), []).append(r)
    groups = [v for v in by_schema.values() if len(v) >= args.shots + 1]
    print(f"{len(pool)} rows, {len(groups)} schemas with >= {args.shots+1} rows",
          flush=True)

    rng = random.Random(args.seed)
    print(f"\n{'condition':<12}{'exact':>9}{'keys':>8}{'parsed':>9}   n={args.n}, "
          f"{args.shots}-shot, greedy")
    dump = []
    for cond in CONDS:
        _, parse = CONDS[cond]
        items = []
        r2 = random.Random(args.seed)
        for _ in range(args.n):
            g = r2.choice(groups)
            rows = r2.sample(g, args.shots + 1)
            km = {k: (EMOJI[i % len(EMOJI)] if cond == "emoji" else k)
                  for i, k in enumerate(rows[0]["fields"])}
            for r in rows:
                r = dict(r); r["_keymap"] = km
                items.append(r)
            rows = [dict(x, _keymap=km) for x in rows]
            body, target = build(rows, cond, args.shots, r2, cond == "emoji")
            items.append({"prompt": f"{USER}{body}{END}{ASST}",
                          "gold": {km[k]: norm(v) for k, v in target["gold"].items()},
                          "keys": set(km.values())})
        items = [x for x in items if "prompt" in x][:args.n]

        outs = []
        for i in range(0, len(items), args.batch_size):
            b = items[i:i + args.batch_size]
            enc = tok([x["prompt"] for x in b], return_tensors="pt",
                      padding=True, add_special_tokens=False,
                      return_token_type_ids=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id,
                                     eos_token_id=eos)
            pl = enc["input_ids"].shape[1]
            outs += [tok.decode(x[pl:], skip_special_tokens=True).strip()
                     for x in gen]

        ok = keys_ok = parsed = 0
        for x, o in zip(items, outs):
            p = parse(o, x["keys"])
            if p:
                parsed += 1
            if set(p) == x["keys"]:
                keys_ok += 1
            if p == x["gold"]:
                ok += 1
            if args.dump and len(dump) < 200:
                dump.append({"cond": cond, "pred": o, "parsed": p,
                             "gold": x["gold"]})
        n = len(items)
        print(f"{cond:<12}{100*ok/n:>8.1f}%{100*keys_ok/n:>7.1f}%"
              f"{100*parsed/n:>8.1f}%", flush=True)

    if args.dump:
        with open(args.dump, "w") as f:
            for d in dump:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"\n-> {args.dump}")


if __name__ == "__main__":
    main()
