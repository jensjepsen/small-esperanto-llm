"""Task-induction ICL: can the model infer a transformation rule from
demonstrations alone, with no instruction?

Everything measured so far on this axis has been schema/format extraction --
"pull these keys out of this passage". That is one narrow slice of in-context
learning, and it is the slice the model was trained on. These tasks share the
k-shot shape and nothing else: the rule is never stated, only demonstrated, and
every output is deterministic so scoring is exact string match rather than a
judge.

The rules are chosen to separate kinds of induction:

  surface        uppercase, swapcase, wrap        character-level rewriting
  positional     last_word, first_word, reverse   requires reading position
  aggregate      word_count, digit_sum, acronym   requires computing over parts
  sequence       series_add, series_double        requires inferring an
                                                  arithmetic relation, the only
                                                  family with no textual cue

Vocabulary is sampled from a real Danish corpus rather than a hand-written
list, so the items are not quietly selected to be easy.

The demonstrations use the SAME scaffold the ICL training data uses
("Eksempler:" / "Tekst:" / "Svar:"). A first version used a bare
input-newline-output layout and the model emitted nothing at all on 9 of 12
rules -- 0% nonempty, identical for the SFT and GRPO models. That measured
scaffold familiarity, not rule induction: two things were novel at once. Only
the RULE should be unfamiliar.

Usage:
  python scripts/probe_icl_rules.py --ckpt <path> [--n 40] [--shots 5]
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


def _vocab(n=4000):
    """Danish words from a real corpus -- a hand-written list would encode
    whatever bias the author had about what is 'a normal word'."""
    ds = load_dataset("jensjepsen/danish-sciq", "default", split="test")
    words = []
    for r in ds:
        words += re.findall(r"[a-zæøåA-ZÆØÅ]{3,12}", r.get("da_question") or "")
    seen, out = set(), []
    for w in words:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            out.append(lw)
        if len(out) >= n:
            break
    return out


def make_tasks(vocab, rng):
    """name -> callable() -> (input_str, output_str)"""
    def words(k):
        return rng.sample(vocab, k)

    def uppercase():
        s = " ".join(words(rng.randint(2, 4)))
        return s, s.upper()

    def swapcase():
        s = " ".join(words(rng.randint(2, 3))).capitalize()
        return s, s.swapcase()

    def wrap():
        s = " ".join(words(rng.randint(2, 3)))
        return s, f"<<{s}>>"

    def last_word():
        w = words(rng.randint(3, 5))
        return " ".join(w), w[-1]

    def first_word():
        w = words(rng.randint(3, 5))
        return " ".join(w), w[0]

    def reverse_words():
        w = words(rng.randint(3, 4))
        return " ".join(w), " ".join(reversed(w))

    def word_count():
        w = words(rng.randint(2, 6))
        return " ".join(w), str(len(w))

    def acronym():
        w = words(rng.randint(3, 4))
        return " ".join(w), "".join(x[0] for x in w).upper()

    def sort_alpha():
        w = words(rng.randint(3, 4))
        return " ".join(w), " ".join(sorted(w))

    def digit_sum():
        n = rng.randint(100, 9999)
        return str(n), str(sum(int(c) for c in str(n)))

    def series_add():
        a, d = rng.randint(1, 20), rng.randint(2, 9)
        seq = [a + d * i for i in range(4)]
        return " ".join(map(str, seq)), str(a + d * 4)

    def series_double():
        a = rng.randint(1, 9)
        seq = [a * 2 ** i for i in range(4)]
        return " ".join(map(str, seq)), str(a * 2 ** 4)

    return {
        "uppercase": uppercase, "swapcase": swapcase, "wrap": wrap,
        "first_word": first_word, "last_word": last_word,
        "reverse_words": reverse_words,
        "word_count": word_count, "acronym": acronym, "sort_alpha": sort_alpha,
        "digit_sum": digit_sum,
        "series_add": series_add, "series_double": series_double,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
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

    vocab = _vocab()
    print(f"{args.ckpt}\nvocab {len(vocab)} Danish words, "
          f"n={args.n}/task, {args.shots}-shot, greedy, NO instruction\n",
          flush=True)

    rng = random.Random(args.seed)
    tasks = make_tasks(vocab, rng)
    print(f"{'rule':<15}{'exact':>8}{'nonempty':>10}   example")
    dump, results = [], {}
    for name, fn in tasks.items():
        items = []
        for _ in range(args.n):
            pairs = [fn() for _ in range(args.shots + 1)]
            demos, (qi, qo) = pairs[:-1], pairs[-1]
            body = "Eksempler:\n\n" + "\n\n".join(
                f"Tekst:\n{i}\nSvar: {o}" for i, o in demos)
            body += f"\n\nTekst:\n{qi}\nSvar:"
            items.append((f"{USER}{body}{END}{ASST}", qo, qi))

        outs = []
        for i in range(0, len(items), args.batch_size):
            b = items[i:i + args.batch_size]
            enc = tok([x[0] for x in b], return_tensors="pt", padding=True,
                      add_special_tokens=False,
                      return_token_type_ids=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=args.max_new,
                                   do_sample=False,
                                   pad_token_id=tok.pad_token_id,
                                   eos_token_id=eos)
            pl = enc["input_ids"].shape[1]
            outs += [tok.decode(x[pl:], skip_special_tokens=True).strip()
                     for x in g]

        ok = sum(1 for (_, gold, _), o in zip(items, outs)
                 # first line only: the model often continues with more pairs
                 if o.split("\n")[0].strip() == gold)
        nonempty = sum(1 for o in outs if o.strip())
        results[name] = 100 * ok / len(items)
        ex = f"{items[0][2][:22]!r} -> {outs[0].split(chr(10))[0][:22]!r}"
        print(f"{name:<15}{100*ok/len(items):>7.1f}%{100*nonempty/len(items):>9.1f}%"
              f"   {ex}", flush=True)
        if args.dump:
            for (_, gold, qi), o in list(zip(items, outs))[:12]:
                dump.append({"rule": name, "in": qi, "gold": gold, "pred": o})

    print(f"\nmean over {len(results)} rules: "
          f"{sum(results.values())/len(results):.1f}%")
    if args.dump:
        with open(args.dump, "w") as f:
            for d in dump:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"-> {args.dump}")


if __name__ == "__main__":
    main()
