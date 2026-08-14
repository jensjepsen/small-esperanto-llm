"""Open-ended SciQ eval — ask the question WITHOUT multiple-choice options,
compare model's free-form answer against the gold answer text via light
Danish normalization + substring match.

Removes the position-bias artifact from the letter-based generative eval.
"""
import argparse
import json
import re
import time
import unicodedata

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"

DA_STOP = {"en", "et", "den", "det", "de", "at", "og", "i", "på", "af",
           "til", "for", "med", "er", "som", "der"}


def norm(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w for w in s.split() if w not in DA_STOP)


def matches(pred: str, gold: str) -> bool:
    ng = norm(gold); np_ = norm(pred)
    return bool(ng) and ng in np_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--split", default="test")
    ap.add_argument("--dump", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--k", type=int, default=1,
                    help="Samples per question. k=1 = greedy. k>1 = pass@k "
                         "with temperature/top_p sampling.")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.float16).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id]
    if end_id != tok.unk_token_id:
        eos_ids.append(end_id)

    ds = load_dataset("jensjepsen/danish-sciq", "default", split=args.split)
    n = len(ds)

    dump_f = open(args.dump, "w") if args.dump else None
    n_ok_any = 0     # pass@k (any of k samples matches)
    n_ok_first = 0   # pass@1 (first sample)
    t0 = time.time()
    rows = list(ds)
    for bstart in range(0, n, args.batch_size):
        batch = rows[bstart:bstart + args.batch_size]
        B = len(batch)
        qs = [r["da_question"] for r in batch]
        golds = [r["da_correct_answer"] for r in batch]
        prompts = [f"{USER}{q}{END}{ASST}" for q in qs]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False, return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            gen_kwargs = dict(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_new,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids,
                repetition_penalty=1.1,
                num_return_sequences=args.k,
            )
            if args.k > 1:
                gen_kwargs.update(do_sample=True, temperature=args.temp, top_p=args.top_p)
            else:
                gen_kwargs.update(do_sample=False, num_beams=1)
            out = model.generate(**gen_kwargs)
        plen = enc["input_ids"].shape[1]
        for row_ix in range(B):
            gens, corrects = [], []
            for sample_ix in range(args.k):
                seq_ix = row_ix * args.k + sample_ix
                g = tok.decode(out[seq_ix, plen:], skip_special_tokens=True).strip()
                ok = matches(g, golds[row_ix])
                gens.append(g); corrects.append(bool(ok))
            n_ok_any += any(corrects)
            n_ok_first += corrects[0]
            if dump_f:
                dump_f.write(json.dumps({
                    "idx": bstart + row_ix, "q": qs[row_ix], "gold": golds[row_ix],
                    "gens": gens, "corrects": corrects,
                }, ensure_ascii=False) + "\n")
        i = bstart + B
        if (i // 100) != ((i - B) // 100) or i == n:
            el = time.time() - t0
            print(f"  {i}/{n} pass@{args.k}={n_ok_any/i:.3f} "
                  f"pass@1={n_ok_first/i:.3f} eta={el*(n-i)/i:.0f}s", flush=True)
    if dump_f: dump_f.close()
    print(f"\nSciQ[da] open-Q  n={n} k={args.k}:")
    print(f"  pass@{args.k}: {100*n_ok_any/n:.2f}%  ({n_ok_any}/{n})")
    print(f"  pass@1:  {100*n_ok_first/n:.2f}%  ({n_ok_first}/{n})")


if __name__ == "__main__":
    main()
