"""Eval Danish citizen tests with GENERATION (not letter-selection).

Ask the question directly (no A/B/C shown), let the model generate the answer
text, then substring-match against the gold option's Danish text.

Same shape as eval_triviaqa_eo.py but Danish + no morpheme preprocess.
"""
import argparse
import re
import sys
import unicodedata

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"

DA_ARTICLES = {"en", "et", "den", "det", "de", "at", "og", "i", "på", "af",
               "til", "for", "med", "er", "har", "har", "som"}


def normalize(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    words = [w for w in s.split() if w not in DA_ARTICLES]
    return " ".join(words)


def matches(pred: str, gold_text: str) -> bool:
    np_, ng = normalize(pred), normalize(gold_text)
    return bool(ng) and ng in np_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--prompt-format", choices=["chat", "spm"], default="chat",
                    help="chat=<|user|>…<|assistant|> (SFT). "
                         "spm='Spørgsmål: …\\nSvar:' (base-LM pretrain format).")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.float16).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)

    ds = load_dataset("alexandrainst/danish-citizen-tests", split="train")
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))
    n = len(ds)
    print(f"  {n} rows, format={args.prompt_format}, bs={args.batch_size}", flush=True)

    rows_list = list(ds)
    n_ok = 0
    for bstart in range(0, n, args.batch_size):
        batch = rows_list[bstart:bstart + args.batch_size]
        B = len(batch)
        qs = [r["question"] for r in batch]
        gold_texts = [r[f"option_{r['answer'].lower()}"] for r in batch]

        if args.prompt_format == "chat":
            prompts = [f"{USER}{q}{END}{ASST}" for q in qs]
            eos_id = [tok.eos_token_id, end_id] if end_id != tok.unk_token_id else tok.eos_token_id
        else:
            prompts = [f"Spørgsmål: {q}\nSvar:" for q in qs]
            eos_id = tok.eos_token_id

        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False, return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_new, do_sample=False, num_beams=1,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=eos_id,
            )
        plen = enc["input_ids"].shape[1]
        for row_ix in range(B):
            pred = tok.decode(out[row_ix, plen:], skip_special_tokens=True).strip()
            ok = matches(pred, gold_texts[row_ix])
            n_ok += ok
            i = bstart + row_ix + 1
            if args.verbose or i <= 8 or (not ok and i <= 20):
                flag = "✓" if ok else "✗"
                print(f"[{i}/{n}] {flag} gold={gold_texts[row_ix]!r}", flush=True)
                print(f"   Q: {qs[row_ix]}")
                print(f"   pred: {pred[:180]}")
        i = bstart + B
        if (i // 100) != ((i - B) // 100) or i == n:
            print(f"  {i}/{n} acc={n_ok/i:.3f}", flush=True)

    print(f"\n=== citizen (gen) {n_ok}/{n} = {100*n_ok/n:.1f}% ===")


if __name__ == "__main__":
    main()
