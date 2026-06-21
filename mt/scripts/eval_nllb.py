"""Eval NLLB-200-distilled-600M on the same FLORES devtest JSONL we use.

Uses sacrebleu (same metric flavor as our eval_bleu.py) for an apples-to-apples
comparison with our own model.

Direction: en→eo by default; flag --direction eo2en flips it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import sacrebleu
import torch
from tqdm import tqdm


NLLB_LANG = {"en": "eng_Latn", "eo": "epo_Latn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--direction", default="en2eo", choices=["en2eo", "eo2en"])
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--hf-cache", default="/mnt/data/hf_cache")
    ap.add_argument("--lowercase", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Lowercase preds AND refs before scoring (default on, "
                         "matches eval_bleu.py default). Disable with "
                         "--no-lowercase to see NLLB's true cased perf.")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_cache)

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    src_lang, tgt_lang = ("en", "eo") if args.direction == "en2eo" else ("eo", "en")
    nllb_src = NLLB_LANG[src_lang]
    nllb_tgt = NLLB_LANG[tgt_lang]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading {args.model} on {device} … (src_lang={nllb_src}, tgt={nllb_tgt})")
    tok = AutoTokenizer.from_pretrained(args.model, src_lang=nllb_src)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if device == "cuda":
        model = model.half().to(device).eval()
    else:
        model = model.to(device).eval()
    forced_bos = tok.convert_tokens_to_ids(nllb_tgt)

    pairs = []
    with args.eval.open() as f:
        for line in f:
            r = json.loads(line)
            pairs.append((r[src_lang], r[tgt_lang]))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Evaluating {len(pairs)} pairs  direction={args.direction}")

    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), args.batch_size)):
            batch = pairs[i : i + args.batch_size]
            srcs = [s for s, _ in batch]
            enc = tok(srcs, padding=True, truncation=True, max_length=args.max_length,
                       return_tensors="pt").to(device)
            out = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                num_beams=args.num_beams,
                max_length=args.max_length,
                early_stopping=True,
                no_repeat_ngram_size=5,
            )
            for seq in out:
                preds.append(tok.decode(seq, skip_special_tokens=True))

    refs = [g for _, g in pairs]
    srcs_all = [s for s, _ in pairs]
    if args.lowercase:
        preds_for_score = [p.lower() for p in preds]
        refs_for_score = [r.lower() for r in refs]
    else:
        preds_for_score = preds
        refs_for_score = refs
    bleu = sacrebleu.corpus_bleu(preds_for_score, [refs_for_score])
    chrf = sacrebleu.corpus_chrf(preds_for_score, [refs_for_score])
    chrfpp = sacrebleu.corpus_chrf(preds_for_score, [refs_for_score], word_order=2)
    print(f"\n=== {args.eval.name} | {args.direction} | beam={args.num_beams}"
          f" | lowercase={args.lowercase} ===")
    print(f"BLEU:   {bleu.score:.2f}    {bleu.format(width=2)}")
    print(f"chrF:   {chrf.score:.2f}")
    print(f"chrF++: {chrfpp.score:.2f}")
    print(f"N:      {len(pairs)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for s, g, p in zip(srcs_all, refs, preds):
                f.write(json.dumps({"src": s, "gold": g, "pred": p}, ensure_ascii=False) + "\n")
        print(f"Per-record predictions -> {args.out}")


if __name__ == "__main__":
    main()
