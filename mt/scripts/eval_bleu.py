"""Sacrebleu eval of a checkpoint on a JSONL of {en, eo} pairs.

Generates en→eo with beam search and reports BLEU + chrF.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sacrebleu
import torch
from tqdm import tqdm
from transformers import MarianMTModel

sys.path.insert(0, str(Path(__file__).parent))
from sp_tokenizer import SPMTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default="mt/data/tokenizer/spm_eneo_32k.model")
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--direction", default="en2eo", choices=["en2eo", "eo2en"])
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional: dump per-record (src, gold, pred) JSONL")
    args = ap.parse_args()

    tok = SPMTokenizer(args.tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.checkpoint} -> {device}")
    model = MarianMTModel.from_pretrained(args.checkpoint).to(device).eval()

    src_lang, tgt_lang = ("en", "eo") if args.direction == "en2eo" else ("eo", "en")

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
            ids_list = [tok.encode(src, lang=tgt_lang) for src, _ in batch]
            be = tok.pad_batch(ids_list)
            input_ids = be.input_ids.to(device)
            attn = be.attention_mask.to(device)
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                num_beams=args.num_beams,
                max_length=args.max_length,
                early_stopping=True,
                no_repeat_ngram_size=5,
            )
            for seq in out:
                preds.append(tok.decode(seq))

    refs = [g for _, g in pairs]
    srcs = [s for s, _ in pairs]
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    chrfpp = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    print(f"\n=== {args.eval.name} | {args.direction} | beam={args.num_beams} ===")
    print(f"BLEU:   {bleu.score:.2f}    {bleu.format(width=2)}")
    print(f"chrF:   {chrf.score:.2f}")
    print(f"chrF++: {chrfpp.score:.2f}")
    print(f"N:      {len(pairs)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for s, g, p in zip(srcs, refs, preds):
                f.write(json.dumps({"src": s, "gold": g, "pred": p}, ensure_ascii=False) + "\n")
        print(f"Per-record predictions -> {args.out}")


if __name__ == "__main__":
    main()
