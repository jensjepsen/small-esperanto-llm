"""Try to reproduce the L5-translation garbling.

For a known-dropped article, translate its first ~20 sentence-pairs THREE ways:
  1. Alone (batch_size=1) — pure context
  2. Batch with other dropped-article sentences (cross-article batching, same as production)
  3. Batch with neutral filler (clean sentences from a different article)

Reports per-chunk: 'pasko' count, length, and whether the output matches across runs.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import MarianMTModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mt" / "scripts"))
from sp_tokenizer import SPMTokenizer

DEFAULT_CKPT = "/mnt/data2/checkpoints/mt/eneo_v6/final"
DEFAULT_SPM = "mt/data/tokenizer/spm_eneo_32k.model"

_SENT_BREAK = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')


def split_sentences(t: str) -> list[str]:
    return [s.strip() for s in _SENT_BREAK.split(t.strip()) if s.strip()]


def chunk_pairs(sents: list[str]) -> list[str]:
    return [" ".join(sents[i:i+2]) for i in range(0, len(sents), 2)]


def translate(model, tok, en_chunks: list[str], device, num_beams=4) -> list[str]:
    ids_lists = [tok.encode(c, lang="eo", add_eos=True)[:512] for c in en_chunks]
    max_len = max(len(ids) for ids in ids_lists)
    pad_id = tok.pad_id
    in_ids = torch.full((len(ids_lists), max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros_like(in_ids)
    for i, ids in enumerate(ids_lists):
        in_ids[i, :len(ids)] = torch.tensor(ids, device=device)
        attn[i, :len(ids)] = 1
    with torch.no_grad():
        out = model.generate(
            input_ids=in_ids, attention_mask=attn,
            num_beams=num_beams, max_length=256, early_stopping=True,
            no_repeat_ngram_size=5,
            repetition_penalty=1.2,
            encoder_repetition_penalty=1.1,
        )
    return [tok.decode(seq) for seq in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-title", default="Acid salt")
    ap.add_argument("--n-chunks", type=int, default=8)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--spm", default=DEFAULT_SPM)
    ap.add_argument("--num-beams", type=int, default=4)
    args = ap.parse_args()

    # Load source EN texts
    dropped_path = Path("/mnt/data2/wiki_gaps/eo_vital_level5_dropped.jsonl")
    clean_path = Path("/mnt/data2/wiki_gaps/eo_vital_level5_clean.jsonl")
    target = None
    other_dropped = []
    with dropped_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["title"] == args.target_title:
                target = r
            else:
                other_dropped.append(r)
    if target is None:
        print(f"target {args.target_title!r} not in dropped"); sys.exit(1)
    fillers = []
    with clean_path.open() as f:
        for i, line in enumerate(f):
            if i >= 5: break
            fillers.append(json.loads(line))

    target_chunks = chunk_pairs(split_sentences(target["en_text"]))[: args.n_chunks]
    print(f"target: {target['title']!r}  ({len(target_chunks)} chunks of ~2 sentences each)")
    print(f"target EO from dropped (head 250):", target["eo_text"][:250].replace("\n", " "))
    print()

    print("loading model...", flush=True)
    tok = SPMTokenizer(args.spm)
    device = torch.device("cuda")
    model = MarianMTModel.from_pretrained(args.ckpt).to(device).eval()

    # 1) Alone (one chunk per batch)
    print(f"\n=== ALONE (batch=1, {len(target_chunks)} sequential calls) ===")
    alone_outs = []
    for c in target_chunks:
        alone_outs.append(translate(model, tok, [c], device, args.num_beams)[0])

    # 2) Cross-article: target chunks + other dropped chunks in same batch
    other_chunks = []
    for r in other_dropped[:3]:
        other_chunks.extend(chunk_pairs(split_sentences(r["en_text"]))[:3])
    mixed_batch = target_chunks + other_chunks[: 32 - len(target_chunks)]
    print(f"\n=== CROSS-ARTICLE BATCH (batch={len(mixed_batch)}, target+others-dropped) ===")
    cross_outs = translate(model, tok, mixed_batch, device, args.num_beams)[: len(target_chunks)]

    # 3) Clean-filler batch
    filler_chunks = []
    for r in fillers:
        filler_chunks.extend(chunk_pairs(split_sentences(r["en_text"]))[:3])
    clean_batch = target_chunks + filler_chunks[: 32 - len(target_chunks)]
    print(f"\n=== CLEAN-FILLER BATCH (batch={len(clean_batch)}, target+5 clean articles) ===")
    clean_outs = translate(model, tok, clean_batch, device, args.num_beams)[: len(target_chunks)]

    # Compare
    print(f"\n{'chunk':>5s}  {'alone':>10s}  {'cross':>10s}  {'clean':>10s}  match")
    print(f"{'':>5s}  {'len/pasko':>10s}  {'len/pasko':>10s}  {'len/pasko':>10s}")
    for i, (a, c, cl) in enumerate(zip(alone_outs, cross_outs, clean_outs)):
        ap_ = a.count("pasko"); cp_ = c.count("pasko"); clp_ = cl.count("pasko")
        match = "alone≈cross" if a == c else ("alone≈clean" if a == cl else "ALL DIFFER")
        print(f"{i:>5d}  {len(a):>4d}/{ap_:<5d}  {len(c):>4d}/{cp_:<5d}  {len(cl):>4d}/{clp_:<5d}  {match}")

    # Print interesting chunks
    print(f"\n=== Per-chunk content ===")
    for i, (en, a, c, cl) in enumerate(zip(target_chunks, alone_outs, cross_outs, clean_outs)):
        any_pasko = any(o.count("pasko") for o in (a, c, cl))
        if not any_pasko:
            continue
        print(f"\n[chunk {i}] EN: {en[:200]}")
        print(f"   alone : {a[:240]}")
        print(f"   cross : {c[:240]}")
        print(f"   clean : {cl[:240]}")


if __name__ == "__main__":
    main()
