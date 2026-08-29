"""Translate full articles from EN-only candidates JSONL via v6.

Production version of sample_translate_wiki_gaps.py — translates ALL rows
(no sampling), translates FULL text (no truncation), outputs JSONL ready
for v10 pretrain mixing.

Uses cross-article sentence-pair batching so the GPU sees real batches
even when individual articles have few sentences.

Output row format:
    {page_id, title, views, en_length, eo_length, eo_text}

Usage:
    uv run python scripts/translate_wiki_gaps.py \\
        --input /mnt/data2/wiki_gaps/en_only_vital_direct.jsonl \\
        --out   /mnt/data2/wiki_gaps/eo_vital_translated.jsonl \\
        --num-beams 4 --batch-size 32
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import MarianMTModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mt" / "scripts"))
from sp_tokenizer import SPMTokenizer

DEFAULT_CKPT = "/mnt/data2/checkpoints/mt/eneo_v6/final"
DEFAULT_SPM = "mt/data/tokenizer/spm_eneo_32k.model"

_SENT_BREAK = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in _SENT_BREAK.split(text) if p.strip()]


def chunk_pairs(sentences: list[str]) -> list[str]:
    return [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_SPM)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-src-len", type=int, default=256,
                    help="cap per-pair source tokens (longer pairs get truncated)")
    ap.add_argument("--max-tgt-len", type=int, default=256)
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="auto = bf16 on Ampere+, fp32 otherwise. Override "
                         "if a card without native bf16 still gets a "
                         "memory-bandwidth win from fp16-storage emulation.")
    args = ap.parse_args()

    print(f"loading tokenizer: {args.tokenizer}", flush=True)
    tok = SPMTokenizer(args.tokenizer)
    print(f"loading model: {args.ckpt}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        # auto: bf16 only on Ampere+ (compute cap >= 8.0)
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = MarianMTModel.from_pretrained(args.ckpt, torch_dtype=dtype).to(device).eval()
    print(f"model dtype: {dtype} (compute cap: "
          f"{torch.cuda.get_device_capability() if device=='cuda' else 'cpu'})", flush=True)

    print(f"reading {args.input}...", flush=True)
    rows = []
    with open(args.input) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"  {len(rows):,} articles to translate", flush=True)

    # Flatten: (article_idx, chunk_idx, en_chunk) across ALL articles
    queue: list[tuple[int, str]] = []
    chunks_per_article: list[int] = []
    for row in rows:
        sents = split_sentences(row["text"])
        chunks = chunk_pairs(sents)
        chunks_per_article.append(len(chunks))
        for c in chunks:
            queue.append((len(chunks_per_article) - 1, c))
    print(f"  total sentence-pairs: {len(queue):,}", flush=True)

    eo_per_article: list[list[str]] = [[] for _ in rows]

    def translate_batch(en_chunks: list[str]) -> list[str]:
        ids_lists = [tok.encode(c, lang="eo", add_eos=True)[: args.max_src_len]
                     for c in en_chunks]
        max_len = max(len(ids) for ids in ids_lists)
        pad_id = tok.pad_id
        in_ids = torch.full((len(ids_lists), max_len), pad_id,
                            dtype=torch.long, device=device)
        attn = torch.zeros_like(in_ids)
        for i, ids in enumerate(ids_lists):
            in_ids[i, :len(ids)] = torch.tensor(ids, device=device)
            attn[i, :len(ids)] = 1
        with torch.no_grad():
            out = model.generate(input_ids=in_ids, attention_mask=attn,
                                 num_beams=args.num_beams,
                                 max_length=args.max_tgt_len,
                                 early_stopping=True,
                                 no_repeat_ngram_size=5,
                                 repetition_penalty=1.2,
                                 encoder_repetition_penalty=1.1)
        return [tok.decode(seq) for seq in out]

    t0 = time.time()
    n_batches = (len(queue) + args.batch_size - 1) // args.batch_size
    for b in range(0, len(queue), args.batch_size):
        batch = queue[b : b + args.batch_size]
        en_chunks = [c for _, c in batch]
        eo_chunks = translate_batch(en_chunks)
        for (art_idx, _), eo in zip(batch, eo_chunks):
            eo_per_article[art_idx].append(eo)
        done = (b // args.batch_size) + 1
        elap = time.time() - t0
        rate = done / max(1, elap) * 60
        eta_min = (n_batches - done) / max(1, rate)
        if done % 10 == 0 or done == n_batches:
            print(f"  batch {done:>4}/{n_batches}  ({rate:.1f} batch/min, "
                  f"ETA {eta_min:.1f}min)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_emitted = 0
    total_eo = 0
    with out_path.open("w") as f:
        for row, eo_chunks in zip(rows, eo_per_article):
            eo_text = " ".join(eo_chunks)
            rec = {
                "page_id": row.get("page_id"),
                "title": row["title"],
                "views": row.get("views", 0),
                "en_length": row.get("length", len(row.get("text", ""))),
                "eo_length": len(eo_text),
                "en_text": row.get("text", ""),  # preserve source for future re-MT
                "eo_text": eo_text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_emitted += 1
            total_eo += len(eo_text)

    elap = time.time() - t0
    print(f"\nwrote {n_emitted:,} rows -> {out_path}", flush=True)
    print(f"total EO chars: {total_eo:,} (~{total_eo//4:,} tokens)", flush=True)
    print(f"wall time: {elap:.1f}s ({elap/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
