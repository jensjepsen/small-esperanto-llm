"""Sample-translate EN-only Wikipedia candidates via v6 for quality inspection.

Pulls N random rows from the candidates JSONL, translates the first
~600 chars of each via the local v6 MarianMT, and prints side-by-side
for human review. Skips full-article translation — we just need a
quality signal before committing GPU time to the full 50k.

Usage:
    uv run python scripts/sample_translate_wiki_gaps.py \\
        --n 50 --max-chars 600
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from transformers import MarianMTModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mt" / "scripts"))
from sp_tokenizer import SPMTokenizer

# Sentence splitter: break on .!? followed by whitespace + uppercase letter.
# Won't be perfect on abbreviations (Mr., U.S., etc.) but good enough for
# Wikipedia body text — pretrain-quality, not human-publishing.
_SENT_BREAK = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')

def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_BREAK.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_pairs(sentences: list[str]) -> list[str]:
    """Pair adjacent sentences. Odd remainder → standalone last chunk."""
    return [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]

DEFAULT_CKPT = "/mnt/data2/checkpoints/mt/eneo_v6/final"
DEFAULT_SPM = "mt/data/tokenizer/spm_eneo_32k.model"
DEFAULT_INPUT = "/mnt/data2/wiki_gaps/en_only_candidates.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="v6 MT checkpoint dir")
    ap.add_argument("--tokenizer", default=DEFAULT_SPM,
                    help="path to spm_eneo_32k.model (uses project's SPMTokenizer "
                         "with <eo> direction prefix)")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--n", type=int, default=50,
                    help="number of random candidates to translate")
    ap.add_argument("--max-chars", type=int, default=600,
                    help="truncate EN text to this many chars before translating")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("runs/sft_evals/wiki_gaps_sample_translate.log"))
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="how many sentence-pairs to translate in parallel")
    args = ap.parse_args()

    print(f"loading tokenizer: {args.tokenizer}", flush=True)
    tok = SPMTokenizer(args.tokenizer)

    print(f"loading model: {args.ckpt}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MarianMTModel.from_pretrained(args.ckpt).to(device).eval()

    print(f"reading {args.input}, sampling {args.n} rows...", flush=True)
    rng = random.Random(args.seed)
    rows = []
    with open(args.input) as f:
        for line in f:
            rows.append(json.loads(line))
    sample = rng.sample(rows, min(args.n, len(rows)))
    print(f"  total candidates: {len(rows):,}; sampled: {len(sample)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.out.open("w")

    def emit(line):
        print(line, flush=True)
        out_f.write(line + "\n")

    emit(f"# v6 sample translations on {len(sample)} EN-only Wikipedia articles")
    emit(f"# checkpoint: {args.ckpt}")
    emit(f"# max-chars: {args.max_chars}  beams: {args.num_beams}  seed: {args.seed}")
    emit("")

    def translate_batch(en_chunks: list[str]) -> list[str]:
        """Translate a batch of EN chunks → EO. Pads to longest in batch."""
        ids_lists = [tok.encode(c, lang="eo", add_eos=True)[:256] for c in en_chunks]
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
                                 max_length=256,
                                 early_stopping=True,
                                 no_repeat_ngram_size=5,
                                 repetition_penalty=1.2,
                                 encoder_repetition_penalty=1.1)
        return [tok.decode(seq) for seq in out]

    # Flatten: build one queue of (article_idx, en_chunk) across ALL articles
    # so the GPU sees real batches even when individual articles are short.
    queue: list[tuple[int, str]] = []
    article_chunks_count: list[int] = []  # n chunks per article
    article_sents_count: list[int] = []
    truncated_text: list[str] = []
    for row in sample:
        en_text = row["text"][: args.max_chars]
        truncated_text.append(en_text)
        sents = split_sentences(en_text)
        chunks = chunk_pairs(sents)
        article_sents_count.append(len(sents))
        article_chunks_count.append(len(chunks))
        for c in chunks:
            queue.append((len(article_chunks_count) - 1, c))

    # Translate the queue in real batches of N pairs (across articles)
    eo_outputs: list[list[str]] = [[] for _ in sample]
    for b in range(0, len(queue), args.batch_size):
        batch = queue[b:b + args.batch_size]
        en_chunks = [c for _, c in batch]
        eo_chunks = translate_batch(en_chunks)
        for (art_idx, _), eo in zip(batch, eo_chunks):
            eo_outputs[art_idx].append(eo)

    for i, row in enumerate(sample, 1):
        eo_text = " ".join(eo_outputs[i - 1])
        emit(f"--- [{i:2d}/{len(sample)}] {row['title']}  "
             f"(views={row.get('views', 0):,}; len={row['length']:,} chars; "
             f"{article_sents_count[i-1]} sents → {article_chunks_count[i-1]} pairs) ---")
        emit(f"EN: {truncated_text[i-1]}")
        emit(f"EO: {eo_text}")
        emit("")

    out_f.close()
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
