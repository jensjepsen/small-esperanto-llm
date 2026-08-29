"""Replay the EXACT original batch context for a dropped article.

Re-run the translate_wiki_gaps queue construction (same input file, same
order, same chunking). Find which batch contains the target article's
chunks. Translate that batch. Compare to dropped output.

If the SAME batch composition NOW produces clean output → the issue
isn't structural (batch composition isn't the trigger). Could be:
  - float-nondeterministic float reductions in beam search
  - one-shot GPU/driver state that doesn't repeat
  - PYTORCH_CUDNN_DETERMINISTIC flag drift between runs
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

CKPT = "/mnt/data2/checkpoints/mt/eneo_v6/final"
SPM = "mt/data/tokenizer/spm_eneo_32k.model"
INPUT = "/mnt/data2/wiki_gaps/en_only_vital_level5_direct.jsonl"

_SENT_BREAK = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')


def split_sentences(t):
    return [s.strip() for s in _SENT_BREAK.split(t.strip()) if s.strip()]


def chunk_pairs(s):
    return [" ".join(s[i:i+2]) for i in range(0, len(s), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Acid salt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-runs", type=int, default=3,
                    help="re-translate the same batch this many times to test determinism")
    args = ap.parse_args()

    # Load input in original order, build same queue
    rows = [json.loads(l) for l in open(INPUT)]
    queue = []  # (art_idx, chunk_idx, en_chunk)
    target_art_idx = None
    for i, row in enumerate(rows):
        if row["title"] == args.target:
            target_art_idx = i
        sents = split_sentences(row["text"])
        chunks = chunk_pairs(sents)
        for j, c in enumerate(chunks):
            queue.append((i, j, c))
    if target_art_idx is None:
        print(f"target {args.target!r} not in input"); sys.exit(1)

    # Find batches containing the target
    target_batches = []
    for b_start in range(0, len(queue), args.batch_size):
        batch = queue[b_start: b_start + args.batch_size]
        if any(art == target_art_idx for art, _, _ in batch):
            target_batches.append((b_start, batch))
    print(f"target {args.target!r} appears in {len(target_batches)} batches")

    print("loading model...", flush=True)
    tok = SPMTokenizer(SPM)
    device = torch.device("cuda")
    model = MarianMTModel.from_pretrained(CKPT).to(device).eval()

    def translate(en_chunks):
        ids_lists = [tok.encode(c, lang="eo", add_eos=True)[:512] for c in en_chunks]
        max_len = max(len(ids) for ids in ids_lists)
        pad_id = tok.pad_id
        in_ids = torch.full((len(ids_lists), max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros_like(in_ids)
        for i, ids in enumerate(ids_lists):
            in_ids[i, :len(ids)] = torch.tensor(ids, device=device)
            attn[i, :len(ids)] = 1
        with torch.no_grad():
            out = model.generate(input_ids=in_ids, attention_mask=attn,
                                 num_beams=4, max_length=256, early_stopping=True,
                                 no_repeat_ngram_size=5,
                                 repetition_penalty=1.2,
                                 encoder_repetition_penalty=1.1)
        return [tok.decode(seq) for seq in out]

    # For each batch with target, translate it multiple times
    for batch_idx, (b_start, batch) in enumerate(target_batches[:3]):  # cap at first 3 batches
        print(f"\n=== Batch starting at queue pos {b_start} (size {len(batch)}) ===")
        # Show batch composition
        comp = {}
        for art, _, _ in batch:
            comp[rows[art]["title"]] = comp.get(rows[art]["title"], 0) + 1
        print(f"composition (titles → #chunks in batch): {comp}")
        target_positions = [pos for pos, (art, _, _) in enumerate(batch) if art == target_art_idx]
        print(f"target occupies positions {target_positions} in batch")

        en_chunks = [c for _, _, c in batch]
        for run in range(args.num_runs):
            outs = translate(en_chunks)
            for p in target_positions:
                pasko = outs[p].count("pasko")
                tail = outs[p][:200]
                print(f"  run {run+1}  pos={p}  pasko={pasko}  out: {tail}")


if __name__ == "__main__":
    main()
