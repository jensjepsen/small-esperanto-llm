"""Train a joint en+eo SentencePiece BPE tokenizer on the parallel corpus."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import sentencepiece as spm


def iter_pairs(jsonl_path: Path):
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            yield r["en"]
            yield r["eo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, default=[
        Path("mt/data/parallel/opus100_train.jsonl"),
        Path("mt/data/parallel/opusbooks_train.jsonl"),
    ])
    ap.add_argument("--out-prefix", type=Path, default=Path("mt/data/tokenizer/spm_eneo_32k"))
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    ap.add_argument("--character-coverage", type=float, default=0.9999)
    ap.add_argument("--max-sentence-length", type=int, default=4192)
    args = ap.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
        n = 0
        for p in args.inputs:
            for sent in iter_pairs(p):
                tmp.write(sent.replace("\n", " ") + "\n")
                n += 1
        tmp_path = tmp.name
    print(f"Wrote {n} sentences to {tmp_path}")

    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=str(args.out_prefix),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        max_sentence_length=args.max_sentence_length,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<en>", "<eo>"],
        normalization_rule_name="nmt_nfkc_cf",
    )
    print(f"Saved tokenizer to {args.out_prefix}.model / .vocab")

    sp = spm.SentencePieceProcessor(model_file=f"{args.out_prefix}.model")
    samples = [
        "Hello, how are you today?",
        "Saluton, kiel vi fartas hodiaŭ?",
        "The quick brown fox jumps over the lazy dog.",
        "La rapida bruna vulpo saltas super la maldiligenta hundo.",
    ]
    for s in samples:
        ids = sp.encode(s)
        print(f"  {len(ids):3d} tokens  '{s}'")


if __name__ == "__main__":
    main()
