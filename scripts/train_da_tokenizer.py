"""Train a Danish byte-level BPE tokenizer from the pretrain corpus.

Uses HuggingFace `tokenizers` library with:
  - Model: BPE
  - Pre-tokenizer: ByteLevel (GPT-2 / Llama-3 style — no unicode OOV)
  - Decoder: ByteLevel
  - Trainer: BpeTrainer, 16k vocab, standard special tokens

Sampling strategy — cap contribution per source (in chars) so that FineWeb-2
(91M docs) doesn't dominate vocab merges over curated content:

  wikipedia          all              (~800 MB text)
  gutenberg_delta    all              (~8 MB)
  ia_danish          all              (~900 MB)
  dynaword           first ~2 GB
  fineweb2           first ~2 GB

Total ~5-6 GB. Written to a single text file (one doc per line).

Output: /workspace/tokenizer_da/tokenizer.json (HF-native single file).
Load with AutoTokenizer.from_pretrained("path/to/tokenizer_da/").
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

WORK = Path("/workspace/work")
DEDUP_DIR = WORK / "dedup"
OUT_DIR = Path("/workspace/tokenizer_da")
SAMPLE_PATH = Path("/workspace/work/tokenizer_sample.txt")

SOURCE_CAPS = {
    "wikipedia": None,          # take all (~800 MB)
    "gutenberg_delta": None,    # take all (~8 MB)
    "ia_danish": None,          # take all (~900 MB)
    "dynaword": 2_000_000_000,  # cap at 2 GB
    "fineweb2": 2_000_000_000,  # cap at 2 GB
}


def sample_corpus() -> None:
    """Stream deduped shards, write source-balanced sample."""
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_src_chars = {s: 0 for s in SOURCE_CAPS}
    n_written = 0
    t0 = time.time()

    with SAMPLE_PATH.open("w", encoding="utf-8") as fout:
        for shard in sorted(DEDUP_DIR.glob("*.jsonl.gz")):
            fname = shard.name.removeprefix("keep_")
            src = fname.split("_shard_")[0]
            if src not in SOURCE_CAPS:
                continue
            cap = SOURCE_CAPS[src]
            if cap is not None and per_src_chars[src] >= cap:
                continue
            with gzip.open(shard, "rt", encoding="utf-8") as fin:
                for line in fin:
                    try:
                        text = json.loads(line)["text"]
                    except Exception:
                        continue
                    if len(text) > 100_000:
                        text = text[:100_000]
                    text = text.replace("\n", " ").replace("\r", " ").strip()
                    if len(text) < 100:
                        continue
                    fout.write(text + "\n")
                    per_src_chars[src] += len(text)
                    n_written += 1
                    if cap is not None and per_src_chars[src] >= cap:
                        break
            if n_written % 500_000 == 0 and n_written > 0:
                el = time.time() - t0
                print(f"  sampled {n_written:,} docs  "
                      f"chars={sum(per_src_chars.values())/1e9:.2f}GB  "
                      f"({el/60:.1f}min)", flush=True)

    print(f"[sample] done. {n_written:,} docs, "
          f"{sum(per_src_chars.values())/1e9:.2f} GB text  "
          f"in {(time.time()-t0)/60:.1f}min", flush=True)
    for src, chars in per_src_chars.items():
        print(f"  {src}: {chars/1e9:.2f} GB", flush=True)


def train_bpe(vocab_size: int) -> None:
    """Train byte-level BPE via HuggingFace tokenizers."""
    from tokenizers import Tokenizer, Regex
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre, Split, Sequence
    from tokenizers.decoders import ByteLevel as ByteLevelDec
    from tokenizers.processors import ByteLevel as ByteLevelProc

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[bpe] training byte-level BPE vocab={vocab_size} "
          f"(single-digit split)", flush=True)
    t0 = time.time()

    tok = Tokenizer(BPE(unk_token=None, byte_fallback=False))
    # Isolate each digit BEFORE ByteLevel so BPE never merges "2024" → 1 token.
    # This matches Llama-3 / Mistral / OLMo digit-splitting practice.
    tok.pre_tokenizer = Sequence([
        Split(pattern=Regex(r"\d"), behavior="isolated"),
        ByteLevelPre(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = ByteLevelDec()
    tok.post_processor = ByteLevelProc(trim_offsets=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        initial_alphabet=ByteLevelPre.alphabet(),  # all 256 byte codepoints
    )

    tok.train([str(SAMPLE_PATH)], trainer=trainer)

    out_path = OUT_DIR / "tokenizer.json"
    tok.save(str(out_path))
    print(f"[bpe] saved to {out_path}", flush=True)
    print(f"[bpe] vocab size: {tok.get_vocab_size()}", flush=True)
    print(f"[bpe] done in {(time.time()-t0)/60:.1f}min", flush=True)

    # Sanity-check on Danish samples
    print("\n[bpe] sanity check:", flush=True)
    for s in [
        "Kongen af Danmark bor i København.",
        "H.C. Andersens eventyr blev læst højt af farmor.",
        "Sundhedsforsikringspolice for udenlandske arbejdstagere.",
        "æøåÆØÅ",
        "1234567890",
    ]:
        enc = tok.encode(s)
        print(f"  {s!r}")
        print(f"    → {len(enc.tokens)} tokens: {enc.tokens}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--skip-sample", action="store_true",
                    help="reuse existing /workspace/work/tokenizer_sample.txt")
    args = ap.parse_args()

    if args.skip_sample and SAMPLE_PATH.exists():
        print(f"[sample] reusing existing {SAMPLE_PATH} "
              f"({SAMPLE_PATH.stat().st_size/1e9:.2f} GB)", flush=True)
    else:
        print(f"[sample] extracting source-balanced text sample", flush=True)
        sample_corpus()
    train_bpe(args.vocab_size)


if __name__ == "__main__":
    main()
