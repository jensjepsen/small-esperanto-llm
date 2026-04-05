"""Train a ByteLevel BPE tokenizer with morpheme-aware pre-tokenization.

Words are decomposed into morphemes before BPE training. Known morphemes
(roots, affixes, endings) become single tokens. Unknown words get
byte-level subword encoding as fallback.
"""

import argparse
import re
from pathlib import Path
from typing import Iterator

from rich.console import Console
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast

from esperanto_lm.data import download_dataset
from esperanto_lm.morphology import (
    decompose, get_roots, get_prefixes, get_suffixes,
    ENDINGS, DO_NOT_DECOMPOSE,
)

console = Console()

SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>"]

# Separator between morphemes within a word — must not appear in normal text.
# BPE will treat this as a token boundary.
MORPH_SEP = " ◈ "


def morpheme_corpus_iterator(dataset, max_docs: int = 0) -> Iterator[str]:
    """Yield text with morpheme boundaries marked for BPE training."""
    for i, example in enumerate(dataset):
        if max_docs and i >= max_docs:
            break
        text = example["text"]
        words = text.split()
        result = []
        for word in words:
            # Separate punctuation
            clean = word.strip(".,;:!?\"'()[]{}–—-")
            prefix_punct = word[:len(word) - len(word.lstrip(".,;:!?\"'()[]{}–—-"))]
            suffix_punct = word[len(clean) + len(prefix_punct):]

            if clean and clean[0].isalpha():
                morphemes = decompose(clean)
                result.append(prefix_punct + MORPH_SEP.join(morphemes) + suffix_punct)
            else:
                result.append(word)
        yield " ".join(result)


def main():
    parser = argparse.ArgumentParser(description="Train morpheme-aware BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Max documents to use (0 = all)")
    parser.add_argument("--output", type=Path, default=Path("tokenizer_morpheme"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Loading dataset...")
    ds = download_dataset()

    # Build initial vocabulary from all known morphemes
    console.print("[bold green]Building morpheme vocabulary...")
    all_morphemes = set()
    all_morphemes.update(get_roots())
    all_morphemes.update(get_prefixes())
    all_morphemes.update(get_suffixes())
    all_morphemes.update(ENDINGS)
    all_morphemes.update(DO_NOT_DECOMPOSE)
    # Remove empty strings
    all_morphemes.discard("")

    # Create initial vocab: special tokens + all morphemes + byte fallbacks
    initial_vocab = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        initial_vocab[tok] = i

    offset = len(SPECIAL_TOKENS)
    # Add byte-level fallback tokens (0x00-0xFF)
    for byte_val in range(256):
        byte_tok = f"<0x{byte_val:02X}>"
        initial_vocab[byte_tok] = offset + byte_val

    offset += 256
    # Add all known morphemes
    for morpheme in sorted(all_morphemes):
        if morpheme not in initial_vocab:
            initial_vocab[morpheme] = offset
            offset += 1

    console.print(f"[bold]Pre-seeded vocab:[/] {len(initial_vocab):,} tokens "
                  f"({len(all_morphemes):,} morphemes + {len(SPECIAL_TOKENS)} special + 256 bytes)")

    # Create BPE tokenizer with pre-seeded vocab
    tokenizer = Tokenizer(models.BPE(
        vocab=initial_vocab,
        merges=[],
        unk_token="<unk>",
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=3,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=list(initial_vocab.keys()),
        show_progress=True,
    )

    console.print(f"[bold green]Training BPE (target vocab_size={args.vocab_size})...")
    tokenizer.train_from_iterator(
        morpheme_corpus_iterator(ds["train"], max_docs=args.max_docs),
        trainer=trainer,
    )

    tokenizer.decoder = decoders.BPEDecoder()

    # Save raw tokenizer
    tokenizer.save(str(args.output / "tokenizer.json"))

    # Wrap as PreTrainedTokenizerFast
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    wrapped.save_pretrained(str(args.output))

    console.print(f"[bold]Vocab size:[/] {wrapped.vocab_size}")
    console.print(f"[bold green]Saved to {args.output}")

    # Demo — compare tokenization
    console.print("\n[bold]Demo:")
    test_texts = [
        "Esperanto estas internacia lingvo.",
        "La kato sidis sur la tablo.",
        "Malbonfarintoj fuĝis el la malliberejo.",
        "Ĉu vi parolas Esperanton?",
        "La ĉefurbo de Germanio estas Berlino.",
        "gepatroj malfeliĉuloj nekompreneble",
        "malbonfarintoj",
        "senato",
    ]

    for text in test_texts:
        # Pre-process: decompose into morphemes
        words = text.split()
        preprocessed_parts = []
        for word in words:
            clean = word.strip(".,;:!?\"'()[]{}–—-")
            suffix_punct = word[len(clean):]
            if clean and clean[0].isalpha():
                morphemes = decompose(clean)
                preprocessed_parts.append(MORPH_SEP.join(morphemes) + suffix_punct)
            else:
                preprocessed_parts.append(word)
        preprocessed = " ".join(preprocessed_parts)

        ids = wrapped.encode(preprocessed)
        tokens = [wrapped.decode([t]).strip() for t in ids]
        # Filter out separator tokens
        tokens = [t for t in tokens if t and t != "◈"]

        console.print(f"  [dim]{text}[/]")
        console.print(f"  morphemes: {' · '.join(sum([decompose(w.strip('.,;:!?')) for w in words if w.strip('.,;:!?')], []))}")
        console.print(f"  tokens ({len(tokens)}): {tokens}")
        console.print()


if __name__ == "__main__":
    main()
