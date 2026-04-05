"""Train a tokenizer on the Esperanto corpus.

By default trains a morpheme-aware BPE tokenizer. Use --legacy for the
old ByteLevel BPE without morpheme pre-tokenization.
"""

import argparse
from pathlib import Path

from rich.console import Console

from esperanto_lm.data import download_dataset, train_tokenizer as train_legacy_tokenizer

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train an Esperanto tokenizer")
    parser.add_argument("--legacy", action="store_true",
                        help="Train legacy ByteLevel BPE without morpheme awareness")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Max documents to use (0 = all)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: tokenizer_morpheme or tokenizer)")
    args = parser.parse_args()

    console.print("[bold green]Loading dataset...")
    dataset = download_dataset()

    if args.legacy:
        output = args.output or Path("tokenizer")
        console.print(f"[bold green]Training legacy BPE (vocab_size={args.vocab_size})...")
        tokenizer = train_legacy_tokenizer(dataset["train"], save_dir=output,
                                           vocab_size=args.vocab_size)
        console.print(f"[bold]Vocab size:[/] {tokenizer.vocab_size}")
        console.print(f"[bold green]Done! Saved to {output}")
    else:
        output = args.output or Path("tokenizer_morpheme")
        # Import here to avoid circular deps
        from scripts.train_morpheme_tokenizer import morpheme_corpus_iterator, SPECIAL_TOKENS
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        from transformers import PreTrainedTokenizerFast
        from esperanto_lm.morphology import get_roots, get_prefixes, get_suffixes, ENDINGS, DO_NOT_DECOMPOSE

        output.mkdir(parents=True, exist_ok=True)

        console.print("[bold green]Building morpheme vocabulary...")
        all_morphemes = set()
        all_morphemes.update(get_roots())
        all_morphemes.update(get_prefixes())
        all_morphemes.update(get_suffixes())
        all_morphemes.update(ENDINGS)
        all_morphemes.update(DO_NOT_DECOMPOSE)
        all_morphemes.discard("")

        initial_vocab = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            initial_vocab[tok] = i

        offset = len(SPECIAL_TOKENS)
        for byte_val in range(256):
            byte_tok = f"<0x{byte_val:02X}>"
            initial_vocab[byte_tok] = offset + byte_val

        offset += 256
        for morpheme in sorted(all_morphemes):
            if morpheme not in initial_vocab:
                initial_vocab[morpheme] = offset
                offset += 1

        console.print(f"[bold]Pre-seeded vocab:[/] {len(initial_vocab):,} tokens")

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

        console.print(f"[bold green]Training morpheme BPE (vocab_size={args.vocab_size})...")
        tokenizer.train_from_iterator(
            morpheme_corpus_iterator(dataset["train"], max_docs=args.max_docs),
            trainer=trainer,
        )
        tokenizer.decoder = decoders.BPEDecoder()
        tokenizer.save(str(output / "tokenizer.json"))

        wrapped = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )
        wrapped.save_pretrained(str(output))

        console.print(f"[bold]Vocab size:[/] {wrapped.vocab_size}")
        console.print(f"[bold green]Done! Saved to {output}")


if __name__ == "__main__":
    main()
