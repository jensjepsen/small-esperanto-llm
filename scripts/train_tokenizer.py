"""Train a tokenizer on the Esperanto corpus.

By default trains a morpheme-aware BPE tokenizer. Use --legacy for the
old ByteLevel BPE without morpheme pre-tokenization.
"""

import argparse
from pathlib import Path

from rich.console import Console

from esperanto_lm.data import download_dataset, load_combined_dataset
from esperanto_lm.data import train_tokenizer as train_legacy_tokenizer

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
    parser.add_argument("--use-hplt", action="store_true")
    parser.add_argument("--use-gutenberg", action="store_true")
    parser.add_argument("--use-mc4", action="store_true")
    parser.add_argument("--use-factoids", action="store_true")
    parser.add_argument("--use-sentences", action="store_true")
    args = parser.parse_args()

    console.print("[bold green]Loading dataset...")
    any_extra = args.use_hplt or args.use_gutenberg or args.use_mc4 or args.use_factoids or args.use_sentences
    if any_extra:
        dataset = load_combined_dataset(
            use_hplt=args.use_hplt, use_gutenberg=args.use_gutenberg,
            use_mc4=args.use_mc4, use_factoids=args.use_factoids,
            use_sentences=args.use_sentences,
        )
    else:
        dataset = download_dataset()
    console.print(f"[bold]Train examples:[/] {len(dataset['train']):,}")

    if args.legacy:
        output = args.output or Path("tokenizer")
        console.print(f"[bold green]Training legacy BPE (vocab_size={args.vocab_size})...")
        tokenizer = train_legacy_tokenizer(dataset["train"], save_dir=output,
                                           vocab_size=args.vocab_size)
        console.print(f"[bold]Vocab size:[/] {tokenizer.vocab_size}")
        console.print(f"[bold green]Done! Saved to {output}")
    else:
        output = args.output or Path("tokenizer_morpheme")
        from scripts.train_morpheme_tokenizer import train_morpheme_bpe

        output.mkdir(parents=True, exist_ok=True)

        train_morpheme_bpe(
            dataset=dataset,
            vocab_size=args.vocab_size,
            max_docs=args.max_docs,
            output=output,
        )


if __name__ == "__main__":
    main()
