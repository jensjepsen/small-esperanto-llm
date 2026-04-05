"""Train a morpheme-aware BPE tokenizer for Esperanto.

Text is pre-decomposed into morphemes. BPE is trained on individual
morphemes as "words", so merges only happen within morphemes.
Known morphemes become single tokens; unknown stems get subword splits.
"""

import argparse
import re
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.progress import Progress
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

from esperanto_lm.data import download_dataset, load_combined_dataset
from esperanto_lm.morphology import decompose

console = Console()

SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>", "<w>"]


def morpheme_iterator(dataset, max_docs: int = 0) -> Iterator[str]:
    """Yield text where each morpheme is a separate whitespace-delimited word.

    <w> tokens are inserted between words to mark word boundaries.
    """
    for i, example in enumerate(dataset):
        if max_docs and i >= max_docs:
            break
        text = example["text"]
        words = re.findall(r'[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+|[^\s]', text)
        parts = []
        for word in words:
            if parts:
                parts.append("<w>")
            if word[0].isalpha():
                parts.extend(decompose(word))
            else:
                parts.append(word)
        yield " ".join(parts)


EO_INITIAL_ALPHABET = list(
    "abcĉdefgĝhĥijĵklmnoprsŝtuŭvz"
    "ABCĈDEFGĜHĤIJĴKLMNOPRSŜTUŬVZ"
    "0123456789"
    ".,;:!?-()[]{}\"'/\\@#$%&*+=<>~`^_|"
    " \t\n"
)


def _build_morpheme_special_tokens() -> list[str]:
    """Build list of all known morphemes to protect as special tokens."""
    from esperanto_lm.morphology import get_roots, get_prefixes, get_suffixes, ENDINGS, DO_NOT_DECOMPOSE

    morphemes = set()
    morphemes.update(get_roots())
    morphemes.update(get_prefixes())
    morphemes.update(get_suffixes())
    morphemes.update(ENDINGS)
    morphemes.update(DO_NOT_DECOMPOSE)
    morphemes.discard("")

    # Sort by length descending so longer morphemes get priority
    return SPECIAL_TOKENS + sorted(morphemes, key=len, reverse=True)


def train_morpheme_bpe(dataset, vocab_size: int = 8000, max_docs: int = 0,
                       output: Path = Path("tokenizer_morpheme")):
    """Train a morpheme-aware BPE tokenizer and save it."""
    output.mkdir(parents=True, exist_ok=True)

    all_special = _build_morpheme_special_tokens()
    console.print(f"[bold]Protected morpheme tokens:[/] {len(all_special):,}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=max(vocab_size, len(all_special) + 256),
        min_frequency=3,
        special_tokens=all_special,
        initial_alphabet=EO_INITIAL_ALPHABET,
        show_progress=True,
    )

    console.print(f"[bold green]Training BPE on morphemes (vocab_size={vocab_size})...")
    tokenizer.train_from_iterator(
        morpheme_iterator(dataset["train"], max_docs=max_docs),
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
    console.print(f"[bold green]Saved to {output}")
    return wrapped


def main():
    parser = argparse.ArgumentParser(description="Train morpheme-aware BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("tokenizer_morpheme"))
    parser.add_argument("--use-hplt", action="store_true")
    parser.add_argument("--use-gutenberg", action="store_true")
    parser.add_argument("--use-mc4", action="store_true")
    parser.add_argument("--use-factoids", action="store_true")
    parser.add_argument("--use-sentences", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Loading dataset...")
    any_extra = args.use_hplt or args.use_gutenberg or args.use_mc4 or args.use_factoids or args.use_sentences
    if any_extra:
        ds = load_combined_dataset(
            use_hplt=args.use_hplt, use_gutenberg=args.use_gutenberg,
            use_mc4=args.use_mc4, use_factoids=args.use_factoids,
            use_sentences=args.use_sentences,
        )
    else:
        ds = download_dataset()
    console.print(f"[bold]Train examples:[/] {len(ds['train']):,}")

    train_morpheme_bpe(
        dataset=ds,
        vocab_size=args.vocab_size,
        max_docs=args.max_docs,
        output=args.output,
    )

    # Demo
    wrapped = PreTrainedTokenizerFast.from_pretrained(str(args.output))
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
        "John Cleese estas aktoro.",
        "COVID-19 estas malsano.",
    ]

    for text in test_texts:
        words = re.findall(r'[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+|[^\s]', text)
        morphemes = []
        for word in words:
            if word[0].isalpha():
                morphemes.extend(decompose(word))
            else:
                morphemes.append(word)

        morpheme_text = " ".join(morphemes)
        ids = wrapped.encode(morpheme_text)
        tokens = wrapped.convert_ids_to_tokens(ids)

        console.print(f"  [dim]{text}[/]")
        console.print(f"  morphemes: {' · '.join(morphemes)}")
        console.print(f"  tokens ({len(tokens)}): {tokens}")
        console.print()


if __name__ == "__main__":
    main()
