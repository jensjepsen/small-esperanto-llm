"""Train and query a word2vec model on Esperanto tokens."""

import argparse
import logging
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from rich.console import Console
from rich.progress import Progress

from esperanto_lm.data import load_combined_dataset, load_tokenizer
from esperanto_lm.morphology import classify_morpheme, decompose, decompose_text

console = Console()

DEFAULT_MODEL_PATH = Path("models/word2vec.model")


def token_sentences(dataset, tokenizer, max_docs: int = 0,
                    whole_words: bool = False, morphemes: bool = False):
    """Yield lists of tokens from the dataset."""
    for i, example in enumerate(dataset):
        if max_docs and i >= max_docs:
            break
        if morphemes:
            tokens = decompose_text(example["text"])
        elif whole_words:
            tokens = example["text"].lower().split()
        else:
            ids = tokenizer.encode(example["text"])
            tokens = [tokenizer.decode([t]) for t in ids]
        yield tokens


# --- Vector helpers ---

def make_word_vector_fn(model, args, tok=None):
    """Create a word_vector function based on the mode."""
    def word_vector(word: str):
        if getattr(args, 'whole_words', False):
            w = word.lower()
            return model.wv[w] if w in model.wv else None
        if getattr(args, 'morphemes', False):
            morphs = decompose(word)
            vecs = [model.wv[m] for m in morphs if m in model.wv]
            if not vecs:
                return None
            return np.mean(vecs, axis=0)
        if tok:
            ids = tok.encode(word, add_special_tokens=False)
            tokens = [tok.decode([t]) for t in ids]
            vecs = [model.wv[t] for t in tokens if t in model.wv]
            if not vecs:
                return None
            return np.mean(vecs, axis=0)
        # Fallback: direct lookup
        w = word.lower()
        return model.wv[w] if w in model.wv else None
    return word_vector


def similar_words(model, word_vector, word: str, topn: int = 5):
    vec = word_vector(word)
    if vec is None:
        return None
    return model.wv.similar_by_vector(vec, topn=topn + 5)


def analogy(word_vector, pos1: str, neg: str, pos2: str, model=None, topn: int = 5):
    v1 = word_vector(pos1)
    vn = word_vector(neg)
    v2 = word_vector(pos2)
    if any(v is None for v in (v1, vn, v2)):
        return None
    vec = v1 - vn + v2
    return model.wv.similar_by_vector(vec, topn=topn + 5)


def decode_results(results, exclude_short=True, min_len=3):
    if results is None:
        return "not computable"
    filtered = []
    for w, s in results:
        if exclude_short and len(w.strip()) <= 2:
            continue
        filtered.append(f"{w.strip()} ({s:.2f})")
        if len(filtered) >= 5:
            break
    return ", ".join(filtered)


def similar_by_type(model, word_vector, word: str, morph_type: str, topn: int = 5):
    vec = word_vector(word)
    if vec is None:
        return "not computable"
    results = model.wv.similar_by_vector(vec, topn=100)
    filtered = []
    for w, s in results:
        if classify_morpheme(w) == morph_type and w != word.lower():
            filtered.append(f"{w} ({s:.2f})")
            if len(filtered) >= topn:
                break
    return ", ".join(filtered) if filtered else "none found"


def print_analogy_result(model, word_vector, pos1, neg, pos2, expected, morphemes=False):
    results = analogy(word_vector, pos1, neg, pos2, model=model)
    console.print(f"  {pos1} - {neg} + {pos2} = {decode_results(results)}  (expected: {expected})")
    if morphemes and results:
        by_type: dict[str, list[str]] = {}
        for w, s in results:
            mt = classify_morpheme(w.strip())
            if mt not in by_type:
                by_type[mt] = []
            if len(by_type[mt]) < 3:
                by_type[mt].append(f"{w.strip()} ({s:.2f})")
        for mt in ["root", "prefix", "suffix", "ending", "particle"]:
            if mt in by_type:
                console.print(f"    {mt:>8}: {', '.join(by_type[mt])}")


def run_demo(model, word_vector, morphemes=False):
    """Run the standard demo suite."""
    console.print("\n[bold]Similar words:")
    for word in ["Esperanto", "urbo", "lando", "libro", "kato", "muziko",
                 "rivero", "Parizo", "reĝino", "granda"]:
        results = similar_words(model, word_vector, word)
        console.print(f"  {word}: {decode_results(results)}")

    if morphemes:
        console.print("\n[bold]Closest roots:")
        for word in ["lern", "patr", "kant", "manĝ", "labor", "skrib"]:
            console.print(f"  {word}: {similar_by_type(model, word_vector, word, 'root')}")

        console.print("\n[bold]Closest prefixes to 'mal':")
        console.print(f"  {similar_by_type(model, word_vector, 'mal', 'prefix')}")

        console.print("\n[bold]Closest suffixes to 'ist':")
        console.print(f"  {similar_by_type(model, word_vector, 'ist', 'suffix')}")

        console.print("\n[bold]Closest suffixes to 'ej':")
        console.print(f"  {similar_by_type(model, word_vector, 'ej', 'suffix')}")

        console.print("\n[bold]Closest endings:")
        for ending in ["o", "a", "e", "i", "as", "is", "os"]:
            console.print(f"  {ending}: {similar_by_type(model, word_vector, ending, 'ending')}")

    console.print("\n[bold]Analogies:")
    tests = [
        ("reĝino", "reĝo", "viro", "virino"),
        ("patrino", "patro", "filo", "filino"),
        ("malbona", "bona", "bela", "malbela"),
        ("mallonga", "longa", "granda", "malgranda"),
        ("lernejo", "lerni", "manĝi", "manĝejo"),
        ("Parizo", "Francio", "Germanio", "Berlino"),
        ("kantisto", "kanti", "pentri", "pentristo"),
        ("hundido", "hundo", "kato", "katido"),
    ]
    for pos1, neg, pos2, expected in tests:
        print_analogy_result(model, word_vector, pos1, neg, pos2, expected, morphemes)


def cmd_train(args):
    """Train a new word2vec model."""
    args.output.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Loading tokenizer and dataset...")
    tok = load_tokenizer()
    ds = load_combined_dataset(
        use_hplt=args.use_hplt,
        use_gutenberg=args.use_gutenberg,
        use_mc4=args.use_mc4,
        use_factoids=args.use_factoids,
        use_sentences=args.use_sentences,
    )
    console.print(f"[bold]Train examples:[/] {len(ds['train']):,}")

    console.print("[bold green]Tokenizing documents...")
    subset = ds["train"]
    if args.max_docs:
        subset = subset.select(range(min(args.max_docs, len(subset))))

    if args.morphemes:
        # Use dataset.map for parallel morpheme decomposition
        def morpheme_tokenize(examples):
            return {"tokens": [decompose_text(t) for t in examples["text"]]}
        tokenized = subset.map(morpheme_tokenize, batched=True, num_proc=4,
                               remove_columns=subset.column_names)
        sentences = tokenized["tokens"]
    elif args.whole_words:
        def word_tokenize(examples):
            return {"tokens": [t.lower().split() for t in examples["text"]]}
        tokenized = subset.map(word_tokenize, batched=True, num_proc=4,
                               remove_columns=subset.column_names)
        sentences = tokenized["tokens"]
    else:
        sentences = []
        with Progress() as progress:
            task = progress.add_task("Tokenizing...", total=len(subset))
            for s in token_sentences(subset, tok):
                sentences.append(s)
                progress.advance(task)

    console.print(f"[bold]Documents tokenized:[/] {len(sentences):,}")

    console.print("[bold green]Training word2vec...")
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    model = Word2Vec(
        sentences=sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=4,
        epochs=args.epochs,
    )
    console.print(f"[bold]Vocab size:[/] {len(model.wv):,}")

    model.save(str(args.output))
    console.print(f"[bold green]Saved to {args.output}")

    word_vector = make_word_vector_fn(model, args, tok)
    run_demo(model, word_vector, morphemes=args.morphemes)


def cmd_query(args):
    """Interactive query mode for a trained model."""
    console.print(f"[bold green]Loading model from {args.model}...")
    model = Word2Vec.load(str(args.model))
    console.print(f"[bold]Vocab size:[/] {len(model.wv):,}")

    word_vector = make_word_vector_fn(model, args)

    if args.demo:
        run_demo(model, word_vector, morphemes=args.morphemes)
        return

    console.print("\n[bold]Interactive mode. Commands:")
    console.print("  similar <word>              — find similar words")
    console.print("  roots <word>                — closest roots (morpheme mode)")
    console.print("  analogy <a> <b> <c>         — solve a - b + c = ?")
    console.print("  decompose <word>            — show morpheme breakdown")
    console.print("  quit                        — exit")
    console.print()

    while True:
        try:
            line = input("w2v> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "quit" or cmd == "exit":
            break

        elif cmd == "similar" and len(parts) >= 2:
            word = parts[1]
            results = similar_words(model, word_vector, word, topn=10)
            console.print(decode_results(results, exclude_short=False, min_len=1))
            if args.morphemes:
                for mtype in ["root", "prefix", "suffix", "ending"]:
                    console.print(f"  {mtype}: {similar_by_type(model, word_vector, word, mtype)}")

        elif cmd == "roots" and len(parts) >= 2:
            word = parts[1]
            console.print(similar_by_type(model, word_vector, word, "root"))

        elif cmd == "analogy" and len(parts) >= 4:
            pos1, neg, pos2 = parts[1], parts[2], parts[3]
            expected = parts[4] if len(parts) >= 5 else "?"
            print_analogy_result(model, word_vector, pos1, neg, pos2, expected, args.morphemes)

        elif cmd == "decompose" and len(parts) >= 2:
            word = parts[1]
            tagged = [(m, classify_morpheme(m)) for m in decompose(word)]
            console.print(" · ".join(f"{m}[{t}]" for m, t in tagged))

        else:
            console.print("[red]Unknown command. Try: similar, roots, analogy, decompose, quit")


def main():
    parser = argparse.ArgumentParser(description="Word2vec for Esperanto")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--max-docs", type=int, default=0)
    train_parser.add_argument("--vector-size", type=int, default=100)
    train_parser.add_argument("--window", type=int, default=5)
    train_parser.add_argument("--min-count", type=int, default=5)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--whole-words", action="store_true")
    train_parser.add_argument("--morphemes", action="store_true")
    train_parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--use-hplt", action="store_true")
    train_parser.add_argument("--use-gutenberg", action="store_true")
    train_parser.add_argument("--use-mc4", action="store_true")
    train_parser.add_argument("--use-factoids", action="store_true")
    train_parser.add_argument("--use-sentences", action="store_true")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query a trained model")
    query_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    query_parser.add_argument("--morphemes", action="store_true")
    query_parser.add_argument("--whole-words", action="store_true")
    query_parser.add_argument("--demo", action="store_true",
                              help="Run the demo suite instead of interactive mode")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
