"""Materialize the tokenized v9 pretrain dataset and push to HF Hub.

Re-runs load_combined_dataset + the tokenize step from tokenize_and_chunk.
Hits the cache in $HF_HOME if you have it from a prior run (instant);
otherwise re-tokenizes (slow). Pushes the result as a private HF
dataset so any future pod can `load_dataset(<repo>)` and skip the
~50-min morpheme-tokenize phase.

Chunking is deliberately NOT done here — chunking is fast (~10 min on
any pod) and re-doing it after pulling is trivial. Tokenization is the
expensive step worth caching across pods.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from esperanto_lm.data import (
    load_combined_dataset,
    load_tokenizer,
    num_proc,
    tokenize_dataset,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="HF Hub repo ID, e.g. jensjepsen/esperanto-pretrain-tokenized-v9")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--no-private", action="store_false", dest="private")
    ap.add_argument("--tokenizer", default="tokenizer_morpheme")
    ap.add_argument("--min-article-length", type=int, default=500)
    args = ap.parse_args()

    print(f"loading tokenizer from {args.tokenizer}...", flush=True)
    tokenizer = load_tokenizer(Path(args.tokenizer))

    print("loading combined source dataset (all defaults on)...", flush=True)
    dataset = load_combined_dataset(
        use_wiki=True, use_hplt=True, use_gutenberg=True, use_mc4=False,
        use_factoids=True, use_sentences=True, use_tekstaro=True,
        use_liberafolio=True, use_fineweb=True,
        min_article_length=args.min_article_length,
    )
    print(f"  train: {len(dataset['train']):,}  test: {len(dataset['test']):,}", flush=True)

    print(f"tokenizing (num_proc={num_proc()}, hits HF_HOME cache if prior run completed)...", flush=True)
    train_tok = tokenize_dataset(dataset["train"], tokenizer)
    test_tok = tokenize_dataset(dataset["test"], tokenizer)
    print(f"  tokenized train rows: {len(train_tok):,}  test: {len(test_tok):,}", flush=True)

    from datasets import DatasetDict
    out = DatasetDict({"train": train_tok, "test": test_tok})

    print(f"pushing to {args.repo} (private={args.private})...", flush=True)
    out.push_to_hub(args.repo, private=args.private)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
