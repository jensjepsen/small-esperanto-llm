"""Download tokenizer, factoids, and sentences from HF Hub."""

import argparse
from pathlib import Path

from rich.console import Console

console = Console()

HF_TOKENIZER = "jensjepsen/esperanto-morpheme-tokenizer"
HF_FACTOIDS = "jensjepsen/esperanto-factoids"
HF_SENTENCES = "jensjepsen/esperanto-sentences"


def main():
    parser = argparse.ArgumentParser(description="Download data from HF Hub")
    parser.add_argument("--tokenizer", action="store_true")
    parser.add_argument("--factoids", action="store_true")
    parser.add_argument("--sentences", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not (args.tokenizer or args.factoids or args.sentences or args.all):
        parser.print_help()
        return

    if args.tokenizer or args.all:
        output = Path("tokenizer_morpheme")
        if output.exists():
            console.print(f"[dim]Tokenizer already exists at {output}, skipping")
        else:
            console.print(f"[bold green]Downloading tokenizer from {HF_TOKENIZER}...")
            from huggingface_hub import snapshot_download
            snapshot_download(HF_TOKENIZER, local_dir=str(output))
            console.print(f"[bold]Done! Saved to {output}")

    if args.factoids or args.all:
        console.print(f"[bold green]Downloading factoids from {HF_FACTOIDS}...")
        from datasets import load_dataset
        ds = load_dataset(HF_FACTOIDS, split="train")
        console.print(f"[bold]Loaded {len(ds):,} factoid paragraphs")
        output = Path("data/factoids")
        output.mkdir(parents=True, exist_ok=True)
        output_path = output / "factoid_text.jsonl"
        import json
        with open(output_path, "w") as f:
            for example in ds:
                f.write(json.dumps({"text": example["text"]}, ensure_ascii=False) + "\n")
        console.print(f"[bold]Done! Saved to {output_path}")

    if args.sentences or args.all:
        console.print(f"[bold green]Downloading sentences from {HF_SENTENCES}...")
        from datasets import load_dataset
        ds = load_dataset(HF_SENTENCES, split="train")
        console.print(f"[bold]Loaded {len(ds):,} sentences")
        output = Path("data")
        output.mkdir(parents=True, exist_ok=True)
        output_path = output / "epo_sentences.tsv"
        with open(output_path, "w") as f:
            for i, example in enumerate(ds):
                f.write(f"{i}\teo\t{example['text']}\n")
        console.print(f"[bold]Done! Saved to {output_path}")


if __name__ == "__main__":
    main()
