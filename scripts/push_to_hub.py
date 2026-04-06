"""Push tokenizer, factoids, and sentences to Hugging Face Hub."""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi
from rich.console import Console

console = Console()

DEFAULT_ORG = "jensjepsen"


def main():
    parser = argparse.ArgumentParser(description="Push data to Hugging Face Hub")
    parser.add_argument("--org", type=str, default=DEFAULT_ORG)
    parser.add_argument("--tokenizer", action="store_true", help="Push tokenizer")
    parser.add_argument("--factoids", action="store_true", help="Push factoids dataset")
    parser.add_argument("--sentences", action="store_true", help="Push sentences dataset")
    parser.add_argument("--all", action="store_true", help="Push everything")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("tokenizer_morpheme"))
    parser.add_argument("--factoids-path", type=Path,
                        default=Path("/mnt/data2/wikidata5m/eo_factoids/factoid_text.jsonl"))
    parser.add_argument("--sentences-path", type=Path, default=Path("data/epo_sentences.tsv"))
    args = parser.parse_args()

    if not (args.tokenizer or args.factoids or args.sentences or args.all):
        parser.print_help()
        return

    api = HfApi()

    if args.tokenizer or args.all:
        repo_id = f"{args.org}/esperanto-morpheme-tokenizer"
        console.print(f"[bold green]Pushing tokenizer to {repo_id}...")
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(folder_path=str(args.tokenizer_path), repo_id=repo_id)
        console.print(f"[bold]Done! {repo_id}")

    if args.factoids or args.all:
        repo_id = f"{args.org}/esperanto-factoids"
        console.print(f"[bold green]Loading factoids from {args.factoids_path}...")
        total_lines = sum(1 for _ in open(args.factoids_path))
        texts = []
        with open(args.factoids_path) as f:
            from rich.progress import Progress
            with Progress() as progress:
                task = progress.add_task("Reading...", total=total_lines)
                for line in f:
                    texts.append(json.loads(line)["text"])
                    progress.advance(task)
        console.print(f"[bold green]Pushing {len(texts):,} paragraphs to {repo_id}...")
        ds = Dataset.from_dict({"text": texts})
        ds.push_to_hub(repo_id)
        console.print(f"[bold]Done! {repo_id}")

    if args.sentences or args.all:
        repo_id = f"{args.org}/esperanto-sentences"
        console.print(f"[bold green]Pushing sentences to {repo_id}...")
        texts = []
        with open(args.sentences_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    texts.append(parts[2])
        ds = Dataset.from_dict({"text": texts})
        ds.push_to_hub(repo_id)
        console.print(f"[bold]Done! {len(texts):,} sentences → {repo_id}")


if __name__ == "__main__":
    main()
