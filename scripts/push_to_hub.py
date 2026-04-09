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
    parser.add_argument("--sft-factoid", action="store_true", help="Push SFT factoid dataset")
    parser.add_argument("--sft-creative", action="store_true", help="Push SFT creative dataset")
    parser.add_argument("--gsm8k", action="store_true", help="Push translated GSM8K dataset")
    parser.add_argument("--all", action="store_true", help="Push everything")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("tokenizer_morpheme"))
    parser.add_argument("--factoids-path", type=Path,
                        default=Path("/mnt/data2/wikidata5m/eo_factoids/factoid_text.jsonl"))
    parser.add_argument("--sentences-path", type=Path, default=Path("data/epo_sentences.tsv"))
    parser.add_argument("--sft-factoid-path", type=Path, default=Path("data/sft/sft_factoid.jsonl"))
    parser.add_argument("--sft-creative-path", type=Path, default=Path("data/sft/sft_creative.jsonl"))
    parser.add_argument("--gsm8k-dir", type=Path, default=Path("data/sft/gsm8k"))
    args = parser.parse_args()

    if not (args.tokenizer or args.factoids or args.sentences or args.sft_factoid or args.sft_creative or args.gsm8k or args.all):
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

    for flag, path_attr, hub_name in [
        ("sft_factoid", "sft_factoid_path", "esperanto-sft-factoid"),
        ("sft_creative", "sft_creative_path", "esperanto-sft-creative"),
    ]:
        if getattr(args, flag) or args.all:
            path = getattr(args, path_attr)
            repo_id = f"{args.org}/{hub_name}"
            console.print(f"[bold green]Pushing SFT data from {path} to {repo_id}...")
            data = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            ds = Dataset.from_list(data)
            ds.push_to_hub(repo_id)
            console.print(f"[bold]Done! {len(data):,} conversations → {repo_id}")

    if args.gsm8k or args.all:
        from datasets import DatasetDict
        repo_id = f"{args.org}/esperanto-gsm8k"
        console.print(f"[bold green]Pushing GSM8K from {args.gsm8k_dir} to {repo_id}...")
        splits = {}
        for split in ["train", "test"]:
            path = args.gsm8k_dir / f"{split}.jsonl"
            data = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            splits[split] = Dataset.from_list(data)
            console.print(f"[bold]  {split}:[/] {len(data):,}")
        ds = DatasetDict(splits)
        ds.push_to_hub(repo_id)
        console.print(f"[bold]Done! {repo_id}")


if __name__ == "__main__":
    main()
