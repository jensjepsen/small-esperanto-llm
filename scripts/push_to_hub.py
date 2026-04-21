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
    parser.add_argument("--sft-atomic-icl", action="store_true", help="Push ATOMIC ICL dataset")
    parser.add_argument("--sft-atomic-qa", action="store_true", help="Push ATOMIC multi-turn QA dataset")
    parser.add_argument("--sft-icl", action="store_true", help="Push Wikidata ICL dataset")
    parser.add_argument("--sft-morphology-icl", action="store_true", help="Push morphology ICL dataset")
    parser.add_argument("--sft-quantity-reasoning", action="store_true", help="Push quantity-reasoning word problems")
    parser.add_argument("--sft-dolly", action="store_true", help="Push translated Dolly-15K instructions")
    parser.add_argument("--gsm8k", action="store_true", help="Push translated GSM8K dataset")
    parser.add_argument("--arithmetic-cot", action="store_true", help="Push arithmetic CoT dataset")
    parser.add_argument("--hplt", action="store_true", help="Push HPLT Esperanto dataset (raw buckets, HPLT's own filter)")
    parser.add_argument("--hplt-filtered", action="store_true",
                        help="Push verifier-filtered HPLT Esperanto dataset (from data/hplt_filtered/)")
    parser.add_argument("--alpaca-cleaned", action="store_true",
                        help="Push cleaned Alpaca-EO dataset (from data/sft/alpaca_eo_clean_{train,test}.jsonl)")
    parser.add_argument("--gutenberg", action="store_true", help="Push Gutenberg books dataset")
    parser.add_argument("--all", action="store_true", help="Push everything")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("tokenizer_morpheme"))
    parser.add_argument("--factoids-path", type=Path,
                        default=Path("/mnt/data2/wikidata5m/eo_factoids_v2/factoid_text.jsonl"))
    parser.add_argument("--sentences-path", type=Path, default=Path("data/epo_sentences.tsv"))
    parser.add_argument("--sft-factoid-path", type=Path, default=Path("data/sft/sft_factoid.jsonl"))
    parser.add_argument("--sft-creative-path", type=Path, default=Path("data/sft/sft_creative.jsonl"))
    parser.add_argument("--sft-atomic-icl-path", type=Path, default=Path("data/sft/sft_atomic_icl.jsonl"))
    parser.add_argument("--sft-atomic-qa-path", type=Path, default=Path("data/sft/sft_atomic_qa.jsonl"))
    parser.add_argument("--sft-icl-path", type=Path, default=Path("data/sft/sft_icl.jsonl"))
    parser.add_argument("--sft-morphology-icl-path", type=Path, default=Path("data/sft/sft_morphology_icl.jsonl"))
    parser.add_argument("--sft-quantity-reasoning-path", type=Path, default=Path("data/sft/sft_quantity_reasoning.jsonl"))
    parser.add_argument("--sft-dolly-path", type=Path, default=Path("data/sft/sft_dolly.jsonl"))
    parser.add_argument("--gsm8k-dir", type=Path, default=Path("data/sft/gsm8k"))
    parser.add_argument("--arithmetic-cot-dir", type=Path, default=Path("data/sft/arithmetic_cot"))
    args = parser.parse_args()

    if not (args.tokenizer or args.factoids or args.sentences or args.sft_factoid or args.sft_creative or args.sft_atomic_icl or args.sft_atomic_qa or args.sft_icl or args.sft_morphology_icl or args.sft_quantity_reasoning or args.sft_dolly or args.gsm8k or args.arithmetic_cot or args.hplt or args.hplt_filtered or args.alpaca_cleaned or args.gutenberg or args.all):
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
        ("sft_atomic_icl", "sft_atomic_icl_path", "esperanto-sft-atomic-icl"),
        ("sft_atomic_qa", "sft_atomic_qa_path", "esperanto-sft-atomic-qa"),
        ("sft_icl", "sft_icl_path", "esperanto-sft-wikidata-icl"),
        ("sft_morphology_icl", "sft_morphology_icl_path", "esperanto-sft-morphology-icl"),
        ("sft_quantity_reasoning", "sft_quantity_reasoning_path", "esperanto-sft-quantity-reasoning"),
        ("sft_dolly", "sft_dolly_path", "esperanto-sft-dolly"),
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

    if args.arithmetic_cot or args.all:
        from datasets import DatasetDict
        repo_id = f"{args.org}/esperanto-arithmetic-cot"
        console.print(f"[bold green]Pushing arithmetic CoT from {args.arithmetic_cot_dir} to {repo_id}...")
        splits = {}
        for split in ["train", "test"]:
            path = args.arithmetic_cot_dir / f"{split}.jsonl"
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

    if args.hplt or args.all:
        from esperanto_lm.data import load_hplt_dataset
        repo_id = f"{args.org}/esperanto-hplt"
        console.print(f"[bold green]Loading and filtering HPLT Esperanto data...")
        ds = load_hplt_dataset()
        if ds is not None:
            console.print(f"[bold]  Filtered rows:[/] {len(ds):,}")
            ds.push_to_hub(repo_id)
            console.print(f"[bold]Done! {repo_id}")
        else:
            console.print("[red]No HPLT data found in data/hplt/")

    if args.hplt_filtered:
        repo_id = f"{args.org}/esperanto-hplt-filtered"
        filtered_dir = Path("data/hplt_filtered")
        jsonl_files = sorted(filtered_dir.glob("*.jsonl"))
        if not jsonl_files:
            console.print(f"[red]No filtered HPLT data found in {filtered_dir}/")
        else:
            console.print(f"[bold green]Loading verifier-filtered HPLT from "
                          f"{len(jsonl_files)} buckets...")
            ds = Dataset.from_json([str(f) for f in jsonl_files])
            console.print(f"[bold]  Docs:[/] {len(ds):,}")
            ds.push_to_hub(repo_id)
            console.print(f"[bold]Done! {repo_id}")

    if args.alpaca_cleaned:
        from datasets import DatasetDict
        repo_id = f"{args.org}/esperanto-alpaca-cleaned"
        train_path = Path("data/sft/alpaca_eo_clean_train.jsonl")
        test_path = Path("data/sft/alpaca_eo_clean_test.jsonl")
        if not train_path.exists() or not test_path.exists():
            console.print(f"[red]Missing files. Run scripts/clean_alpaca_eo.py first to "
                          f"produce {train_path} and {test_path}.")
        else:
            console.print(f"[bold green]Loading cleaned Alpaca-EO splits...")
            splits = {}
            for name, path in [("train", train_path), ("test", test_path)]:
                ds = Dataset.from_json(str(path))
                console.print(f"  {name}: {len(ds):,} rows")
                splits[name] = ds
            DatasetDict(splits).push_to_hub(repo_id)
            console.print(f"[bold]Done! {repo_id}")

    if args.gutenberg or args.all:
        from esperanto_lm.data import load_gutenberg_dataset
        repo_id = f"{args.org}/esperanto-gutenberg"
        console.print(f"[bold green]Loading Gutenberg books...")
        ds = load_gutenberg_dataset()
        if ds is not None:
            console.print(f"[bold]  Books:[/] {len(ds):,}")
            ds.push_to_hub(repo_id)
            console.print(f"[bold]Done! {repo_id}")
        else:
            console.print("[red]No Gutenberg data found in data/gutenberg/")


if __name__ == "__main__":
    main()
