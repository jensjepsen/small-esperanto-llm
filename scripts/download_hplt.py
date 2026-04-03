"""Download HPLT v3.0 Esperanto web corpus."""

import argparse
import json
import subprocess
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

console = Console()

MAP_URL = "https://data.hplt-project.org/three/sorted/epo_Latn.map"
DATA_DIR = Path("data/hplt")


def main():
    parser = argparse.ArgumentParser(description="Download HPLT v3.0 Esperanto data")
    parser.add_argument(
        "--min-score",
        type=int,
        default=7,
        help="Minimum quality score bucket to download (5-10, default: 7)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory for downloaded data",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch the map file to get URLs
    console.print("[bold green]Fetching HPLT file list...")
    result = subprocess.run(
        ["curl", "-sL", MAP_URL], capture_output=True, text=True, check=True
    )
    urls = [u.strip() for u in result.stdout.strip().splitlines() if u.strip()]

    # Filter by minimum quality score
    selected = []
    for url in urls:
        filename = url.rsplit("/", 1)[-1]
        score = int(filename.split("_")[0])
        if score >= args.min_score:
            selected.append(url)

    console.print(f"[bold]Files matching score >= {args.min_score}:[/] {len(selected)}")
    for url in selected:
        console.print(f"  {url}")

    # Download and decompress
    for url in selected:
        filename = url.rsplit("/", 1)[-1]
        jsonl_name = filename.replace(".zst", "")
        output_path = args.output_dir / jsonl_name

        if output_path.exists():
            console.print(f"[dim]Skipping {jsonl_name} (already exists)[/]")
            continue

        console.print(f"[bold green]Downloading {filename}...")
        subprocess.run(
            f'curl -sL "{url}" | zstd -d > "{output_path}"',
            shell=True,
            check=True,
        )

    # Print stats
    total_docs = 0
    for jsonl_file in sorted(args.output_dir.glob("*.jsonl")):
        count = sum(1 for _ in open(jsonl_file))
        console.print(f"[bold]{jsonl_file.name}:[/] {count:,} documents")
        total_docs += count

    console.print(f"[bold green]Total:[/] {total_docs:,} documents")


if __name__ == "__main__":
    main()
