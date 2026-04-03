"""Download mc4 Esperanto corpus from allenai/c4."""

from pathlib import Path

from rich.console import Console

from datasets import load_dataset

console = Console()

DATA_DIR = Path("data/mc4")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_path = DATA_DIR / "eo"

    if save_path.exists():
        console.print("[dim]mc4 already downloaded, skipping[/]")
        return

    console.print("[bold green]Downloading mc4 Esperanto...")
    ds = load_dataset("allenai/c4", "eo", split="train")
    console.print(f"[bold]Documents:[/] {len(ds):,}")

    ds.save_to_disk(str(save_path))
    console.print(f"[bold green]Done! Saved to {save_path}")


if __name__ == "__main__":
    main()
