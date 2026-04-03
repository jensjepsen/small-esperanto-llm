"""Standalone script to download Esperanto Wikipedia via datasets."""

from rich.console import Console

from esperanto_lm.data import download_dataset

console = Console()


def main():
    console.print("[bold green]Downloading Esperanto Wikipedia...")
    dataset = download_dataset()
    console.print(f"[bold]Train examples:[/] {len(dataset['train']):,}")
    console.print(f"[bold]Test examples:[/] {len(dataset['test']):,}")
    console.print("[bold green]Done! Saved to data/eo_wiki")


if __name__ == "__main__":
    main()
