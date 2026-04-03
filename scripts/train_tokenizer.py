"""Standalone script to train a BPE tokenizer on the Esperanto Wikipedia corpus."""

from rich.console import Console

from esperanto_lm.data import download_dataset, train_tokenizer

console = Console()


def main():
    console.print("[bold green]Loading dataset...")
    dataset = download_dataset()

    console.print("[bold green]Training tokenizer (vocab_size=8000)...")
    tokenizer = train_tokenizer(dataset["train"])
    console.print(f"[bold]Vocab size:[/] {tokenizer.vocab_size}")
    console.print("[bold green]Done! Saved to tokenizer/")


if __name__ == "__main__":
    main()
