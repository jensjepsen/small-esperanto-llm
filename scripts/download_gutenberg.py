"""Download Esperanto books from Project Gutenberg via Gutendex API."""

import time
from pathlib import Path

import requests
from rich.console import Console

console = Console()

GUTENDEX_URL = "https://gutendex.com/books/"
DATA_DIR = Path("data/gutenberg")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Fetching Esperanto book list from Gutendex...")
    books = []
    url = GUTENDEX_URL
    params = {"languages": "eo"}

    while url:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        books.extend(data["results"])
        url = data.get("next")
        params = None  # next URL already has params

    console.print(f"[bold]Found {len(books)} Esperanto books")

    downloaded = 0
    skipped = 0
    for book in books:
        book_id = book["id"]
        title = book.get("title", "Unknown")
        output_path = DATA_DIR / f"{book_id}.txt"

        if output_path.exists():
            skipped += 1
            continue

        # Find plain text URL
        formats = book.get("formats", {})
        txt_url = None
        for fmt, url in formats.items():
            if "text/plain" in fmt and "utf-8" in fmt:
                txt_url = url
                break
        if txt_url is None:
            for fmt, url in formats.items():
                if "text/plain" in fmt:
                    txt_url = url
                    break

        if txt_url is None:
            console.print(f"[yellow]No text format for {book_id}: {title}[/]")
            continue

        console.print(f"[dim]Downloading {book_id}: {title}[/]")
        try:
            resp = requests.get(txt_url, timeout=30)
            resp.raise_for_status()
            output_path.write_text(resp.text, encoding="utf-8")
            downloaded += 1
        except Exception as e:
            console.print(f"[red]Failed {book_id}: {e}[/]")

        time.sleep(2)  # respect rate limits

    console.print(f"[bold green]Done![/] Downloaded: {downloaded}, Skipped: {skipped}")
    console.print(f"[bold]Total files:[/] {len(list(DATA_DIR.glob('*.txt')))}")


if __name__ == "__main__":
    main()
