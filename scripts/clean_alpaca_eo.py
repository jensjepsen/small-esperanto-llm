"""Clean the saillab/alpaca-esperanto-cleaned dataset for SFT use.

Applies three filters:
  1. Replace literal "nan" input strings with "" (pandas artifact).
  2. Drop rows whose output contains AI-assistant-leak phrases
     ("Kiel AI", "As an AI", etc.) — bleed-through from ChatGPT distillation.
  3. Drop rows whose output is code-heavy (triple backticks, dense braces,
     HTML tags, or many code keywords). Prose-about-code rows are kept.

Usage:
    uv run python scripts/clean_alpaca_eo.py \\
        --out-train data/sft/alpaca_eo_clean_train.jsonl \\
        --out-test  data/sft/alpaca_eo_clean_test.jsonl

Then push to HF Hub via `scripts/push_to_hub.py --alpaca-cleaned`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from rich.console import Console

console = Console()

SOURCE = "saillab/alpaca-esperanto-cleaned"

# ─── Filter 1: AI-assistant leak phrases ────────────────────────────────────
_AI_LEAK = re.compile(
    r"\b(?:Kiel AI|Mi estas AI|Kiel granda lingva? model|As an AI|I am an AI)\b",
    re.IGNORECASE,
)

# ─── Filter 2: code detection (output) ──────────────────────────────────────
_RE_CODE_FENCE = re.compile(r"```")
_RE_HTML_TAG   = re.compile(r"<(?:/?[A-Za-z][A-Za-z0-9]*|!DOCTYPE|!--)")
_RE_SCRIPT_KW  = re.compile(
    r"\b(?:def|class|public|private|protected|function|return|import|from|const|let|var|"
    r"if\s*\(|else\s*[{:]|while\s*\(|for\s*\(|switch\s*\(|try\s*[{:]|catch\s*\(|"
    r"true|false|null|undefined|void|int|float|double|string|bool|boolean)\b"
)
_RE_CURLY      = re.compile(r"[{};]")
_RE_CAMELCASE  = re.compile(r"\b[a-z]+[A-Z][a-zA-Z]+\b")


def is_code_heavy(text: str) -> bool:
    """Any one of: code fence, ≥3 HTML tags, ≥5 script keywords,
    ≥10 curly/semicolon, or ≥3 distinct camelCase identifiers."""
    if _RE_CODE_FENCE.search(text):
        return True
    if len(_RE_HTML_TAG.findall(text)) >= 3:
        return True
    if len(_RE_SCRIPT_KW.findall(text)) >= 5:
        return True
    if len(_RE_CURLY.findall(text)) >= 10:
        return True
    if len(set(_RE_CAMELCASE.findall(text))) >= 3:
        return True
    return False


def clean_row(row: dict) -> dict | None:
    """Apply filters, return cleaned row dict, or None if the row is dropped."""
    instruction = (row.get("instruction") or "").strip()
    input_ = (row.get("input") or "").strip()
    output = (row.get("output") or "").strip()

    # normalize "nan" inputs → ""
    if input_.lower() == "nan":
        input_ = ""

    # drop empty outputs
    if not output:
        return None

    # drop AI-leak outputs
    if _AI_LEAK.search(output):
        return None

    # drop code-heavy outputs
    if is_code_heavy(output):
        return None

    return {"instruction": instruction, "input": input_, "output": output}


def clean_split(ds) -> tuple[list[dict], dict]:
    kept, stats = [], {"total": 0, "empty_output": 0, "ai_leak": 0,
                       "code_heavy": 0, "kept": 0, "input_nan_normalized": 0}
    for row in ds:
        stats["total"] += 1
        instruction = (row.get("instruction") or "").strip()
        input_ = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        if input_.lower() == "nan":
            stats["input_nan_normalized"] += 1
            input_ = ""
        if not output:
            stats["empty_output"] += 1; continue
        if _AI_LEAK.search(output):
            stats["ai_leak"] += 1; continue
        if is_code_heavy(output):
            stats["code_heavy"] += 1; continue
        kept.append({"instruction": instruction, "input": input_, "output": output})
        stats["kept"] += 1
    return kept, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=SOURCE,
                        help=f"HF dataset id (default: {SOURCE})")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write cleaned rows to this JSONL path (combined train+test).")
    parser.add_argument("--out-train", type=Path, default=None,
                        help="Write cleaned train split here (JSONL).")
    parser.add_argument("--out-test", type=Path, default=None,
                        help="Write cleaned test split here (JSONL).")
    args = parser.parse_args()

    console.print(f"[bold green]Loading {args.source}...")
    raw = load_dataset(args.source)
    console.print(f"[bold]Splits:[/] {list(raw.keys())}")

    cleaned = {}
    for split_name, ds in raw.items():
        console.print(f"\n[bold green]Cleaning split: {split_name} ({len(ds):,} rows)[/]")
        kept, stats = clean_split(ds)
        cleaned[split_name] = kept
        console.print(f"  total:               {stats['total']:,}")
        console.print(f"  input 'nan' → '':    {stats['input_nan_normalized']:,}")
        console.print(f"  dropped empty:       {stats['empty_output']:,}")
        console.print(f"  dropped AI-leak:     {stats['ai_leak']:,}")
        console.print(f"  dropped code-heavy:  {stats['code_heavy']:,}")
        console.print(f"  [bold]kept:                {stats['kept']:,} "
                      f"({100*stats['kept']/stats['total']:.1f}%)[/]")

    # Write local JSONL output(s)
    def write_jsonl(path: Path, rows: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        console.print(f"[bold]wrote {len(rows):,} rows → {path}")

    if args.out:
        combined = []
        for split_rows in cleaned.values():
            combined.extend(split_rows)
        write_jsonl(args.out, combined)
    if args.out_train and "train" in cleaned:
        write_jsonl(args.out_train, cleaned["train"])
    if args.out_test and "test" in cleaned:
        write_jsonl(args.out_test, cleaned["test"])


if __name__ == "__main__":
    main()
