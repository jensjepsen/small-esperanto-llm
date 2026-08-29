"""Push SQuAD-EO to HF Hub.

Two configs:
  - default: parallel EN/EO schema
      {id, title, split, context_en, context_eo, question_en, question_eo,
       answers_en (list), answers_eo (list)}
  - sft: chat-messages format for direct AR-LLM SFT consumption (EO only)
      {messages: [{role, content}...], id, title, split}

Splits mirror SQuAD: ``train`` and ``validation``, taken from the ``split``
field in the source JSONL.

Usage:
  uv run python scripts/push_squad_eo_hf.py \\
    --input /mnt/data2/squad_eo.jsonl \\
    --repo jensjepsen/esperanto-squad
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict


SFT_TEMPLATE = (
    "Kunteksto:\n{context}\n\n"
    "Demando: {question}"
)


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Skip translator failures — need all four EO fields plus ≥1 answer
            if not (r.get("context_eo") and r.get("question_eo")
                    and r.get("answers_eo")):
                continue
            rows.append(r)
    return rows


def split_by_field(rows: list[dict], field: str = "split") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r.get(field, "train"), []).append(r)
    return out


def to_sft(rows: list[dict]) -> list[dict]:
    """Convert to messages format. Uses the first EO answer as the target;
    other gold variants stay accessible via the default config for eval."""
    out = []
    for r in rows:
        ans_list = r.get("answers_eo") or []
        if not ans_list:
            continue
        target = ans_list[0].strip()
        if not target:
            continue
        user = SFT_TEMPLATE.format(
            context=r["context_eo"].strip(),
            question=r["question_eo"].strip(),
        )
        out.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": target},
            ],
            "id": r["id"],
            "title": r["title"],
            "split": r["split"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/mnt/data2/squad_eo.jsonl"))
    ap.add_argument("--repo", default="jensjepsen/esperanto-squad")
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = args.token or os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        token_path = Path.home() / ".cache/huggingface/token"
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        print("No HF token found.", file=sys.stderr)
        sys.exit(2)

    print(f"loading {args.input}…", flush=True)
    rows = load_rows(args.input)
    print(f"  {len(rows):,} rows loaded (post filter)", flush=True)

    # Default config: parallel schema, keep the natural train/validation splits
    default_splits = split_by_field(rows)
    default_dd = DatasetDict({s: Dataset.from_list(rs)
                              for s, rs in default_splits.items()})
    print(f"  default splits: "
          f"{ {s: len(rs) for s, rs in default_splits.items()} }", flush=True)

    # SFT config: EO messages format
    sft_rows = to_sft(rows)
    sft_splits = split_by_field(sft_rows)
    sft_dd = DatasetDict({s: Dataset.from_list(rs)
                          for s, rs in sft_splits.items()})
    print(f"  sft splits:     "
          f"{ {s: len(rs) for s, rs in sft_splits.items()} }", flush=True)

    print(f"\npushing default config to {args.repo}…", flush=True)
    default_dd.push_to_hub(args.repo, config_name="default", token=token,
                           private=args.private)

    print(f"pushing sft config to {args.repo}…", flush=True)
    sft_dd.push_to_hub(args.repo, config_name="sft", token=token,
                       private=args.private)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")
    for s, ds in default_dd.items():
        print(f"  default[{s}]: {len(ds):,}")
    for s, ds in sft_dd.items():
        print(f"  sft[{s}]:     {len(ds):,}")


if __name__ == "__main__":
    main()
