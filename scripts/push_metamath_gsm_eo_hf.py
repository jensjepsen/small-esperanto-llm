"""Push MetaMath-GSM-EO (Gemini-translated GSM_* subset) to HF Hub.

Two configs:
  - default: parallel EN/EO schema with metadata and per-row token usage
      {orig_idx, type, q_en, q_eo, a_en, a_eo, original_question,
       input_tokens, output_tokens, thoughts_tokens}
  - sft: chat-messages format for AR-LLM SFT (EO only)
      {messages: [{role, content}...], orig_idx, type}

MetaMathQA has no validation split — everything lives under `train`. Rows
whose translation was rejected (missing q_eo/a_eo or a `reject_reason`
field) are dropped.

Usage:
  uv run python scripts/push_metamath_gsm_eo_hf.py \\
    --input /mnt/data2/metamath_gsm_eo.jsonl \\
    --repo jensjepsen/esperanto-metamath-gsm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Drop translator failures (either translation missing or rejected)
            if r.get("reject_reason"):
                continue
            if not (r.get("q_eo") and r.get("a_eo")):
                continue
            rows.append({
                "orig_idx": r["orig_idx"],
                "type": r.get("type", ""),
                "original_question": r.get("original_question", ""),
                "q_en": r["q_en"],
                "q_eo": r["q_eo"],
                "a_en": r["a_en"],
                "a_eo": r["a_eo"],
                "input_tokens": int(r.get("input_tokens", 0) or 0),
                "output_tokens": int(r.get("output_tokens", 0) or 0),
                "thoughts_tokens": int(r.get("thoughts_tokens", 0) or 0),
            })
    return rows


def to_sft(rows: list[dict]) -> list[dict]:
    """Convert to messages format for SFT (question -> answer, EO only)."""
    out = []
    for r in rows:
        q = (r["q_eo"] or "").strip()
        a = (r["a_eo"] or "").strip()
        if not q or not a:
            continue
        out.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "orig_idx": r["orig_idx"],
            "type": r["type"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/mnt/data2/metamath_gsm_eo.jsonl"))
    ap.add_argument("--repo", default="jensjepsen/esperanto-metamath-gsm")
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

    default_dd = DatasetDict({"train": Dataset.from_list(rows)})
    sft_rows = to_sft(rows)
    sft_dd = DatasetDict({"train": Dataset.from_list(sft_rows)})

    print(f"  default: {len(rows):,}")
    print(f"  sft:     {len(sft_rows):,}")

    print(f"\npushing default config to {args.repo}…", flush=True)
    default_dd.push_to_hub(args.repo, config_name="default", token=token,
                           private=args.private)

    print(f"pushing sft config to {args.repo}…", flush=True)
    sft_dd.push_to_hub(args.repo, config_name="sft", token=token,
                       private=args.private)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
