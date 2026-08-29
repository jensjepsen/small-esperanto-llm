"""Push MetaMath-GSM-DA (gemma-3-12b-translated GSM_* subset) to HF Hub.

Mirror of push_metamath_gsm_eo_hf.py adapted for Danish fields (q_da, a_da)
and gemma-via-OpenRouter cost fields.

Two configs:
  - default: parallel EN/DA schema with metadata and per-row token usage
      {orig_idx, type, q_en, q_da, a_en, a_da, original_question,
       input_tokens, output_tokens, cost}
  - sft: chat-messages format (DA only)
      {messages: [{role, content}...], orig_idx, type}

Rows with reject_reason or missing translations are dropped.

Usage:
  uv run python scripts/push_metamath_gsm_da_hf.py \\
      --input /mnt/data2/metamath_da/gsm.jsonl \\
      --repo jensjepsen/danish-metamath-gsm
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
            if r.get("reject_reason"):
                continue
            if not (r.get("q_da") and r.get("a_da")):
                continue
            rows.append({
                "orig_idx": r["orig_idx"],
                "type": r.get("type", ""),
                "original_question": r.get("original_question", ""),
                "q_en": r["q_en"],
                "q_da": r["q_da"],
                "a_en": r["a_en"],
                "a_da": r["a_da"],
                "input_tokens": int(r.get("input_tokens", 0) or 0),
                "output_tokens": int(r.get("output_tokens", 0) or 0),
                "cost": float(r.get("cost", 0) or 0),
            })
    return rows


def to_sft(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        q = (r["q_da"] or "").strip()
        a = (r["a_da"] or "").strip()
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
    ap.add_argument("--input", type=Path,
                    default=Path("/mnt/data2/metamath_da/gsm.jsonl"))
    ap.add_argument("--repo", default="jensjepsen/danish-metamath-gsm")
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = args.token or os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        p = Path.home() / ".cache/huggingface/token"
        if p.exists(): token = p.read_text().strip()
    if not token:
        print("no HF token", file=sys.stderr); sys.exit(2)

    print(f"loading {args.input}…", flush=True)
    rows = load_rows(args.input)
    print(f"  {len(rows):,} rows after filter", flush=True)

    default_dd = DatasetDict({"train": Dataset.from_list(rows)})
    sft_rows = to_sft(rows)
    sft_dd = DatasetDict({"train": Dataset.from_list(sft_rows)})

    print(f"  default: {len(rows):,}")
    print(f"  sft:     {len(sft_rows):,}")

    print(f"\npushing default → {args.repo}…", flush=True)
    default_dd.push_to_hub(args.repo, config_name="default", token=token,
                            private=args.private)
    print(f"pushing sft → {args.repo}…", flush=True)
    sft_dd.push_to_hub(args.repo, config_name="sft", token=token,
                        private=args.private)
    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
