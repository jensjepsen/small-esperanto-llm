"""Push Danish Dolly translation to HF Hub.

Two configs:
  - default: full parallel schema (both EN source + DA translation per row)
  - sft:     Danish-only messages format for direct SFT consumption

Usage:
  HF_HUB_TOKEN=$(cat ~/.cache/huggingface/token) \\
  python scripts/push_dolly_da_hf.py \\
      --input /mnt/data2/dolly_da_full.jsonl \\
      --repo jensjepsen/danish-dolly-15k
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset


def load_rows(path: Path):
    rows = []
    for line in path.open():
        r = json.loads(line)
        if r.get("da") is None:
            continue
        rows.append(r)
    return rows


def to_default(rows):
    """Full parallel schema — flat top-level fields for HF Arrow."""
    return [{
        "id": r["id"],
        "category": r["category"],
        "en_instruction": r["en"]["instruction"],
        "en_context":     r["en"]["context"],
        "en_response":    r["en"]["response"],
        "da_instruction": r["da"]["instruction"],
        "da_context":     r["da"]["context"],
        "da_response":    r["da"]["response"],
    } for r in rows]


def to_sft(rows):
    """Danish messages format:
       user = instruction (with context appended when present)
       assistant = response"""
    out = []
    for r in rows:
        da = r["da"]
        user_content = da["instruction"]
        if da["context"].strip():
            user_content = f"{da['instruction']}\n\n{da['context']}"
        out.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": da["response"]},
            ],
            "category": r["category"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-configs", nargs="*", default=[])
    args = ap.parse_args()

    token = args.token or os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        p = Path.home() / ".cache/huggingface/token"
        if p.exists():
            token = p.read_text().strip()
    if not token:
        print("No HF token found. Pass --token or set HF_HUB_TOKEN.",
              file=sys.stderr)
        sys.exit(2)

    print(f"loading {args.input}…", flush=True)
    rows = load_rows(args.input)
    print(f"  {len(rows):,} translated rows", flush=True)

    if "default" not in args.skip_configs:
        print("pushing default config…", flush=True)
        ds = Dataset.from_list(to_default(rows)).shuffle(seed=args.seed)
        ds.push_to_hub(args.repo, config_name="default", token=token,
                       private=args.private)

    if "sft" not in args.skip_configs:
        print("pushing sft config…", flush=True)
        ds = Dataset.from_list(to_sft(rows)).shuffle(seed=args.seed)
        ds.push_to_hub(args.repo, config_name="sft", token=token,
                       private=args.private)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
