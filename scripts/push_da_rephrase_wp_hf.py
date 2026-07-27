"""Push Danish wp-rephrase SFT dataset to HF Hub.

Data comes from `build_da_rephrase_wp.py`, which pairs each
(q_orig, q_new) from wp-reworded-v1 into forward + reverse
rephrase-instruction rows using ~80 varied templates.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("/home/jepsen/src/espllm/data/da_rephrase_wp_v1.jsonl"))
    ap.add_argument("--repo", default="jensjepsen/danish-rephrase-wp-v1")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        p = Path.home() / ".cache/huggingface/token"
        if p.exists():
            token = p.read_text().strip()
    if not token:
        print("no HF token", file=sys.stderr); sys.exit(2)

    rows = [json.loads(l) for l in args.input.open() if l.strip()]
    print(f"loaded {len(rows):,} rows", flush=True)

    fwd = sum(1 for r in rows if r.get("direction") == "orig_to_new")
    rev = sum(1 for r in rows if r.get("direction") == "new_to_orig")
    print(f"  forward (orig → new): {fwd:,}")
    print(f"  reverse (new → orig): {rev:,}")

    # sft config: messages-only, drop-in for train_sft_packed.py
    sft_rows = [{"messages": r["messages"]} for r in rows]
    print(f"\npushing sft → {args.repo}", flush=True)
    Dataset.from_list(sft_rows).shuffle(seed=42).push_to_hub(
        args.repo, config_name="sft", token=token, private=args.private)

    # default config: full rows (messages + direction + orig_idx)
    print(f"\npushing default → {args.repo}", flush=True)
    Dataset.from_list(rows).shuffle(seed=42).push_to_hub(
        args.repo, config_name="default", token=token, private=args.private)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
