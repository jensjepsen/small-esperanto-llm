"""Push diverse word-problems to HF Hub.

Two configs:
  - default: full schema (type, question_eo, chain_eo, answer, math_language,
             wrapper_tone, question_form, strategy)
  - sft: messages format
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset


def to_messages(rows):
    out = []
    for r in rows:
        out.append({
            "messages": [
                {"role": "user", "content": r["question_eo"]},
                {"role": "assistant", "content": r["chain_eo"]},
            ],
            "type": r.get("type", ""),
            "answer": r.get("answer"),
            "math_language": r.get("math_language", ""),
            "wrapper_tone": r.get("wrapper_tone", ""),
            "question_form": r.get("question_form", ""),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    token = os.getenv("HF_HUB_TOKEN") or (Path.home() / ".cache/huggingface/token").read_text().strip()

    rows = [json.loads(l) for l in args.input.open()]
    # default schema (strip dict params if present to keep schema simple)
    for r in rows:
        r.pop("params", None)
        if "answer" in r:
            r["answer"] = str(r["answer"])  # uniform string for HF schema
    print(f"loaded {len(rows):,} rows", flush=True)

    full = Dataset.from_list(rows).shuffle(seed=args.seed)
    sft = Dataset.from_list(to_messages(rows)).shuffle(seed=args.seed)

    print(f"pushing default to {args.repo}…", flush=True)
    full.push_to_hub(args.repo, config_name="default", token=token, private=True)
    print(f"pushing sft to {args.repo}…", flush=True)
    sft.push_to_hub(args.repo, config_name="sft", token=token, private=True)
    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
