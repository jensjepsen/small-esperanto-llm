"""Push wiki-grounded Danish SFT rows to HF Hub.

Two configs:
  - default: full schema {category, title, instruction, context, response}
  - sft: messages format for direct SFT consumption
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset


VALID_CATEGORIES = {"open_qa", "closed_qa", "general_qa", "brainstorming",
                    "creative_writing", "information_extraction",
                    "summarization", "classification"}


def load_ok(path: Path):
    rows = []
    for line in path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error") or r.get("skip"):
            continue
        cat = r.get("category", "")
        # Fix common typo: general_qc → general_qa
        if cat not in VALID_CATEGORIES:
            if cat == "general_qc":
                cat = "general_qa"
            else:
                continue
        rows.append({
            "category": cat,
            "title": r.get("title", ""),
            "instruction": r["instruction"],
            "context": r.get("context") or "",
            "response": r["response"],
        })
    return rows


def to_sft(rows):
    """Messages format. If context exists, prepend to instruction."""
    out = []
    for r in rows:
        if r["context"]:
            user = f"{r['context']}\n\n{r['instruction']}"
        else:
            user = r["instruction"]
        out.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": r["response"]},
            ],
            "category": r["category"],
            "title": r["title"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("/mnt/data2/wiki_grounded_da/full.jsonl"))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    token = args.token or os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        p = Path.home() / ".cache/huggingface/token"
        if p.exists(): token = p.read_text().strip()
    if not token:
        print("No HF token found.", file=sys.stderr); sys.exit(2)

    print(f"loading {args.input}…", flush=True)
    rows = load_ok(args.input)
    print(f"  {len(rows):,} rows after filter/normalize", flush=True)

    print("building configs…", flush=True)
    full = Dataset.from_list(rows).shuffle(seed=args.seed)
    sft  = Dataset.from_list(to_sft(rows)).shuffle(seed=args.seed)

    print(f"pushing to {args.repo} (default + sft)…", flush=True)
    full.push_to_hub(args.repo, config_name="default",
                      token=token, private=args.private)
    sft.push_to_hub(args.repo,  config_name="sft",
                     token=token, private=args.private)
    print("done")


if __name__ == "__main__":
    main()
