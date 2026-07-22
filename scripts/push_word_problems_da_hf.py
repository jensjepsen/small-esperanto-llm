"""Push the Danish word-problems dataset to HF Hub.

Mirrors scripts/push_word_problems_hf.py for the EO version, adapted for
Danish fields (question_da/chain_da) and the multi-turn `funcall` config.

Three configs:
  - default: full schema (type, question_da, chain_da, answer, strategy,
             params-as-JSON, steps-as-JSON, funcall-as-JSON)
  - sft:     single-turn messages (user=question, assistant=chain_da)
  - funcall: multi-turn messages incl. calculator tool calls

Usage:
  HF_HUB_TOKEN=$(cat ~/.cache/huggingface/token) \\
  uv run python scripts/push_word_problems_da_hf.py \\
    --input /mnt/data2/word_problems_da/all_word_problems.jsonl \\
    --repo jensjepsen/danish-word-problems
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset


def load_full(path: Path):
    rows = []
    for line in path.open():
        try:
            r = json.loads(line)
            # HF Arrow can't hold heterogeneous nested dicts across all rows;
            # serialize the compositional fields as JSON strings.
            if isinstance(r.get("params"), dict):
                r["params"] = json.dumps(r["params"], ensure_ascii=False)
            if isinstance(r.get("steps"), list):
                r["steps"] = json.dumps(r["steps"], ensure_ascii=False)
            if isinstance(r.get("funcall"), list):
                r["funcall"] = json.dumps(r["funcall"], ensure_ascii=False)
            rows.append(r)
        except json.JSONDecodeError:
            continue
    return rows


def to_sft_messages(rows):
    """Single-turn: user=question, assistant=full inline-prose chain."""
    return [{
        "messages": [
            {"role": "user", "content": r["question_da"]},
            {"role": "assistant", "content": r["chain_da"]},
        ],
        "type": r.get("type", ""),
        "answer": r.get("answer"),
        "strategy": r.get("strategy", ""),
    } for r in rows]


def to_funcall_messages(rows):
    """Multi-turn: use the pre-computed funcall list which interleaves
    assistant narration, calculator tool calls, and tool results."""
    out = []
    for r in rows:
        fc = r.get("funcall")
        if isinstance(fc, str):
            fc = json.loads(fc)
        if not fc:
            continue
        out.append({
            "messages": fc,
            "type": r.get("type", ""),
            "answer": r.get("answer"),
            "strategy": r.get("strategy", ""),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--repo", required=True,
                    help="e.g. jensjepsen/danish-word-problems")
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-configs", nargs="*", default=[],
                    choices=["default", "sft", "funcall"],
                    help="skip these configs (e.g. for retries)")
    args = ap.parse_args()

    token = args.token or os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        token_path = Path.home() / ".cache/huggingface/token"
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        print("No HF token found. Pass --token or set HF_HUB_TOKEN.",
              file=sys.stderr)
        sys.exit(2)

    print(f"loading {args.input}…", flush=True)
    rows = load_full(args.input)
    print(f"  {len(rows):,} rows", flush=True)

    if "default" not in args.skip_configs:
        print("pushing default config…", flush=True)
        full_ds = Dataset.from_list(rows).shuffle(seed=args.seed)
        full_ds.push_to_hub(args.repo, config_name="default",
                            token=token, private=args.private)

    if "sft" not in args.skip_configs:
        print("pushing sft config…", flush=True)
        sft_ds = Dataset.from_list(to_sft_messages(rows)).shuffle(seed=args.seed)
        sft_ds.push_to_hub(args.repo, config_name="sft",
                           token=token, private=args.private)

    if "funcall" not in args.skip_configs:
        print("pushing funcall config…", flush=True)
        fc_ds = Dataset.from_list(to_funcall_messages(rows)).shuffle(seed=args.seed)
        fc_ds.push_to_hub(args.repo, config_name="funcall",
                          token=token, private=args.private)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
