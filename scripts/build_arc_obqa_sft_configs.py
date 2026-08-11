"""Build `sft` config for jensjepsen/danish-arc + jensjepsen/danish-openbookqa.

For each source row, emit TWO SFT rows:
  1. with-choices  →  user shows Q + A/B/C/D options, assistant emits "X) text"
  2. no-choices    →  user shows Q only, assistant emits "text"

Style 1 mirrors danish-sciq:sft. Style 2 is free-form answer prediction.

Pushes to the SAME repo as `sft` config, preserving train/validation/test splits.

Usage:
    uv run python scripts/build_arc_obqa_sft_configs.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


def rows_for(item: dict) -> list[dict]:
    q = item["question"].strip()
    choices = item["choices"]
    labels = [c["label"] for c in choices]
    texts = [c["text"] for c in choices]
    gold = item["answerKey"]
    if gold not in labels:
        return []
    gold_text = texts[labels.index(gold)]

    # Style 1: with choices, letter+text answer
    choice_block = "\n".join(f"{L}) {T}" for L, T in zip(labels, texts))
    row_wc = {
        "messages": [
            {"role": "user", "content": f"Spørgsmål: {q}\n{choice_block}"},
            {"role": "assistant", "content": f"{gold}) {gold_text}"},
        ]
    }
    # Style 2: no choices, free-form answer
    row_nc = {
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": gold_text},
        ]
    }
    return [row_wc, row_nc]


def push_config(repo: str, cfg_name: str | None, sft_cfg: str = "sft"):
    """Load `repo` at `cfg_name`, build SFT rows, push under `sft_cfg`."""
    src_desc = f"{repo}:{cfg_name or '<default>'}"
    print(f"\n=== {src_desc}")
    if cfg_name:
        dd = load_dataset(repo, cfg_name)
    else:
        dd = load_dataset(repo)
    out_dd = {}
    for split, ds in dd.items():
        rows = []
        for item in ds:
            rows.extend(rows_for(item))
        print(f"  {split:<12} {len(ds):>5} src → {len(rows):>5} sft rows")
        out_dd[split] = Dataset.from_list(rows)
    token = os.getenv("HF_TOKEN")
    if not token:
        tp = Path.home() / ".cache/huggingface/token"
        if tp.exists():
            token = tp.read_text().strip()
    if not token:
        print("no HF token", file=sys.stderr); sys.exit(2)
    print(f"  pushing → {repo} :: {sft_cfg}")
    DatasetDict(out_dd).push_to_hub(repo, config_name=sft_cfg, token=token)


def main():
    # danish-arc has two configs (arc_challenge, arc_easy). Build one merged sft
    # config that combines both configs, keeping split boundaries.
    print("=== building merged arc sft (challenge + easy)")
    from collections import defaultdict
    merged = defaultdict(list)
    for cfg in ("arc_challenge", "arc_easy"):
        dd = load_dataset("jensjepsen/danish-arc", cfg)
        for split, ds in dd.items():
            for item in ds:
                merged[split].extend(rows_for(item))
    for split, rows in merged.items():
        print(f"  {split:<12} → {len(rows):>5} sft rows")
    token = os.getenv("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").read_text().strip()
    DatasetDict({s: Dataset.from_list(rs) for s, rs in merged.items()}).push_to_hub(
        "jensjepsen/danish-arc", config_name="sft", token=token)

    # openbookqa has a single "main" config
    push_config("jensjepsen/danish-openbookqa", "main", sft_cfg="sft")


if __name__ == "__main__":
    main()
