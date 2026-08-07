"""Split SFT rows into train/validation and push to HF Hub.

Also uploads the raw generation JSONL (rc.jsonl / reason.jsonl / textman.jsonl)
and prompt_templates.json to `raw/` in the same repo so anyone can regenerate
the SFT view.

Val split: last 1000 orig_idx (sorted asc) — all rows tied to those articles
go to validation, everything else to train.

Repos:
    jensjepsen/danish-rc-v1
    jensjepsen/danish-reason-v1
    jensjepsen/danish-textman-v1
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi


DATASETS = [
    ("rc",      "jensjepsen/danish-rc-v1"),
    ("reason",  "jensjepsen/danish-reason-v1"),
    ("textman", "jensjepsen/danish-textman-v1"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/task_expansion_v1"))
    ap.add_argument("--val-articles", type=int, default=1000)
    args = ap.parse_args()

    api = HfApi()

    for name, repo in DATASETS:
        sft_path = args.data_dir / "sft" / f"{name}.jsonl"
        raw_path = args.data_dir / f"{name}.jsonl"
        assert sft_path.exists() and raw_path.exists(), f"missing {name}"

        print(f"\n=== {name} → {repo} ===", flush=True)
        rows = [json.loads(l) for l in sft_path.open()]
        all_idx = sorted({r["orig_idx"] for r in rows})
        val_idx = set(all_idx[-args.val_articles:])
        train = [r for r in rows if r["orig_idx"] not in val_idx]
        val   = [r for r in rows if r["orig_idx"] in val_idx]
        print(f"  train={len(train):,}  val={len(val):,}  "
              f"(val holds {args.val_articles} articles)", flush=True)

        # subtype dist per split
        from collections import Counter
        for lbl, split in [("train", train), ("val", val)]:
            c = Counter(r["subtype"] for r in split)
            print(f"  {lbl}: {dict(c)}")

        api.create_repo(repo, repo_type="dataset", exist_ok=True)
        Dataset.from_list(train).push_to_hub(repo, split="train",
                                              commit_message=f"train split ({len(train)} rows)")
        Dataset.from_list(val).push_to_hub(repo, split="validation",
                                            commit_message=f"validation split ({len(val)} rows)")

        # Upload raw + templates so consumers can re-flatten differently
        api.upload_file(path_or_fileobj=str(raw_path),
                        path_in_repo=f"raw/{name}.jsonl",
                        repo_id=repo, repo_type="dataset",
                        commit_message=f"raw {name} generation output")
        api.upload_file(path_or_fileobj=str(args.data_dir / "prompt_templates.json"),
                        path_in_repo="raw/prompt_templates.json",
                        repo_id=repo, repo_type="dataset",
                        commit_message="prompt template pools used for flattening")
        print(f"  pushed to https://huggingface.co/datasets/{repo}", flush=True)


if __name__ == "__main__":
    main()
