"""Split data/da_stem_v1/sft.jsonl into 3 buckets + push each to HF.

Buckets:
    danish-sci-reasoning-v1  — worked_calc/mechanism/counterfactual forward
    danish-sci-factcheck-v1  — SAND/FALSK verification forward
    danish-sci-taskgen-v1    — reverse generation (model produces tasks)

Val split: last N pageids in each bucket → validation. Also uploads the
raw sources (rows.jsonl) + prompt_templates.json under raw/ so consumers
can re-flatten with different templates.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi

BUCKETS = {
    "danish-sci-reasoning-v1":
        lambda st: not st.startswith("stem_gen_") and not st.startswith("stem_fact_check_"),
    "danish-sci-factcheck-v1":
        lambda st: st.startswith("stem_fact_check_"),
    "danish-sci-taskgen-v1":
        lambda st: st.startswith("stem_gen_"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", type=Path, default=Path("data/da_stem_v1/sft.jsonl"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/da_stem_v1"))
    ap.add_argument("--sci-raw", type=Path, default=Path("data/da_sci_reasoning_v1/rows.jsonl"))
    ap.add_argument("--fc-raw", type=Path, default=Path("data/da_fact_check_v1/rows.jsonl"))
    ap.add_argument("--val-pageids", type=int, default=200,
                    help="Last N distinct pageids in each bucket go to validation.")
    ap.add_argument("--user", default="jensjepsen")
    args = ap.parse_args()

    print(f"reading {args.sft}...", flush=True)
    by_bucket = {b: [] for b in BUCKETS}
    for line in args.sft.open():
        r = json.loads(line)
        for b, keep in BUCKETS.items():
            if keep(r["subtype"]):
                by_bucket[b].append(r); break

    api = HfApi()

    for bucket, rows in by_bucket.items():
        repo = f"{args.user}/{bucket}"
        # Val split: last N distinct pageids (sorted asc → tail is val)
        all_idx = sorted({r["pageid"] for r in rows})
        val_idx = set(all_idx[-args.val_pageids:])
        train = [r for r in rows if r["pageid"] not in val_idx]
        val = [r for r in rows if r["pageid"] in val_idx]

        print(f"\n=== {repo} ===", flush=True)
        print(f"  train={len(train):,}  val={len(val):,}  (val holds "
              f"{args.val_pageids} pageids)", flush=True)
        from collections import Counter
        c_tr = Counter(r["subtype"] for r in train)
        c_va = Counter(r["subtype"] for r in val)
        for st in sorted(set(c_tr) | set(c_va)):
            print(f"  {st:36s} train={c_tr.get(st,0):>7,}  val={c_va.get(st,0):>6,}")

        api.create_repo(repo, repo_type="dataset", exist_ok=True)
        Dataset.from_list(train).push_to_hub(repo, split="train",
                                              commit_message=f"train ({len(train)} rows)")
        Dataset.from_list(val).push_to_hub(repo, split="validation",
                                            commit_message=f"validation ({len(val)} rows)")

        # Upload raw source + templates. For taskgen, both raws are relevant.
        raw_files = []
        if bucket == "danish-sci-reasoning-v1":
            raw_files = [(args.sci_raw, "raw/rows.jsonl")]
        elif bucket == "danish-sci-factcheck-v1":
            raw_files = [(args.fc_raw, "raw/rows.jsonl")]
        else:  # taskgen — uses both
            raw_files = [(args.sci_raw, "raw/sci_reasoning.jsonl"),
                         (args.fc_raw, "raw/fact_check.jsonl")]
        for src, dst in raw_files:
            api.upload_file(path_or_fileobj=str(src),
                            path_in_repo=dst, repo_id=repo, repo_type="dataset",
                            commit_message="raw source")
        tpl = args.data_dir / "prompt_templates.json"
        if tpl.exists():
            api.upload_file(path_or_fileobj=str(tpl),
                            path_in_repo="raw/prompt_templates.json",
                            repo_id=repo, repo_type="dataset",
                            commit_message="prompt templates")
        print(f"  pushed https://huggingface.co/datasets/{repo}", flush=True)


if __name__ == "__main__":
    main()
