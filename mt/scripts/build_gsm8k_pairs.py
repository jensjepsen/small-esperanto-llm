"""Build en↔eo parallel from openai/gsm8k EN + local EO translations.

Aligns by row index. Pairs only the *question* (not the chain-of-thought
solution) — translation training shouldn't have the reasoning text in
the target.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eo-train", type=Path, default=Path("data/sft/gsm8k/train.jsonl"))
    ap.add_argument("--eo-test", type=Path, default=Path("data/sft/gsm8k/test.jsonl"))
    ap.add_argument("--out-train", type=Path, default=Path("mt/data/parallel/gsm8k_train.jsonl"))
    ap.add_argument("--out-test", type=Path, default=Path("mt/data/parallel/gsm8k_test.jsonl"))
    ap.add_argument("--hf-cache", type=str, default="/mnt/data/hf_cache")
    args = ap.parse_args()

    os.environ["HF_HOME"] = args.hf_cache
    from datasets import load_dataset

    for split, eo_path, out_path in [
        ("train", args.eo_train, args.out_train),
        ("test", args.eo_test, args.out_test),
    ]:
        ds = load_dataset("openai/gsm8k", "main", split=split)
        with eo_path.open() as f:
            eo_rows = [json.loads(l) for l in f]
        # alignment by index
        if len(ds) != len(eo_rows):
            print(f"  WARN {split}: en={len(ds)} eo={len(eo_rows)} — using min len")
        n = min(len(ds), len(eo_rows))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        with out_path.open("w") as fo:
            for i in range(n):
                en = ds[i]["question"].strip().replace("\n", " ")
                en = " ".join(en.split())
                # EO is in messages[0]['content']
                msgs = eo_rows[i].get("messages", [])
                if not msgs or msgs[0]["role"] != "user":
                    continue
                eo = msgs[0]["content"].strip().replace("\n", " ")
                eo = " ".join(eo.split())
                if not en or not eo:
                    continue
                fo.write(json.dumps({"en": en, "eo": eo, "src": "gsm8k"}, ensure_ascii=False) + "\n")
                kept += 1
        print(f"  {split}: kept {kept} pairs -> {out_path}")


if __name__ == "__main__":
    main()
