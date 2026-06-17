"""Download FLORES+ en↔eo (devtest, 1012 pairs) — clean OOD eval.

Uses the open mirror `alexei-v-ivanov-amd/flores_plus` (parquet, ungated)
since the canonical `facebook/flores` is gated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("mt/data/parallel/flores_devtest.jsonl"))
    ap.add_argument("--hf-cache", type=str, default="/mnt/data/hf_cache")
    args = ap.parse_args()

    os.environ["HF_HOME"] = args.hf_cache

    import pandas as pd
    from huggingface_hub import hf_hub_download

    en_p = hf_hub_download("alexei-v-ivanov-amd/flores_plus", "eng_Latn.parquet", repo_type="dataset")
    eo_p = hf_hub_download("alexei-v-ivanov-amd/flores_plus", "epo_Latn.parquet", repo_type="dataset")
    en = pd.read_parquet(en_p)[["id", "text"]].rename(columns={"text": "en"})
    eo = pd.read_parquet(eo_p)[["id", "text"]].rename(columns={"text": "eo"})
    df = en.merge(eo, on="id", how="inner")
    assert len(df) == len(en) == len(eo), f"id mismatch: en={len(en)} eo={len(eo)} pair={len(df)}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for _, r in df.iterrows():
            f.write(json.dumps({"en": r["en"], "eo": r["eo"], "src": "flores+"}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(df)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
