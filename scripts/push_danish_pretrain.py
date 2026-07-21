"""Push /workspace/work/dedup/*.jsonl.gz directly to HF as a raw JSONL dataset.

Uses HfApi.upload_folder to avoid materializing a full HF Dataset object
(which would balloon disk with arrow files for ~95M docs). JSONL.gz shards
are uploaded as-is; the HF dataset loader can stream them natively via
the load_dataset("json", ...) code path.

Usage:
    HF_TOKEN=... python push_danish_pretrain.py [--repo jensjepsen/danish-pretrain]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DEDUP_DIR = Path("/workspace/work/dedup")

README_TEMPLATE = """---
license: cc-by-4.0
language:
- da
size_categories:
- 10M<n<100M
---

# Danish Pretrain Corpus v1

Combined Danish text corpus for LM pretraining, assembled from:

| Source | Docs (approx) | Description |
|--|--|--|
| `wikipedia` | 250,781 | Danish Wikipedia articles (20231101 snapshot) |
| `dynaword` | 3,921,480 | [danish-foundation-models/danish-dynaword](https://huggingface.co/datasets/danish-foundation-models/danish-dynaword) — 45 curated subsets (parliament, court judgments, EU legal, historical newspapers, literature, forums, etc.) |
| `ia_danish` | 3,161 | Internet Archive Danish PD books (OCR'd `_djvu.txt`) |
| `gutenberg_da_delta` | 19 | Project Gutenberg Danish books not in `dynaword` |
| `fineweb2` | 91,008,012 | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) `dan_Latn` config (web crawl) |

Total: **~95M docs** (~30-45B tokens after dedup).

## Filtering

- fasttext lang-ID (threshold 0.55 for `__label__da`)
- Length >= 200 chars
- Exact-hash dedup across all sources (xxh64) — kept higher-priority source on ties

## Schema

```jsonl
{"text": str, "source": str, "id": str}
```

Files are gzipped JSONL shards.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="jensjepsen/danish-pretrain")
    ap.add_argument("--private", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    args = ap.parse_args()

    shards = sorted(DEDUP_DIR.glob("*.jsonl.gz"))
    if not shards:
        raise SystemExit(f"no shards found in {DEDUP_DIR}")
    total_bytes = sum(p.stat().st_size for p in shards)
    print(f"[push] {len(shards)} shards, {total_bytes/1e9:.1f} GB compressed",
          flush=True)

    if args.dry_run:
        print("[push] dry-run — not uploading")
        return

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN not set")

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])

    print(f"[push] ensuring repo {args.repo} exists", flush=True)
    api.create_repo(args.repo, repo_type="dataset", private=args.private,
                    exist_ok=True)

    # Write README first
    readme_path = DEDUP_DIR / "README.md"
    readme_path.write_text(README_TEMPLATE)

    print(f"[push] uploading large folder to {args.repo} "
          f"(chunked, resumable)", flush=True)
    api.upload_large_folder(
        folder_path=str(DEDUP_DIR),
        repo_id=args.repo,
        repo_type="dataset",
        num_workers=8,
    )
    print("[push] done", flush=True)


if __name__ == "__main__":
    main()
