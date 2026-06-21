"""Convert HuggingFaceFW/finetranslations epo_Latn parquets to MT JSONL.

FineTranslations is the LLM-translated parallel companion to FineWeb-2:
each EO row has its English translation under `translated_text`. Source
EO content is FineWeb-quality (filtered CommonCrawl).

Produces mt/data/parallel/finetranslations.jsonl with rows {en, eo, src}.
Source parquets expected at /mnt/data2/datasets/finetranslations/*.parquet
(download via `curl` from the HF dataset).
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir",
                    default="/mnt/data2/datasets/finetranslations")
    ap.add_argument("--out",
                    default="/mnt/data2/datasets/finetranslations/finetranslations.jsonl",
                    help="Output JSONL (default /mnt/data2 — large file)")
    ap.add_argument("--min-tokens", type=int, default=4,
                    help="Drop pairs where either side has fewer tokens "
                         "than this (whitespace tokenization).")
    args = ap.parse_args()

    pqs = sorted(Path(args.parquet_dir).glob("*.parquet"))
    if not pqs:
        raise SystemExit(f"no parquets in {args.parquet_dir}")
    print(f"reading {len(pqs)} parquets", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with open(out_path, "w") as f:
        for p in pqs:
            pf = pq.ParquetFile(p)
            print(f"  {p.name}: {pf.metadata.num_rows:,} rows", flush=True)
            for batch in pf.iter_batches(
                batch_size=10_000,
                columns=["og_full_text", "translated_text"],
            ):
                rows = batch.to_pylist()
                for r in rows:
                    n_in += 1
                    eo = (r.get("og_full_text") or "").strip()
                    en = (r.get("translated_text") or "").strip()
                    if not eo or not en:
                        continue
                    if len(eo.split()) < args.min_tokens or \
                       len(en.split()) < args.min_tokens:
                        continue
                    f.write(json.dumps(
                        {"en": en, "eo": eo, "src": "finetranslations"},
                        ensure_ascii=False) + "\n")
                    n_out += 1
            print(f"    running: in={n_in:,} out={n_out:,}", flush=True)
    print(f"\nwrote {n_out:,}/{n_in:,} pairs -> {out_path}")


if __name__ == "__main__":
    main()
