"""Download en↔eo parallel data from HF, write JSONL with {en, eo}."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.split())


def is_valid_pair(en: str, eo: str, min_len: int, max_len: int) -> bool:
    en, eo = en.strip(), eo.strip()
    if not en or not eo:
        return False
    if len(en) < min_len or len(eo) < min_len:
        return False
    if len(en) > max_len or len(eo) > max_len:
        return False
    en_words, eo_words = en.split(), eo.split()
    if not en_words or not eo_words:
        return False
    ratio = len(en_words) / len(eo_words)
    if ratio < 0.4 or ratio > 2.5:
        return False
    return True


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fetch_opus_100(out_dir: Path, min_len: int, max_len: int) -> None:
    """Pull Helsinki-NLP/opus-100 en-eo (train/validation/test).

    WARNING: opus-100 is an aggregated multilingual blob that bundles
    KDE/GNOME/Ubuntu .po localization pairs without src labels. In v13
    the train split is NO LONGER included in build_mt_dataset.py — using
    it directly for training causes the model to memorize UI-label junk
    (see project_mt_opus_ui_contamination memory + build_mt_dataset.py
    module docstring). The validation split is still used as an eval
    set by train.py (mt/data/parallel/opus100_validation.jsonl).
    """
    from datasets import load_dataset
    ds = load_dataset("Helsinki-NLP/opus-100", "en-eo")
    for split in ("train", "validation", "test"):
        rows = []
        kept, dropped = 0, 0
        for r in ds[split]:
            en = normalize(r["translation"]["en"])
            eo = normalize(r["translation"]["eo"])
            if not is_valid_pair(en, eo, min_len, max_len):
                dropped += 1
                continue
            rows.append({"en": en, "eo": eo, "src": "opus-100"})
            kept += 1
        out = out_dir / f"opus100_{split}.jsonl"
        write_jsonl(out, rows)
        print(f"  opus-100 {split:10s} kept={kept} dropped={dropped} -> {out}")


def fetch_opus_books(out_dir: Path, min_len: int, max_len: int) -> None:
    from datasets import load_dataset
    ds = load_dataset("Helsinki-NLP/opus_books", "en-eo")
    rows = []
    kept, dropped = 0, 0
    for r in ds["train"]:
        en = normalize(r["translation"]["en"])
        eo = normalize(r["translation"]["eo"])
        if not is_valid_pair(en, eo, min_len, max_len):
            dropped += 1
            continue
        rows.append({"en": en, "eo": eo, "src": "opus-books"})
        kept += 1
    out = out_dir / "opusbooks_train.jsonl"
    write_jsonl(out, rows)
    print(f"  opus-books train      kept={kept} dropped={dropped} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("mt/data/parallel"))
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--hf-cache", type=str, default="/mnt/data/hf_cache")
    args = ap.parse_args()

    os.environ["HF_HOME"] = args.hf_cache

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.out_dir}")
    print(f"HF cache:   {args.hf_cache}")

    fetch_opus_100(args.out_dir, args.min_len, args.max_len)
    fetch_opus_books(args.out_dir, args.min_len, args.max_len)


if __name__ == "__main__":
    main()
