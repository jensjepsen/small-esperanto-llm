"""Generic downloader for OPUS Moses-format en-eo corpora.

Pulls https://object.pouta.csc.fi/OPUS-<CORPUS>/<VERSION>/moses/en-eo.txt.zip,
parses the aligned .en / .eo line files, applies quality filters and dedup,
writes JSONL.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

HTML_TAG = re.compile(r"<[^>]+>")


def normalize(text: str) -> str:
    return " ".join(text.split())


def quality_ok(en: str, eo: str, min_chars: int, max_chars: int,
               min_words: int, ratio_lo: float, ratio_hi: float) -> str | None:
    en, eo = en.strip(), eo.strip()
    if not en or not eo:
        return "empty"
    if len(en) < min_chars or len(eo) < min_chars:
        return "short_chars"
    if len(en) > max_chars or len(eo) > max_chars:
        return "long_chars"
    en_w, eo_w = en.split(), eo.split()
    if len(en_w) < min_words or len(eo_w) < min_words:
        return "short_words"
    ratio = len(en_w) / len(eo_w)
    if ratio < ratio_lo or ratio > ratio_hi:
        return "ratio"
    if en.lower() == eo.lower():
        return "copy"
    if HTML_TAG.search(en) or HTML_TAG.search(eo):
        return "html"
    return None


def fetch(corpus: str, version: str, out_path: Path, src_label: str,
          min_chars: int, max_chars: int, min_words: int,
          ratio_lo: float, ratio_hi: float) -> None:
    url = f"https://object.pouta.csc.fi/OPUS-{corpus}/{version}/moses/en-eo.txt.zip"
    print(f"  fetching {url}")
    with urllib.request.urlopen(url) as resp:
        zip_bytes = resp.read()
    print(f"  zip size = {len(zip_bytes):,} bytes")

    en_lines: list[str] = []
    eo_lines: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # OPUS Moses zips contain {prefix}.en and {prefix}.eo
        en_name = next(n for n in zf.namelist() if n.endswith(".en"))
        eo_name = next(n for n in zf.namelist() if n.endswith(".eo"))
        en_lines = zf.read(en_name).decode("utf-8", errors="replace").splitlines()
        eo_lines = zf.read(eo_name).decode("utf-8", errors="replace").splitlines()

    assert len(en_lines) == len(eo_lines), f"line count mismatch: {len(en_lines)} en vs {len(eo_lines)} eo"
    print(f"  raw aligned lines: {len(en_lines):,}")

    seen: set[str] = set()
    drops: dict[str, int] = {}
    kept = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for en, eo in zip(en_lines, eo_lines):
            en, eo = normalize(en), normalize(eo)
            reason = quality_ok(en, eo, min_chars, max_chars, min_words, ratio_lo, ratio_hi)
            if reason is not None:
                drops[reason] = drops.get(reason, 0) + 1
                continue
            key = en + " ||| " + eo
            if key in seen:
                drops["dup"] = drops.get("dup", 0) + 1
                continue
            seen.add(key)
            f.write(json.dumps({"en": en, "eo": eo, "src": src_label}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"  kept={kept:,}  drops={drops}  -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="OPUS corpus name, e.g. WikiMatrix")
    ap.add_argument("--version", required=True, help="OPUS version, e.g. v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--src-label", default=None,
                    help="Label written under 'src' field (defaults to corpus name lowercased)")
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--ratio-lo", type=float, default=0.4)
    ap.add_argument("--ratio-hi", type=float, default=2.5)
    args = ap.parse_args()

    label = args.src_label or args.corpus.lower()
    print(f"=== {args.corpus} {args.version} -> {args.out}")
    fetch(args.corpus, args.version, args.out, label,
          args.min_chars, args.max_chars, args.min_words,
          args.ratio_lo, args.ratio_hi)


if __name__ == "__main__":
    main()
