"""Stream YAGO 4.5 and keep only the Danish lines.

The full dump is 12 GB compressed and no local mount has room for it plus the
extraction, so it is never written to disk: the HTTP response is decompressed
on the fly and only lines carrying a Danish literal are kept. That mirrors how
the Esperanto slice was produced (`unzip | grep | filter`), minus the need to
store the archive.

Two outputs:
  <out>/yago_da_labels.tsv   rdfs:label / rdfs:comment / alternateName @da
  <out>/yago_da_subjects.txt distinct subjects that have a Danish label

The facts themselves are NOT kept here -- the Esperanto equivalent ran to
8.9 GB. Subjects are written so a second pass can pull facts for exactly those
entities if wanted.

Usage:
  python scripts/fetch_yago_da.py --out /mnt/data/yago_da
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://yago-knowledge.org/data/yago4.5/yago-4.5.0.2.zip"
UA = "espllm-dataset-builder/0.1 (research)"
DA = re.compile(rb'"@da[\s.]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", type=Path, default=Path("/mnt/data/yago_da"))
    ap.add_argument("--max-bytes", type=int, default=0,
                    help="stop after N compressed bytes (smoke test)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        from stream_unzip import stream_unzip
    except ImportError:
        sys.exit("needs stream-unzip: uv run --with stream-unzip python ...")

    req = urllib.request.Request(args.url, headers={"User-Agent": UA})
    resp = urllib.request.urlopen(req, timeout=120)

    def chunks():
        got = 0
        while True:
            b = resp.read(1 << 20)
            if not b:
                return
            got += len(b)
            if args.max_bytes and got > args.max_bytes:
                return
            yield b

    lab = (args.out / "yago_da_labels.tsv").open("wb")
    subs = set()
    t0, seen, kept, tail = time.time(), 0, 0, b""
    # A 12GB stream can drop mid-flight; keep whatever was already parsed
    # rather than losing the whole pass. --max-bytes also lands here by design.
    try:
      for name, size, data in stream_unzip(chunks()):
        fn = name.decode("utf-8", "replace")
        if not fn.endswith((".ttl", ".nt", ".tsv")):
            for _ in data:
                pass
            continue
        print(f"  reading {fn}", flush=True)
        for block in data:
            seen += len(block)
            buf = tail + block
            *lines, tail = buf.split(b"\n")
            for ln in lines:
                if DA.search(ln):
                    kept += 1
                    lab.write(ln + b"\n")
                    sj = ln.split(b"\t", 1)[0] if b"\t" in ln else ln.split(b" ", 1)[0]
                    subs.add(sj)
            if seen % (512 << 20) < (1 << 20):
                print(f"    {seen/1e9:.1f}GB scanned, {kept:,} da-lines, "
                      f"{len(subs):,} subjects, {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"  stream ended: {type(e).__name__} "
              f"(expected when --max-bytes is set)", flush=True)
    lab.close()
    (args.out / "yago_da_subjects.txt").write_bytes(
        b"\n".join(sorted(subs)) + b"\n")
    print(f"\nscanned {seen/1e9:.1f}GB uncompressed in {time.time()-t0:.0f}s")
    print(f"kept {kept:,} Danish lines over {len(subs):,} subjects")
    print(f"-> {args.out}/yago_da_labels.tsv")


if __name__ == "__main__":
    main()
