"""Extract Danish-labelled entities and their facts from the YAGO 4.5 archive.

Two streamed passes over `yago-facts.ttl` (23 GB uncompressed) via `unzip -p`,
so the member is never written to disk:

  pass 1  every `"..."@da` literal -> subject -> Danish label / description
  pass 2  every fact whose SUBJECT has a Danish label, keeping the object too

The point is gold that does not come from a model. For a Danish Wikipedia
article about entity E, YAGO says which properties E actually has and what
their values are; a property E lacks is verifiably empty rather than
"the annotator missed it". Combined with a verbatim check against the article
text, an extraction target is then both correct and actually present.

Usage:
  python scripts/extract_yago_da_facts.py --pass 1 --out /mnt/data2/yago4.5/da
  python scripts/extract_yago_da_facts.py --pass 2 --out /mnt/data2/yago4.5/da
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ZIP = "/mnt/data2/yago4.5/full/yago-4.5.0.2.zip"
MEMBER = "yago-facts.ttl"
DA_LIT = re.compile(rb'"((?:[^"\\]|\\.)*)"@da')


def stream(zip_path, member):
    p = subprocess.Popen(["unzip", "-p", zip_path, member],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         bufsize=1 << 22)
    tail = b""
    while True:
        block = p.stdout.read(1 << 22)
        if not block:
            break
        buf = tail + block
        *lines, tail = buf.split(b"\n")
        yield lines, len(block)
    p.stdout.close()
    p.wait()


def pass1(args):
    """subject -> {label, comment} for every entity with a Danish literal."""
    labels = {}
    t0, seen, n = time.time(), 0, 0
    for lines, nb in stream(args.zip, args.member):
        seen += nb
        for ln in lines:
            m = DA_LIT.search(ln)
            if not m:
                continue
            parts = ln.split(b"\t") if b"\t" in ln else ln.split(b" ", 2)
            if len(parts) < 2:
                continue
            subj, pred = parts[0].strip(), parts[1].strip()
            val = m.group(1).decode("utf-8", "replace")
            d = labels.setdefault(subj.decode("utf-8", "replace"), {})
            if pred.endswith(b"label") and "label" not in d:
                d["label"] = val
            elif pred.endswith(b"comment") and "comment" not in d:
                d["comment"] = val
            n += 1
        if seen % (2 << 30) < (1 << 22):
            print(f"  {seen/1e9:.1f}GB, {n:,} da-literals, "
                  f"{len(labels):,} entities, {time.time()-t0:.0f}s", flush=True)
    out = args.out / "da_entities.json"
    out.write_text(json.dumps(labels, ensure_ascii=False))
    print(f"\npass1: {len(labels):,} Danish-labelled entities "
          f"from {seen/1e9:.1f}GB in {time.time()-t0:.0f}s\n-> {out}")


def pass2(args):
    """facts whose subject is Danish-labelled."""
    ents = json.loads((args.out / "da_entities.json").read_text())
    keep = set(ents)
    print(f"pass2: filtering facts to {len(keep):,} entities", flush=True)
    fh = (args.out / "da_facts.tsv").open("wb")
    t0, seen, kept = time.time(), 0, 0
    for lines, nb in stream(args.zip, args.member):
        seen += nb
        for ln in lines:
            parts = ln.split(b"\t") if b"\t" in ln else ln.split(b" ", 2)
            if len(parts) < 3:
                continue
            if parts[0].strip().decode("utf-8", "replace") in keep:
                fh.write(ln + b"\n")
                kept += 1
        if seen % (2 << 30) < (1 << 22):
            print(f"  {seen/1e9:.1f}GB, {kept:,} facts kept, "
                  f"{time.time()-t0:.0f}s", flush=True)
    fh.close()
    print(f"\npass2: {kept:,} facts in {time.time()-t0:.0f}s"
          f"\n-> {args.out}/da_facts.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="phase", type=int, choices=[1, 2], required=True)
    ap.add_argument("--zip", default=ZIP)
    ap.add_argument("--member", default=MEMBER)
    ap.add_argument("--out", type=Path, default=Path("/mnt/data2/yago4.5/da"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (pass1 if args.phase == 1 else pass2)(args)


if __name__ == "__main__":
    main()
