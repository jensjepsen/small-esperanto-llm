"""Download text for all books across sci/econ/lit deduped JSONLs.

Reads /mnt/data2/sci_books/{sci,econ,lit}_deduped.jsonl, deduplicates by
gutenberg_id (some authors appear across profiles, e.g. Spencer in sci+econ,
Mill in sci+econ), skips books already downloaded, fetches the rest.

Strips Project Gutenberg header/footer boilerplate.

Output:
  - /mnt/data2/sci_books/<gid>.txt        (one per book)
  - /mnt/data2/sci_books/curated_manifest.jsonl   (metadata + path)
"""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

OUT_DIR = Path("/mnt/data2/sci_books")
MANIFEST = OUT_DIR / "curated_manifest.jsonl"
USER_AGENT = "espllm-pretrain-data/1.0 (jens.jepsen@gmail.com)"


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def fetch_book(gid: int) -> str | None:
    urls = [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    ]
    raw = None
    for url in urls:
        try:
            raw = http_get(url).decode("utf-8", errors="replace")
            break
        except Exception:
            continue
    if raw is None:
        return None
    start_re = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                          re.IGNORECASE)
    end_re = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                        re.IGNORECASE)
    m_start = start_re.search(raw)
    m_end = end_re.search(raw)
    if m_start:
        raw = raw[m_start.end():]
    if m_end:
        raw = raw[:m_end.start()]
    return raw.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+",
                    default=["sci", "econ", "lit"],
                    help="which deduped profile jsonls to read")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many new downloads (0 = no cap)")
    ap.add_argument("--rate", type=float, default=0.3,
                    help="seconds between requests (polite Gutenberg)")
    args = ap.parse_args()

    # 1. Aggregate books from all profiles, dedup by gid
    by_gid: dict[int, dict] = {}
    for profile in args.profiles:
        path = OUT_DIR / f"{profile}_deduped.jsonl"
        if not path.exists():
            print(f"  skip {profile}: {path} missing")
            continue
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                gid = r["gutenberg_id"]
                if gid in by_gid:
                    by_gid[gid]["profiles"].add(profile)
                else:
                    r["profiles"] = {profile}
                    by_gid[gid] = r
    print(f"unique books across profiles: {len(by_gid):,}")

    # 2. Skip books already downloaded
    todo = []
    for gid, r in sorted(by_gid.items()):
        out_path = OUT_DIR / f"{gid}.txt"
        if out_path.exists() and out_path.stat().st_size > 5000:
            continue
        todo.append(r)
    print(f"already downloaded: {len(by_gid) - len(todo):,}")
    print(f"to download:        {len(todo):,}")

    if args.limit:
        todo = todo[: args.limit]
        print(f"limited to: {len(todo):,}")

    # 3. Open manifest in append mode (skip dupes by gid)
    have_in_manifest = set()
    if MANIFEST.exists():
        with MANIFEST.open() as f:
            for line in f:
                try:
                    have_in_manifest.add(json.loads(line)["gutenberg_id"])
                except Exception:
                    pass

    n_ok = n_fail = 0
    with MANIFEST.open("a") as mf:
        for i, r in enumerate(todo, 1):
            gid = r["gutenberg_id"]
            title = r["title"]
            text = fetch_book(gid)
            time.sleep(args.rate)
            if not text or len(text) < 5000:
                print(f"[{i:4d}/{len(todo)}] {title[:50]:50s} gid={gid:>6} FAILED")
                n_fail += 1
                continue
            out_path = OUT_DIR / f"{gid}.txt"
            out_path.write_text(text, encoding="utf-8")
            if gid not in have_in_manifest:
                mf.write(json.dumps({
                    "gutenberg_id": gid,
                    "title": title,
                    "authors": r.get("authors", ""),
                    "trusted_author": r.get("trusted_author", ""),
                    "profiles": sorted(r["profiles"]),
                    "length": len(text),
                    "path": str(out_path),
                }, ensure_ascii=False) + "\n")
                mf.flush()
            n_ok += 1
            if i % 25 == 0:
                print(f"[{i:4d}/{len(todo)}] ok={n_ok} fail={n_fail} "
                      f"({title[:40]} → {len(text)/1e3:.0f}k chars)")

    print(f"\n=== done: {n_ok} ok, {n_fail} failed ===")


if __name__ == "__main__":
    main()
