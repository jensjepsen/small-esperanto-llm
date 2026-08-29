"""Fetch all Esperanto Wikisource (Vikifontaro) mainspace articles.

Uses MediaWiki extracts API with `explaintext=1` for plain-text output
(no wiki markup). Batches up to 20 titles per request. Iterates via
`list=allpages` to enumerate ~13.5k mainspace articles.

Output JSONL rows:
    {"title": str, "pageid": int, "text": str}

Rows with empty extract text (redirects, disambigs) are skipped.

Usage:
    uv run python scripts/fetch_eo_wikisource.py \\
        --out /mnt/data2/eo_wikisource.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://eo.wikisource.org/w/api.php"
UA = "espllm-pretrain-data-gathering/1.0 (jens.jepsen@gmail.com)"


def api_get(params: dict, retries: int = 3) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 ** attempt)
    return {}


def enumerate_mainspace_titles() -> list[str]:
    """List all mainspace pages via list=allpages."""
    titles = []
    apcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": "0",
            "aplimit": "500",
            "apfilterredir": "nonredirects",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        data = api_get(params)
        for p in data.get("query", {}).get("allpages", []):
            titles.append(p["title"])
        cont = data.get("continue", {})
        apcontinue = cont.get("apcontinue")
        if not apcontinue:
            break
        if len(titles) % 5000 < 500 and len(titles) > 0:
            print(f"  ...{len(titles):,} titles enumerated", flush=True)
        time.sleep(0.05)
    return titles


def fetch_batch(titles: list[str]) -> list[dict]:
    """Return {title, pageid, text} for titles with non-empty extracts."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exlimit": "max",
        "redirects": 1,
        "titles": "|".join(titles),
    }
    data = api_get(params)
    out = []
    orig_by_norm = {t: t for t in titles}
    for n in data.get("query", {}).get("normalized", []) or []:
        orig_by_norm[n["to"]] = n["from"]
    for r in data.get("query", {}).get("redirects", []) or []:
        orig = orig_by_norm.get(r["from"], r["from"])
        orig_by_norm[r["to"]] = orig

    for page in data.get("query", {}).get("pages", []):
        if page.get("missing"):
            continue
        title = page.get("title", "")
        requested = orig_by_norm.get(title, title)
        text = (page.get("extract") or "").strip()
        if len(text) < 20:
            continue
        out.append({
            "title": requested,
            "resolved_title": title,
            "pageid": page.get("pageid"),
            "text": text,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/data2/eo_wikisource.jsonl"))
    ap.add_argument("--batch", type=int, default=1,
                    help="Must be 1 — MediaWiki caps whole-article extracts "
                         "at 1 title/request when explaintext=1 (silently drops "
                         "extras). Batch >1 loses 95%+ of the content.")
    ap.add_argument("--pause", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

    print("enumerating mainspace titles...", flush=True)
    titles = enumerate_mainspace_titles()
    print(f"total titles: {len(titles):,}\n", flush=True)

    done: set[str] = set()
    if args.out.exists():
        with args.out.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["title"])
                except Exception:
                    pass
        print(f"resume: {len(done):,} already fetched", flush=True)

    todo = [t for t in titles if t not in done]
    print(f"todo: {len(todo):,}\n", flush=True)
    if not todo:
        return

    t0 = time.time()
    n_ok = 0
    n_missing = 0
    with args.out.open("a") as fout:
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start + args.batch]
            try:
                results = fetch_batch(batch)
            except Exception as e:
                print(f"[error batch {start}] {e}", file=sys.stderr, flush=True)
                results = []
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            n_ok += len(results)
            n_missing += len(batch) - len(results)
            if args.pause > 0:
                time.sleep(args.pause)
            processed = start + len(batch)
            if processed % args.log_every == 0 or processed >= len(todo):
                el = time.time() - t0
                rate = processed / el
                eta = (len(todo) - processed) / rate if rate else 0
                print(f"  {processed:,}/{len(todo):,}  ok={n_ok:,}  "
                      f"empty/skip={n_missing:,} ({100*n_missing/processed:.1f}%)  "
                      f"rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)

    print(f"\ndone: {n_ok:,} articles, {n_missing:,} skipped in "
          f"{(time.time()-t0)/60:.1f} min → {args.out}")


if __name__ == "__main__":
    main()
