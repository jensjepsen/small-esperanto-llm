"""Extract Esperanto-language Reddit content from Pushshift HF mirror.

Uses DuckDB to query parquets directly over HTTP (no full download).
Filters comments and submissions where `subreddit` matches r/Esperanto
or related EO subreddits. Adds `lang` tag per row via fasttext lid.176
so downstream filtering (keep only EO content vs English meta-discussion)
is trivial.

Targets the Pushshift HF mirrors:
- fddemarco/pushshift-reddit (submissions, RS_YYYY-MM_NN.parquet)
- fddemarco/pushshift-reddit-comments (comments, RC_YYYY-MM.parquet)

Output JSONL rows include a per-row `lang` (top fasttext prediction)
and `lang_score`. Run with `--filter-lang eo` to drop non-EO rows
inline; default keeps everything tagged.

Run: uv run --with duckdb --with langid python scripts/extract_reddit_eo.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import langid
from huggingface_hub import HfApi


# Subreddits to harvest. Case typically preserved by Pushshift, but
# we lowercase-compare to be safe. Listed in expected-volume order.
TARGET_SUBREDDITS = (
    "Esperanto",          # main subreddit
    "learnesperanto",     # learners
    "esperanto",          # lowercase variant if any
    "Esperanto_humour",   # niche
    "Esperantio",         # rare
)


def list_parquets(repo: str) -> list[str]:
    """List all parquet URLs in an HF dataset's data/ dir."""
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return [
        f"https://huggingface.co/datasets/{repo}/resolve/main/{f}"
        for f in sorted(files) if f.startswith("data/") and f.endswith(".parquet")
    ]


def setup_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    # Authenticate to HF for higher rate limits
    from huggingface_hub import HfFolder
    tok = HfFolder.get_token()
    if tok:
        con.execute(f"CREATE SECRET hf (TYPE HUGGINGFACE, TOKEN '{tok}');")
    return con


def load_lang_id():
    # Pre-warm langid (loads internal model once)
    langid.classify("warmup text")
    return None  # langid is stateless module-level


def detect_lang(model, text: str) -> tuple[str, float]:
    """Return (lang, score) — langid returns ISO-639-1 + a log-prob."""
    if not text or not text.strip():
        return "und", 0.0
    txt = " ".join(text.split())[:2000]
    if len(txt) < 10:
        return "und", 0.0
    lang, score = langid.classify(txt)
    return lang, float(score)


def query_parquet(con, url: str, table_kind: str,
                  max_retries: int = 5) -> list[dict]:
    """Run a filtered SELECT against one parquet URL with HTTP 429 backoff."""
    subs_sql = ", ".join(f"'{s.lower()}'" for s in TARGET_SUBREDDITS)
    if table_kind == "comments":
        want = ["id", "subreddit", "body", "author", "created_utc",
                "score", "parent_id", "link_id"]
    else:
        want = ["id", "subreddit", "title", "selftext", "author",
                "created_utc", "score", "num_comments"]
    for attempt in range(max_retries):
        try:
            cols_df = con.execute(
                f"SELECT * FROM read_parquet('{url}') LIMIT 0"
            ).df()
            avail = set(cols_df.columns)
            sel = ", ".join(c for c in want if c in avail)
            sql = (
                f"SELECT {sel} FROM read_parquet('{url}') "
                f"WHERE LOWER(subreddit) IN ({subs_sql})"
            )
            return con.execute(sql).df().to_dict(orient="records")
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                print(f"    rate-limited, retry in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise


def extract(repo: str, table_kind: str, out_path: Path, limit: int | None,
            lang_model, filter_lang: str | None):
    urls = list_parquets(repo)
    if limit:
        urls = urls[:limit]
    print(f"[{table_kind}] {repo}: {len(urls)} parquets", flush=True)

    con = setup_duckdb()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_kept = 0
    lang_counts: dict[str, int] = {}
    t0 = time.time()
    with open(out_path, "w") as f:
        for i, url in enumerate(urls, 1):
            shard_name = url.split("/")[-1]
            try:
                rows = query_parquet(con, url, table_kind)
            except Exception as e:
                print(f"  [{i}/{len(urls)}] {shard_name}: ERR "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            for r in rows:
                # Convert non-JSON types
                for k, v in r.items():
                    if hasattr(v, "isoformat"):
                        r[k] = v.isoformat()
                # Language detect on the main text field
                if table_kind == "comments":
                    text = r.get("body", "") or ""
                else:
                    text = " ".join(
                        (r.get("title") or "", r.get("selftext") or "")
                    )
                lang, score = detect_lang(lang_model, text)
                r["lang"] = lang
                r["lang_score"] = round(score, 3)
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                if filter_lang and lang != filter_lang:
                    continue
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                n_kept += 1
            n_total += len(rows)
            elap = time.time() - t0
            rate = i / elap if elap > 0 else 0
            eta = (len(urls) - i) / rate if rate > 0 else 0
            top_langs = sorted(lang_counts.items(),
                               key=lambda x: -x[1])[:3]
            print(f"  [{i:>3}/{len(urls)}] {shard_name}: +{len(rows):>5} "
                  f"(kept={n_kept:,}/{n_total:,}, eta={eta:.0f}s, "
                  f"top: {top_langs})", flush=True)
    print(f"[{table_kind}] wrote {n_kept:,}/{n_total:,} rows -> {out_path} "
          f"in {time.time()-t0:.0f}s")
    print(f"  lang distribution: "
          f"{sorted(lang_counts.items(), key=lambda x: -x[1])[:8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/reddit_eo")
    ap.add_argument("--kind", choices=["comments", "submissions", "both"],
                    default="both")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of parquet shards (for testing)")
    ap.add_argument(
        "--filter-lang", default=None,
        help="If set (e.g. 'eo'), drop rows whose top language != this. "
             "Default: keep all rows but tag each with detected lang.")
    args = ap.parse_args()

    lang_model = load_lang_id()
    out_dir = Path(args.out_dir)
    if args.kind in ("submissions", "both"):
        extract("fddemarco/pushshift-reddit", "submissions",
                out_dir / "submissions.jsonl", args.limit,
                lang_model, args.filter_lang)
    if args.kind in ("comments", "both"):
        extract("fddemarco/pushshift-reddit-comments", "comments",
                out_dir / "comments.jsonl", args.limit,
                lang_model, args.filter_lang)


if __name__ == "__main__":
    main()
