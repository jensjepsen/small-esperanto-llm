"""Stream + clean mC4 Esperanto.

Pipeline:
  1. Stream `allenai/c4` config `eo`
  2. Drop docs from bad URL domains (Ido wiki, machine-translation mills, etc.)
  3. Split remaining docs into sentences
  4. Drop admin / low-content lines (<N content words or no finite verb)
  5. Run Verifier; keep sentences with zero diagnostics
  6. Rejoin kept sentences and write as JSONL: {"text": doc}

Writes to data/mc4_filtered/mc4_eo.jsonl by default, mirroring the
layout of data/hplt_filtered/ so push_to_hub can pick it up.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn,
)

console = Console()

OUT_DEFAULT = Path("data/mc4_filtered/mc4_eo.jsonl")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZĈĜĤĴŜŬ])")

# Domains to drop outright. Empirical from a 1k-doc sample:
#   io.wikipedia.org   — Ido, not Esperanto (shares roots, confuses verifier)
#   epo.wikitrans.net  — machine translation from English, low quality
#   neciklopedio.org   — parody wiki, mostly memetic junk
#   *.wikipedi0.org    — typo-squatter mirror
BAD_DOMAINS_DEFAULT = (
    "io.wikipedia.org",
    "epo.wikitrans.net",
    "neciklopedio.org",
    "neciklopedio.miraheze.org",
    "eo.wikipedi0.org",
)

# Worker-process globals
_VERIFIER = None
_TOKENIZER = None
_MIN_CONTENT = 5


def _worker_init(min_content: int):
    global _VERIFIER, _TOKENIZER, _MIN_CONTENT
    from esperanto_lm.verify import Verifier
    from esperanto_lm.data import load_tokenizer, _morpheme_preprocess
    _VERIFIER = Verifier()
    _TOKENIZER = load_tokenizer(Path("tokenizer_morpheme"))
    _MIN_CONTENT = min_content
    _TOKENIZER._preprocess = _morpheme_preprocess


def _iter_sents(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for s in SENT_SPLIT.split(line):
            s = s.strip()
            if 15 <= len(s) <= 500:
                out.append(s)
    return out


def _is_contentful(tokens) -> bool:
    n_content = sum(1 for t in tokens if t.pos in {"N", "V", "A", "Adv"})
    if n_content < _MIN_CONTENT:
        return False
    return any(t.pos == "V" for t in tokens)


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER(_TOKENIZER._preprocess(text), add_special_tokens=False)["input_ids"])


@dataclass
class Stats:
    docs_seen: int = 0
    docs_url_dropped: int = 0
    docs_processed: int = 0
    docs_kept: int = 0
    sents_total: int = 0
    sents_low_content: int = 0
    sents_bad_parse: int = 0
    sents_kept: int = 0
    tokens_raw: int = 0
    tokens_kept: int = 0

    def merge(self, o: "Stats") -> None:
        self.docs_seen += o.docs_seen
        self.docs_url_dropped += o.docs_url_dropped
        self.docs_processed += o.docs_processed
        self.docs_kept += o.docs_kept
        self.sents_total += o.sents_total
        self.sents_low_content += o.sents_low_content
        self.sents_bad_parse += o.sents_bad_parse
        self.sents_kept += o.sents_kept
        self.tokens_raw += o.tokens_raw
        self.tokens_kept += o.tokens_kept


def _process_doc(text: str) -> tuple[Stats, str]:
    from esperanto_lm.verify import tokenize
    st = Stats(docs_processed=1)
    sents = _iter_sents(text)
    kept_text = []
    for s in sents:
        st.sents_total += 1
        toks = tokenize(s)
        if not _is_contentful(toks):
            st.sents_low_content += 1
            continue
        diags = _VERIFIER.verify(s)
        if diags:
            st.sents_bad_parse += 1
            continue
        st.sents_kept += 1
        kept_text.append(s)
    if text.strip():
        st.tokens_raw = _count_tokens(text)
    joined = " ".join(kept_text)
    if joined:
        st.tokens_kept = _count_tokens(joined)
        st.docs_kept = 1
    return st, joined


def _batch_process(texts: list[str]) -> tuple[Stats, list[str]]:
    st = Stats()
    docs = []
    for t in texts:
        d_st, d_text = _process_doc(t)
        st.merge(d_st)
        if d_text:
            docs.append(d_text)
    return st, docs


def _chunked(it, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def _host(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def stream_mc4_texts(bad_domains: set[str], limit: int, split: str):
    """Yield text strings from mC4 EO, skipping bad-domain docs.

    Runs in the main process; the generator is passed to pool.imap_unordered
    which handles worker distribution. IterableDataset never leaves this fn.
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "eo", split=split, streaming=True)
    seen = 0
    url_dropped = 0
    for row in ds:
        if limit and seen >= limit:
            break
        seen += 1
        url = row.get("url") or ""
        if _host(url) in bad_domains:
            url_dropped += 1
            # Yield a sentinel so stats can track dropped docs without
            # spending worker time on them.
            yield ("__URL_DROPPED__", url)
            continue
        text = row.get("text") or ""
        if text.strip():
            yield ("TEXT", text)
    # Final marker so consumer knows how many we URL-dropped (optional;
    # stats reconciled in main loop anyway).
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-file", type=Path, default=OUT_DEFAULT,
                    help=f"Write kept docs here as JSONL (default: {OUT_DEFAULT})")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-pool", action="store_true",
                    help="Run entirely in the main process (no multiprocessing). "
                         "Useful for debugging freezes / getting direct tracebacks.")
    ap.add_argument("--chunk-size", type=int, default=40)
    ap.add_argument("--min-content-words", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap total docs streamed (0 = stream all)")
    ap.add_argument("--split", default="train",
                    help="mC4 split to stream (default: train)")
    ap.add_argument("--drop-urls", nargs="*", default=list(BAD_DOMAINS_DEFAULT),
                    help="Domains to drop (whole hostnames)")
    args = ap.parse_args()

    bad_domains = {d.lower() for d in args.drop_urls}
    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Streaming:[/] allenai/c4 (eo, split={args.split})")
    console.print(f"[bold]Drop hosts:[/] {sorted(bad_domains)}")
    console.print(f"[bold]Writing to:[/] {args.out_file}")
    if args.limit:
        console.print(f"[bold]Limit:[/] {args.limit:,} docs")

    # Partition the stream: text payloads go to workers, URL-dropped go
    # straight to stats.
    agg = Stats()

    def text_iter():
        for kind, payload in stream_mc4_texts(bad_domains, args.limit, args.split):
            if kind == "__URL_DROPPED__":
                agg.docs_seen += 1
                agg.docs_url_dropped += 1
                continue
            agg.docs_seen += 1
            yield payload

    t0 = time.time()
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[rate]}[/] docs/s"),
        TextColumn("url✗[red]{task.fields[url_dropped]}[/]"),
        TextColumn("kept [green]{task.fields[kept_docs]}[/]"),
        TextColumn("[dim]{task.fields[toks]}[/] tok"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=2,
    )
    with args.out_file.open("w") as fout, progress:
        total = args.limit if args.limit else None
        task = progress.add_task("[green]mc4-eo", total=total,
                                 rate="0", url_dropped="0", kept_docs="0", toks="0")

        def consume(results):
            for st, docs in results:
                agg.merge(st)
                for d in docs:
                    fout.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
                progress.update(
                    task,
                    completed=agg.docs_seen,
                    rate=f"{agg.docs_seen/max(time.time()-t0,0.01):.0f}",
                    url_dropped=f"{agg.docs_url_dropped:,}",
                    kept_docs=f"{agg.docs_kept:,}",
                    toks=f"{agg.tokens_kept/1e6:.1f}M",
                )

        if args.no_pool:
            _worker_init(args.min_content_words)
            consume(_batch_process(chunk)
                    for chunk in _chunked(text_iter(), args.chunk_size))
        else:
            with Pool(args.workers, initializer=_worker_init,
                      initargs=(args.min_content_words,)) as pool:
                consume(pool.imap_unordered(
                    _batch_process, _chunked(text_iter(), args.chunk_size)))

    # Summary
    console.print(f"\n[bold]── Summary ──[/]")
    console.print(f"  docs seen:          {agg.docs_seen:,}")
    console.print(f"  docs url-dropped:   {agg.docs_url_dropped:,} "
                  f"({100*agg.docs_url_dropped/max(agg.docs_seen,1):.1f}%)")
    console.print(f"  docs processed:     {agg.docs_processed:,}")
    console.print(f"  docs kept (≥1 sent):{agg.docs_kept:,} "
                  f"({100*agg.docs_kept/max(agg.docs_processed,1):.1f}% of processed)")
    console.print(f"  sents total:        {agg.sents_total:,}")
    console.print(f"  sents low-content:  {agg.sents_low_content:,}")
    console.print(f"  sents bad-parse:    {agg.sents_bad_parse:,}")
    console.print(f"  [bold]sents kept:         {agg.sents_kept:,} "
                  f"({100*agg.sents_kept/max(agg.sents_total,1):.1f}%)[/]")
    console.print(f"  tokens raw:         {agg.tokens_raw/1e6:.1f}M")
    console.print(f"  [bold]tokens kept:        {agg.tokens_kept/1e6:.1f}M "
                  f"({100*agg.tokens_kept/max(agg.tokens_raw,1):.1f}%)[/]")
    console.print(f"\nWrote → {args.out_file}")


if __name__ == "__main__":
    main()
