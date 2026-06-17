"""Audit mC4 EO verifier keep rates broken down by source hostname.

Streams a sample of mC4 EO docs, applies the same sentence-level
verifier filter as clean_mc4_eo.py (without URL-dropping, so we can see
per-source pass rates including the "bad" hosts), and reports:

  host              docs  sents  keep%  tok_raw  tok_kept  tok_keep%
  <hostname>        N     N       N%      N        N         N%
  ...
  TOTAL             ...

Usage:
    uv run python scripts/mc4_source_audit.py --limit 2000 --workers 12
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn,
)

console = Console()

# Reuse the worker / filter internals
from clean_mc4_eo import (  # type: ignore
    _worker_init, _iter_sents, _is_contentful, _count_tokens, Stats,
)
from clean_mc4_eo import _process_doc  # uses worker globals


@dataclass
class HostStats:
    docs: int = 0
    docs_kept: int = 0
    sents_total: int = 0
    sents_kept: int = 0
    tokens_raw: int = 0
    tokens_kept: int = 0

    def merge_from(self, st: Stats) -> None:
        self.docs += 1
        self.docs_kept += st.docs_kept
        self.sents_total += st.sents_total
        self.sents_kept += st.sents_kept
        self.tokens_raw += st.tokens_raw
        self.tokens_kept += st.tokens_kept


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _worker_process(item: tuple[str, str]) -> tuple[str, Stats]:
    """item = (host, text); returns (host, stats for this doc)."""
    host, text = item
    st, _joined = _process_doc(text)
    return host, st


def stream_items(limit: int, split: str):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "eo", split=split, streaming=True)
    n = 0
    for row in ds:
        if limit and n >= limit:
            break
        text = row.get("text") or ""
        if not text.strip():
            continue
        host = _host(row.get("url") or "")
        yield host, text
        n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--min-content-words", type=int, default=5)
    ap.add_argument("--split", default="train")
    ap.add_argument("--top", type=int, default=30,
                    help="Show this many top hosts by doc count")
    args = ap.parse_args()

    by_host: dict[str, HostStats] = defaultdict(HostStats)
    t0 = time.time()

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[rate]}[/] docs/s"),
        TextColumn("hosts [magenta]{task.fields[nhost]}[/]"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=2,
    )

    console.print(f"[bold]Streaming {args.limit:,} mC4-EO docs, "
                  f"tracking per-host keep rates...[/]")

    with progress, Pool(args.workers, initializer=_worker_init,
                        initargs=(args.min_content_words,)) as pool:
        task = progress.add_task("[green]audit", total=args.limit,
                                 rate="0", nhost="0")
        seen = 0
        for host, st in pool.imap_unordered(
                _worker_process, stream_items(args.limit, args.split),
                chunksize=args.chunk_size):
            by_host[host].merge_from(st)
            seen += 1
            if seen % 50 == 0 or seen == args.limit:
                progress.update(
                    task,
                    completed=seen,
                    rate=f"{seen/max(time.time()-t0,0.01):.0f}",
                    nhost=f"{len(by_host)}",
                )

    # Sort hosts by doc count desc; compute totals
    ranked = sorted(by_host.items(), key=lambda kv: kv[1].docs, reverse=True)

    console.print(f"\n[bold]── Per-source keep rates (top {args.top}) ──[/]")
    header = (f"{'host':<32} {'docs':>5} {'sents':>6} {'keep%':>6} "
              f"{'tok_raw':>8} {'tok_kept':>8} {'keep%':>6}")
    console.print(header)
    console.print("─" * len(header))

    def fmt(st: HostStats) -> str:
        ks = 100 * st.sents_kept / max(st.sents_total, 1)
        kt = 100 * st.tokens_kept / max(st.tokens_raw, 1)
        return (f"{st.docs:>5} {st.sents_total:>6} {ks:>5.1f}% "
                f"{st.tokens_raw:>8,} {st.tokens_kept:>8,} {kt:>5.1f}%")

    for host, st in ranked[:args.top]:
        name = host or "(no-host)"
        if len(name) > 31:
            name = name[:28] + "..."
        console.print(f"{name:<32} {fmt(st)}")

    # Total
    tot = HostStats()
    for st in by_host.values():
        tot.docs += st.docs
        tot.docs_kept += st.docs_kept
        tot.sents_total += st.sents_total
        tot.sents_kept += st.sents_kept
        tot.tokens_raw += st.tokens_raw
        tot.tokens_kept += st.tokens_kept
    console.print("─" * len(header))
    console.print(f"{'TOTAL':<32} {fmt(tot)}")
    console.print(f"\n[bold]unique hosts:[/] {len(by_host):,}")
    console.print(f"[bold]docs with zero kept sents:[/] "
                  f"{sum(1 for s in by_host.values() if s.docs_kept == 0 and s.docs > 0)}")


if __name__ == "__main__":
    main()
