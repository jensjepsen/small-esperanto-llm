"""End-to-end quality filter over all HPLT buckets.

For each doc:
  1. Split into sentences
  2. Drop admin / low-content lines (<5 content words or no finite verb)
  3. Run Verifier; keep sentences with zero diagnostics
  4. Rejoin kept sentences into a filtered doc

Also counts morpheme-tokenizer tokens before and after filtering, so we
can see the real training-token yield per bucket.

Multiprocessed across docs. Results per bucket + overall.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

HPLT_DIR = Path("data/hplt")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZĈĜĤĴŜŬ])")

# Globals for worker processes — each worker lazily loads these once.
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
    # Stash the preprocess fn on the tokenizer obj so we can reach it in process_doc
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
    docs: int = 0
    sents_total: int = 0
    sents_low_content: int = 0
    sents_bad_parse: int = 0
    sents_kept: int = 0
    tokens_raw: int = 0
    tokens_kept: int = 0

    def merge(self, o: "Stats") -> None:
        self.docs += o.docs
        self.sents_total += o.sents_total
        self.sents_low_content += o.sents_low_content
        self.sents_bad_parse += o.sents_bad_parse
        self.sents_kept += o.sents_kept
        self.tokens_raw += o.tokens_raw
        self.tokens_kept += o.tokens_kept


def _process_doc(text: str) -> tuple[Stats, str]:
    from esperanto_lm.verify import tokenize
    st = Stats(docs=1)
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
    # Tokens: raw = full doc, kept = only kept sentences
    if text.strip():
        st.tokens_raw = _count_tokens(text)
    joined = " ".join(kept_text)
    if joined:
        st.tokens_kept = _count_tokens(joined)
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


def process_bucket(path: Path, n_workers: int, chunk_size: int,
                   min_content: int, limit: int = 0,
                   out_dir: Path | None = None) -> Stats:
    def iter_texts():
        with path.open() as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                try:
                    yield json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue

    # Count docs for the progress bar total (cheap linear scan).
    with path.open() as f:
        total_docs = sum(1 for _ in f)
    if limit:
        total_docs = min(total_docs, limit)

    out_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / path.name

    agg = Stats()
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[rate]}[/] docs/s"),
        TextColumn("kept [green]{task.fields[kept]}[/]"),
        TextColumn("[dim]{task.fields[toks]}[/] tok"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    )
    t0 = time.time()
    fout = out_path.open("w") if out_path else None
    try:
        with progress, Pool(n_workers, initializer=_worker_init,
                            initargs=(min_content,)) as pool:
            task = progress.add_task(f"[green]{path.name}", total=total_docs,
                                     rate="0", kept="0", toks="0")
            for st, docs in pool.imap_unordered(_batch_process,
                                                _chunked(iter_texts(), chunk_size)):
                agg.merge(st)
                if fout is not None:
                    for d in docs:
                        fout.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
                progress.update(
                    task,
                    completed=agg.docs,
                    rate=f"{agg.docs/max(time.time()-t0,0.01):.0f}",
                    kept=f"{agg.sents_kept:,}",
                    toks=f"{agg.tokens_kept/1e6:.1f}M",
                )
    finally:
        if fout is not None:
            fout.close()
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=HPLT_DIR)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk-size", type=int, default=40)
    ap.add_argument("--min-content-words", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap docs per bucket (0 = no cap)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/hplt_filtered"),
                    help="Write kept-sentence docs as JSONL here, mirroring "
                         "the input filenames. Pass empty string to skip writing.")
    args = ap.parse_args()

    buckets = sorted(args.data_dir.glob("*.jsonl"))
    console.print(f"[bold]Buckets:[/] {[p.name for p in buckets]}")

    out_dir = args.out_dir if str(args.out_dir) else None
    if out_dir is not None:
        console.print(f"[bold]Writing filtered docs to:[/] {out_dir}")

    totals = Stats()
    by_bucket: dict[str, Stats] = {}

    for p in buckets:
        console.print(f"\n[bold green]── {p.name} ──[/]")
        st = process_bucket(p, args.workers, args.chunk_size,
                            args.min_content_words, args.limit, out_dir=out_dir)
        by_bucket[p.name] = st
        totals.merge(st)
        keep_tok = st.tokens_kept / max(st.tokens_raw, 1)
        keep_sent = st.sents_kept / max(st.sents_total, 1)
        console.print(f"  docs={st.docs:,}  sents={st.sents_total:,}  "
                      f"kept={st.sents_kept:,} ({100*keep_sent:.1f}%)  "
                      f"tokens_raw={st.tokens_raw/1e6:.1f}M  "
                      f"tokens_kept={st.tokens_kept/1e6:.1f}M "
                      f"({100*keep_tok:.1f}%)")

    console.print(f"\n[bold]── Summary ──[/]")
    header = f"{'bucket':<12}  {'docs':>7}  {'sents':>9}  {'keep%':>6}  {'tok_raw':>8}  {'tok_kept':>9}  {'keep%':>6}"
    console.print(header)
    for name, st in by_bucket.items():
        ks = st.sents_kept / max(st.sents_total, 1) * 100
        kt = st.tokens_kept / max(st.tokens_raw, 1) * 100
        console.print(f"{name:<12}  {st.docs:>7,}  {st.sents_total:>9,}  "
                      f"{ks:>5.1f}%  {st.tokens_raw/1e6:>7.1f}M  "
                      f"{st.tokens_kept/1e6:>8.1f}M  {kt:>5.1f}%")
    ks = totals.sents_kept / max(totals.sents_total, 1) * 100
    kt = totals.tokens_kept / max(totals.tokens_raw, 1) * 100
    console.print(f"{'TOTAL':<12}  {totals.docs:>7,}  {totals.sents_total:>9,}  "
                  f"{ks:>5.1f}%  {totals.tokens_raw/1e6:>7.1f}M  "
                  f"{totals.tokens_kept/1e6:>8.1f}M  {kt:>5.1f}%")


if __name__ == "__main__":
    main()
