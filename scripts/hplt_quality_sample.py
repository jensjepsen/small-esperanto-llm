"""Probe how our verifier scores HPLT docs vs HPLT's own quality buckets.

Samples N docs from each bucket (7/8/9/10), splits each into sentences,
runs the default Verifier, and reports per-bucket parse-rate and
diagnostics/sentence distributions. Meant to calibrate whether the
verifier is a useful quality filter on top of HPLT scores.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

from rich.console import Console

from esperanto_lm.verify import Verifier, tokenize


CONTENT_POS = {"N", "V", "A", "Adv"}


def is_contentful(sent: str, min_content: int = 5) -> bool:
    """Drop admin lines (URLs, phone numbers, address fragments) so the
    parse-rate measures grammatical sentences, not punctuation survival."""
    toks = tokenize(sent)
    content = [t for t in toks if t.pos in CONTENT_POS]
    if len(content) < min_content:
        return False
    has_verb = any(t.pos == "V" for t in toks)
    return has_verb

console = Console()

HPLT_DIR = Path("data/hplt")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZĈĜĤĴŜŬ])")


def iter_sentences(text: str, max_sents: int) -> list[str]:
    sents = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for s in SENT_SPLIT.split(line):
            s = s.strip()
            if 15 <= len(s) <= 300:
                sents.append(s)
                if len(sents) >= max_sents:
                    return sents
    return sents


def sample_bucket(path: Path, n_docs: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    total = sum(1 for _ in path.open())
    if n_docs >= total:
        idxs = set(range(total))
    else:
        idxs = set(rng.sample(range(total), n_docs))
    out = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i in idxs:
                out.append(json.loads(line))
                if len(out) == len(idxs):
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-per-bucket", type=int, default=200)
    ap.add_argument("--sents-per-doc", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=Path, default=HPLT_DIR,
                    help="Directory with {score}_{shard}.jsonl files")
    ap.add_argument("--show-examples", type=int, default=3,
                    help="How many clean/noisy example sentences to print per bucket")
    ap.add_argument("--min-content-words", type=int, default=5,
                    help="Skip sentences with fewer than N content words "
                         "(nouns/verbs/adjs/advs) or no finite verb. "
                         "Filters out admin lines so parse-rate reflects "
                         "real sentences.")
    args = ap.parse_args()

    verifier = Verifier()

    buckets = {}
    for p in sorted(args.data_dir.glob("*.jsonl")):
        score = int(p.name.split("_")[0])
        buckets[score] = p

    console.print(f"[bold]Buckets found:[/] {list(buckets)}")

    results: dict[int, dict] = {}
    for score, path in sorted(buckets.items()):
        console.print(f"\n[bold green]── bucket {score} "
                      f"({path.name}) ──[/]")
        docs = sample_bucket(path, args.docs_per_bucket, args.seed + score)
        console.print(f"  sampled {len(docs)} docs")

        per_sent_diags: list[int] = []
        per_doc_parse_rate: list[float] = []
        clean_examples: list[str] = []
        noisy_examples: list[tuple[str, list]] = []
        t0 = time.time()
        total_sents = 0
        skipped_low_content = 0

        for d in docs:
            raw = iter_sentences(d.get("text", ""), args.sents_per_doc * 3)
            # After filter we want at least args.sents_per_doc contentful
            # sentences. Over-sample and take the first N that pass.
            sents = []
            for s in raw:
                if is_contentful(s, args.min_content_words):
                    sents.append(s)
                    if len(sents) >= args.sents_per_doc:
                        break
                else:
                    skipped_low_content += 1
            if not sents:
                continue
            clean = 0
            for s in sents:
                diags = verifier.verify(s)
                per_sent_diags.append(len(diags))
                total_sents += 1
                if not diags:
                    clean += 1
                    if len(clean_examples) < args.show_examples:
                        clean_examples.append(s)
                elif len(diags) >= 3 and len(noisy_examples) < args.show_examples:
                    noisy_examples.append((s, diags))
            per_doc_parse_rate.append(clean / len(sents))

        dt = time.time() - t0
        rate = total_sents / dt if dt else 0
        console.print(f"  processed {total_sents:,} sents in {dt:.1f}s "
                      f"({rate:.0f} sents/s, skipped {skipped_low_content:,} "
                      f"low-content)")

        results[score] = {
            "docs_used": len(per_doc_parse_rate),
            "total_sents": total_sents,
            "parse_rate_mean": statistics.fmean(per_doc_parse_rate) if per_doc_parse_rate else 0,
            "parse_rate_median": statistics.median(per_doc_parse_rate) if per_doc_parse_rate else 0,
            "diags_per_sent_mean": statistics.fmean(per_sent_diags) if per_sent_diags else 0,
            "diags_per_sent_p50": statistics.median(per_sent_diags) if per_sent_diags else 0,
            "pct_sents_clean": sum(1 for d in per_sent_diags if d == 0) / len(per_sent_diags) if per_sent_diags else 0,
        }

        for s in clean_examples:
            console.print(f"  [green]✓[/] {s}")
        for s, diags in noisy_examples:
            codes = ",".join(sorted({d.check for d in diags}))
            console.print(f"  [red]✗[/] ({codes}) {s[:120]}")

    console.print("\n[bold]── Summary ──[/]")
    header = f"{'bucket':>6}  {'docs':>5}  {'sents':>6}  {'parse%':>7}  {'med':>5}  {'diag/sent':>10}  {'sent_clean%':>11}"
    console.print(header)
    for score, r in sorted(results.items()):
        console.print(f"{score:>6}  {r['docs_used']:>5}  {r['total_sents']:>6}  "
                      f"{100*r['parse_rate_mean']:>6.1f}%  "
                      f"{100*r['parse_rate_median']:>4.0f}%  "
                      f"{r['diags_per_sent_mean']:>10.2f}  "
                      f"{100*r['pct_sents_clean']:>10.1f}%")


if __name__ == "__main__":
    main()
