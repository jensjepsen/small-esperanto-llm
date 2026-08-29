"""Resumable EN→EO translation of MetaMathQA on local GPU.

* Materializes the source once to a stable JSONL so `orig_idx` is a permanent row id.
* Writes each translated row to `--output` immediately (line-buffered), so ^C only
  ever loses the row currently being written.
* Skips rows already present in `--output` and rows whose estimated EO token count
  exceeds `--max-eo-tokens` (default 512).
* Sentence-splits each Q and A, pools all sentences across a chunk of rows,
  dedups exact repeats, sorts by length for padding efficiency, then reassembles.

Invoke via `uv run --with tokenizers python scripts/translate_metamath_local.py …`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset

# translator lives in mt/scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mt" / "scripts"))
from translate_with_latex import LatexAwareTranslator  # type: ignore  # noqa: E402


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def raw_translate_batch(translator: LatexAwareTranslator, srcs: list[str]) -> list[str]:
    """Translate without sentinel protection. v11 preserves inline $…$, \\cdot,
    \\sqrt, \\frac and currency itself; sentinel scheme destroys math-heavy rows
    because v11's decoder never emits <extra_N> in these positions and falls
    back to nearby rare glyphs (Tibetan) instead."""
    if not srcs:
        return []
    ids_list = [
        translator.tok.encode(s, lang="eo")[: translator.max_input_tokens]
        for s in srcs
    ]
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    sorted_ids = [ids_list[i] for i in order]
    be = translator.tok.pad_batch(sorted_ids)
    with torch.no_grad():
        out = translator.model.generate(
            input_ids=be.input_ids.to(translator.device),
            attention_mask=be.attention_mask.to(translator.device),
            max_length=translator.max_output_tokens,
            do_sample=False,
            num_beams=1,
        )
    decoded_sorted = [translator.tok.decode(out[i]) for i in range(len(sorted_ids))]
    results = [""] * len(srcs)
    for sort_pos, orig_pos in enumerate(order):
        results[orig_pos] = decoded_sorted[sort_pos]
    return results

# From the 3k-row calibration on data/translate_orca_full_fixed.jsonl
# with tokenizer_morpheme: 1 EO token ≈ 2.535 EN chars.
EN_CHARS_PER_EO_TOK = 2.535


def sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def load_done(output_path: Path) -> set[int]:
    done: set[int] = set()
    if not output_path.exists():
        return done
    with output_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            i = r.get("orig_idx")
            if isinstance(i, int):
                done.add(i)
    return done


def materialize_source(src_path: Path) -> None:
    if src_path.exists():
        return
    print(f"materializing MetaMathQA to {src_path}", flush=True)
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    with src_path.open("w") as f:
        for i, r in enumerate(ds):
            f.write(json.dumps({
                "orig_idx": i,
                "type": r.get("type", ""),
                "query": r["query"],
                "response": r["response"],
                "original_question": r.get("original_question", ""),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(ds):,} rows", flush=True)


class Chunker:
    """Iterate the source jsonl and yield chunks of unfinished, size-eligible rows."""

    def __init__(self, src_path: Path, done: set[int], chunk_rows: int, max_en_chars: int):
        self.src_path = src_path
        self.done = done
        self.chunk_rows = chunk_rows
        self.max_en_chars = max_en_chars
        self.n_skip_done = 0
        self.n_skip_long = 0
        self.n_yielded = 0

    def __iter__(self):
        buf: list[dict] = []
        with self.src_path.open() as f:
            for line in f:
                r = json.loads(line)
                if r["orig_idx"] in self.done:
                    self.n_skip_done += 1
                    continue
                total_en = len(r["query"]) + 1 + len(r["response"])
                if total_en > self.max_en_chars:
                    self.n_skip_long += 1
                    continue
                buf.append(r)
                self.n_yielded += 1
                if len(buf) >= self.chunk_rows:
                    yield buf
                    buf = []
        if buf:
            yield buf


def translate_chunk(
    rows: list[dict],
    translator: LatexAwareTranslator,
    bs: int,
) -> list[dict]:
    """Sentence-pool a chunk, translate uniques in length-sorted batches, splice back."""
    # (row_idx, side, sent_idx, sent_text)
    tasks: list[tuple[int, str, int, str]] = []
    for ri, r in enumerate(rows):
        for si, s in enumerate(sentences(r["query"])):
            tasks.append((ri, "q", si, s))
        for si, s in enumerate(sentences(r["response"])):
            tasks.append((ri, "a", si, s))

    # Dedup exact repeats
    unique_map: dict[str, str] = {}
    for _, _, _, s in tasks:
        unique_map.setdefault(s, "")
    uniques = list(unique_map.keys())

    for j in range(0, len(uniques), bs):
        chunk = uniques[j : j + bs]
        translations = raw_translate_batch(translator, chunk)
        for src, dst in zip(chunk, translations):
            unique_map[src] = dst

    # Reassemble per row
    per_row: dict[int, dict[str, dict[int, str]]] = {
        i: {"q": {}, "a": {}} for i in range(len(rows))
    }
    for ri, side, si, s in tasks:
        per_row[ri][side][si] = unique_map[s]

    out: list[dict] = []
    for ri, r in enumerate(rows):
        q_eo = " ".join(per_row[ri]["q"][i] for i in sorted(per_row[ri]["q"]))
        a_eo = " ".join(per_row[ri]["a"][i] for i in sorted(per_row[ri]["a"]))
        out.append({
            "orig_idx": r["orig_idx"],
            "type": r["type"],
            "q_en": r["query"],
            "q_eo": q_eo,
            "a_en": r["response"],
            "a_eo": a_eo,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/data2/eo-mt-v11/checkpoint-99099")
    ap.add_argument("--tokenizer", default="mt/data/tokenizer/spm_eneo_48k_v2.model")
    ap.add_argument("--source", default="/mnt/data2/metamath_en.jsonl")
    ap.add_argument("--output", default="/mnt/data2/metamath_eo.jsonl")
    ap.add_argument("--max-eo-tokens", type=int, default=512,
                    help="Skip rows whose estimated EO token count exceeds this")
    ap.add_argument("--batch-size", type=int, default=48,
                    help="Sentences per model.generate() call")
    ap.add_argument("--chunk-rows", type=int, default=256,
                    help="Rows pooled together for cross-row sentence dedup")
    ap.add_argument("--max-input-tokens", type=int, default=500)
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after writing this many rows (0 = no limit)")
    args = ap.parse_args()

    src_path = Path(args.source)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    materialize_source(src_path)

    done = load_done(out_path)
    print(f"already done: {len(done):,}", flush=True)

    stop = {"flag": False}

    def _handle(signum, frame):
        if stop["flag"]:
            print("\n[double-interrupt] hard exit", flush=True)
            os._exit(1)
        print(f"\n[signal {signum}] finishing current chunk then stopping", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(f"loading translator from {args.checkpoint}", flush=True)
    tr = LatexAwareTranslator(
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
    )

    max_en_chars = int(args.max_eo_tokens * EN_CHARS_PER_EO_TOK)
    print(f"skip threshold: EN chars > {max_en_chars} (≈ {args.max_eo_tokens} EO tokens)", flush=True)

    chunker = Chunker(src_path, done, args.chunk_rows, max_en_chars)
    total_written = 0
    total_sents = 0
    total_uniques = 0
    t0 = time.time()

    # Line-buffered append. Each row is flushed the moment it lands.
    with out_path.open("a", buffering=1) as fout:
        for chunk in chunker:
            if stop["flag"]:
                break
            results = translate_chunk(chunk, tr, args.batch_size)
            for row in results:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_written += len(results)

            # Cheap stats for the heartbeat
            n_sents = sum(
                len(sentences(r["query"])) + len(sentences(r["response"]))
                for r in chunk
            )
            n_uniques = len({
                s
                for r in chunk
                for s in sentences(r["query"]) + sentences(r["response"])
            })
            total_sents += n_sents
            total_uniques += n_uniques

            el = time.time() - t0
            rate = total_written / max(el, 1e-6)
            dedup = 1 - total_uniques / max(total_sents, 1)
            print(
                f"[{total_written:>6,} written]"
                f" {rate:.2f} rows/s"
                f"  dedup {dedup*100:.1f}%"
                f"  skip_done={chunker.n_skip_done:,}"
                f"  skip_long={chunker.n_skip_long:,}",
                flush=True,
            )

            if args.limit and total_written >= args.limit:
                print(f"[limit {args.limit} reached] stopping", flush=True)
                break

    el = time.time() - t0
    print(
        f"\ndone: wrote {total_written:,} rows in {el/60:.1f} min"
        f"  ({total_written / max(el, 1e-6):.2f} rows/s)",
        flush=True,
    )
    print(f"skipped: done={chunker.n_skip_done:,}  over-length={chunker.n_skip_long:,}", flush=True)


if __name__ == "__main__":
    main()
