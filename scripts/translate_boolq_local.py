"""Translate BoolQ (yes/no reading comp) to Esperanto with v11 MT on the
local GPU. Same interrupt-safe design as the SQuAD translator.

Schema:
  BoolQ: {question (no trailing '?'), passage, answer: bool}
  Output: {orig_idx, split, question_en, question_eo, passage_en,
           passage_eo, answer_bool, answer_eo}

Notes:
  * BoolQ's `question` field drops the `?` — v11 then treats it as a
    declarative and loses the interrogative `ĉu`. We append `?` before
    translation; the model reliably produces `ĉu ... ?` output.
  * Answer becomes 'jes' / 'ne' (literal Esperanto for yes/no).
  * Sentence-level pooling + in-chunk dedup + no LaTeX prep needed.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mt" / "scripts"))
from translate_with_latex import LatexAwareTranslator  # type: ignore  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate_squad_local import raw_translate_batch  # type: ignore  # noqa: E402


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def ensure_question_mark(q: str) -> str:
    """BoolQ drops trailing '?' — v11 loses interrogative `ĉu` without it."""
    q = q.strip()
    if q and q[-1] not in "?!.":
        q += "?"
    return q


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


def materialize_source(src_path: Path, splits: list[str]) -> None:
    if src_path.exists():
        return
    print(f"materializing BoolQ splits {splits} → {src_path}", flush=True)
    idx = 0
    with src_path.open("w") as f:
        for split in splits:
            ds = load_dataset("google/boolq", split=split)
            for r in ds:
                f.write(json.dumps({
                    "orig_idx": idx,
                    "split": split,
                    "question": r["question"],
                    "passage": r["passage"],
                    "answer": bool(r["answer"]),
                }, ensure_ascii=False) + "\n")
                idx += 1
            print(f"  wrote {split}: {len(ds):,} rows", flush=True)


class Chunker:
    def __init__(self, src_path: Path, done: set[int], chunk_rows: int):
        self.src_path = src_path
        self.done = done
        self.chunk_rows = chunk_rows
        self.n_skip_done = 0

    def __iter__(self):
        buf: list[dict] = []
        with self.src_path.open() as f:
            for line in f:
                r = json.loads(line)
                if r["orig_idx"] in self.done:
                    self.n_skip_done += 1
                    continue
                buf.append(r)
                if len(buf) >= self.chunk_rows:
                    yield buf
                    buf = []
        if buf:
            yield buf


def translate_chunk(rows: list[dict], tr: LatexAwareTranslator, bs: int) -> list[dict]:
    """Pool all sentences across rows; dedup; length-sort batch; splice back."""
    tasks: list[tuple[int, str, int, str]] = []
    for ri, r in enumerate(rows):
        # passage sentences
        for si, s in enumerate(sentences(r["passage"])):
            tasks.append((ri, "p", si, s))
        # question with '?' appended for interrogative rendering
        tasks.append((ri, "q", 0, ensure_question_mark(r["question"])))

    unique_map: dict[str, str] = {}
    for _, _, _, s in tasks:
        unique_map.setdefault(s, "")
    uniques = list(unique_map.keys())

    for j in range(0, len(uniques), bs):
        chunk = uniques[j : j + bs]
        for src, dst in zip(chunk, raw_translate_batch(tr, chunk)):
            unique_map[src] = dst

    per_row: dict[int, dict[str, dict[int, str]]] = {
        i: {"p": {}, "q": {}} for i in range(len(rows))
    }
    for ri, kind, si, s in tasks:
        per_row[ri][kind][si] = unique_map[s]

    out: list[dict] = []
    for ri, r in enumerate(rows):
        p_eo = " ".join(per_row[ri]["p"][i] for i in sorted(per_row[ri]["p"]))
        q_eo = per_row[ri]["q"][0]
        out.append({
            "orig_idx": r["orig_idx"],
            "split": r["split"],
            "question_en": r["question"],
            "question_eo": q_eo,
            "passage_en": r["passage"],
            "passage_eo": p_eo,
            "answer_bool": r["answer"],
            "answer_eo": "jes" if r["answer"] else "ne",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/data2/eo-mt-v11/checkpoint-99099")
    ap.add_argument("--tokenizer", default="mt/data/tokenizer/spm_eneo_48k_v2.model")
    ap.add_argument("--source", default="/mnt/data2/boolq_en.jsonl")
    ap.add_argument("--output", default="/mnt/data2/boolq_eo.jsonl")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk-rows", type=int, default=256)
    ap.add_argument("--max-input-tokens", type=int, default=500)
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src_path = Path(args.source)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    materialize_source(src_path, args.splits)

    done = load_done(out_path)
    print(f"[resume] already done: {len(done):,}", flush=True)

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

    chunker = Chunker(src_path, done, args.chunk_rows)
    total_written = 0
    t0 = time.time()

    with out_path.open("a", buffering=1) as fout:
        for chunk in chunker:
            if stop["flag"]:
                break
            results = translate_chunk(chunk, tr, args.batch_size)
            for row in results:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_written += len(results)
            el = time.time() - t0
            rate = total_written / max(el, 1e-6)
            print(
                f"[{total_written + len(done):>6,} written]"
                f" {rate:.2f} rows/s"
                f"  skip_done={chunker.n_skip_done:,}",
                flush=True,
            )
            if args.limit and total_written >= args.limit:
                print(f"[limit {args.limit} reached] stopping", flush=True)
                break

    el = time.time() - t0
    print(
        f"\ndone: wrote {total_written:,} rows in {el/60:.1f} min "
        f"({total_written / max(el, 1e-6):.2f} rows/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
