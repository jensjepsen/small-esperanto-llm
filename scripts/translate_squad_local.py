"""Translate SQuAD v1 to Esperanto with v11 MT on the local GPU.

Same interrupt-safe design as translate_metamath_local.py: materialize source
to a stable JSONL, append per-row to output, resume by orig_idx.

SQuAD schema:
  {id, title, context, question, answers: {text: [...], answer_start: [...]}}
Output schema (character offsets are dropped — they don't survive translation
and downstream code can re-align by string search when it needs them):
  {id, title, split, context_en, context_eo, question_en, question_eo,
   answers_en, answers_eo}

The v11 tokenizer's `<eo>` prefix handles direction. Sentence-level pooling +
in-chunk dedup + length-sorted batches keep GPU busy on the 1080 Ti.
LatexAwareTranslator with the terminator patch handles the short-answer case
that used to hallucinate ("Grumman" → "grummancity in germany").
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


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def raw_translate_batch(tr: LatexAwareTranslator, srcs: list[str]) -> list[str]:
    """Translate without the currency/LaTeX sentinel protection.

    v11's decoder doesn't emit `<extra_N>` tokens through translation (it
    substitutes rare glyphs like Tibetan `་` instead), so protecting `$X` /
    `£X` in isolated short spans hurts more than it helps — the sentinel
    never comes back and `_restore` returns the mangled text. Bypassing
    protection lets bare currency answers round-trip via the model's own
    token embeddings, plus the terminator patch keeps short spans stable.
    """
    if not srcs:
        return []
    # Apply the same normalization + terminator fix the wrapper does.
    from translate_with_latex import normalize_unicode_math  # type: ignore
    prepped = [normalize_unicode_math(s) for s in srcs]
    prep_and_flags = [tr._ensure_terminator(s) for s in prepped]
    prepped = [s for s, _ in prep_and_flags]
    added = [f for _, f in prep_and_flags]

    ids_list = [tr.tok.encode(s, lang="eo")[: tr.max_input_tokens] for s in prepped]
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    sorted_ids = [ids_list[i] for i in order]
    be = tr.tok.pad_batch(sorted_ids)
    with torch.no_grad():
        out = tr.model.generate(
            input_ids=be.input_ids.to(tr.device),
            attention_mask=be.attention_mask.to(tr.device),
            max_length=tr.max_output_tokens,
            do_sample=False,
            num_beams=1,
        )
    decoded_sorted = [tr.tok.decode(out[i]) for i in range(len(sorted_ids))]
    results = [""] * len(srcs)
    for sort_pos, orig_pos in enumerate(order):
        text = decoded_sorted[sort_pos]
        if added[orig_pos]:
            text = tr._strip_appended_terminator(text)
        results[orig_pos] = text
    return results


def sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def load_done(output_path: Path) -> set[str]:
    """SQuAD row id is a string ('56be4db0acb8001400a502ec')."""
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("id")
            if isinstance(rid, str):
                done.add(rid)
    return done


def materialize_source(src_path: Path, splits: list[str]) -> None:
    if src_path.exists():
        return
    print(f"materializing SQuAD splits {splits} → {src_path}", flush=True)
    with src_path.open("w") as f:
        for split in splits:
            ds = load_dataset("rajpurkar/squad", split=split)
            for r in ds:
                f.write(json.dumps({
                    "id": r["id"],
                    "title": r["title"],
                    "split": split,
                    "context": r["context"],
                    "question": r["question"],
                    "answers": r["answers"]["text"],  # list; strip offsets
                }, ensure_ascii=False) + "\n")
            print(f"  wrote {split}: {len(ds):,} rows", flush=True)


class Chunker:
    def __init__(self, src_path: Path, done: set[str], chunk_rows: int):
        self.src_path = src_path
        self.done = done
        self.chunk_rows = chunk_rows
        self.n_skip_done = 0

    def __iter__(self):
        buf: list[dict] = []
        with self.src_path.open() as f:
            for line in f:
                r = json.loads(line)
                if r["id"] in self.done:
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
    # (row_idx, kind, sub_idx, text)   kind ∈ {"c","q","a"}
    tasks: list[tuple[int, str, int, str]] = []
    for ri, r in enumerate(rows):
        for si, s in enumerate(sentences(r["context"])):
            tasks.append((ri, "c", si, s))
        # question and answers are usually short — translate as-is
        tasks.append((ri, "q", 0, r["question"]))
        for ai, ans in enumerate(r["answers"]):
            tasks.append((ri, "a", ai, ans))

    unique_map: dict[str, str] = {}
    for _, _, _, s in tasks:
        unique_map.setdefault(s, "")
    uniques = list(unique_map.keys())

    for j in range(0, len(uniques), bs):
        chunk = uniques[j : j + bs]
        for src, dst in zip(chunk, raw_translate_batch(tr, chunk)):
            unique_map[src] = dst

    per_row: dict[int, dict[str, dict[int, str]]] = {
        i: {"c": {}, "q": {}, "a": {}} for i in range(len(rows))
    }
    for ri, kind, si, s in tasks:
        per_row[ri][kind][si] = unique_map[s]

    out: list[dict] = []
    for ri, r in enumerate(rows):
        c_eo = " ".join(per_row[ri]["c"][i] for i in sorted(per_row[ri]["c"]))
        q_eo = per_row[ri]["q"][0]
        answers_eo = [per_row[ri]["a"][i] for i in sorted(per_row[ri]["a"])]
        out.append({
            "id": r["id"],
            "title": r["title"],
            "split": r["split"],
            "context_en": r["context"],
            "context_eo": c_eo,
            "question_en": r["question"],
            "question_eo": q_eo,
            "answers_en": r["answers"],
            "answers_eo": answers_eo,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/data2/eo-mt-v11/checkpoint-99099")
    ap.add_argument("--tokenizer", default="mt/data/tokenizer/spm_eneo_48k_v2.model")
    ap.add_argument("--source", default="/mnt/data2/squad_en.jsonl")
    ap.add_argument("--output", default="/mnt/data2/squad_eo.jsonl")
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
