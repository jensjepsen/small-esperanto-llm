"""Round-trip BoolQ EN → EO → EN via v12 with proper preproc.

Pipeline vs v1 (`roundtrip_boolq.py`):
  1. spaCy en_core_web_lg recase (first-word cap + PROPN/entity title-case)
  2. Append '?' to questions that lack terminal punctuation
  3. Strip trailing '?' from back-translations before scoring

Interrupt-safe: resume via (orig_idx, kind, sent_idx) triple.

Output JSONL row:
  {orig_idx, split, kind, sent_idx,
   en_orig,         # untouched original (lowercase, no ?)
   en_preproc,      # recased + optional ? (what we actually fed to v12)
   en_eo,           # forward hop
   en_eo_en,        # back hop, trailing ? stripped
   chrf, bleu,      # scored vs en_orig
   ...}
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

import spacy
import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mt" / "scripts"))
from sp_tokenizer import SPMTokenizer  # type: ignore  # noqa: E402
from transformers import MarianMTModel  # noqa: E402

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
ENT_TAGS = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "PRODUCT",
            "EVENT", "FAC", "NORP", "LANGUAGE"}


def sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def _make_recaser():
    nlp = spacy.load("en_core_web_lg")

    def _recase_from_doc(doc) -> str:
        out = []
        for t in doc:
            w = t.text
            if (t.i == 0 or t.is_sent_start
                    or t.ent_type_ in ENT_TAGS or t.pos_ == "PROPN"):
                w = w[:1].upper() + w[1:] if w else w
            out.append(w + t.whitespace_)
        return "".join(out).strip()

    def recase(text: str) -> str:
        return _recase_from_doc(nlp(text.lower()))

    def recase_many(texts: list[str], batch_size: int = 64) -> list[str]:
        lowered = [t.lower() for t in texts]
        return [_recase_from_doc(d) for d in nlp.pipe(lowered, batch_size=batch_size)]

    recase.many = recase_many  # type: ignore[attr-defined]
    return recase


def preproc_batch(recase, rows: list[tuple]) -> list[str]:
    """Recase a batch of (i, split, kind, sent_idx, en_orig) rows in one nlp.pipe pass.

    Returns list of preprocessed strings aligned to rows.
    """
    texts = [r[4] for r in rows]
    recased = recase.many(texts)
    out = []
    for (i, split, kind, si, _en), r in zip(rows, recased):
        if kind == "q" and not r.endswith(("?", ".", "!")):
            r = r + "?"
        out.append(r)
    return out


def batch_translate(model, tok, srcs: list[str], tgt_lang: str,
                    max_input: int = 500, max_output: int = 256,
                    device: str = "cuda") -> list[str]:
    if not srcs:
        return []
    ids_list = [tok.encode(s, lang=tgt_lang)[:max_input] for s in srcs]
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    sorted_ids = [ids_list[i] for i in order]
    be = tok.pad_batch(sorted_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids=be.input_ids.to(device),
            attention_mask=be.attention_mask.to(device),
            max_length=max_output,
            do_sample=False,
            num_beams=1,
        )
    decoded_sorted = [tok.decode(out[i]) for i in range(len(sorted_ids))]
    results = [""] * len(srcs)
    for sp, op in enumerate(order):
        results[op] = decoded_sorted[sp]
    return results


def _sacrebleu():
    import sacrebleu
    return sacrebleu


def load_done(output_path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not output_path.exists():
        return done
    with output_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["orig_idx"], r["kind"], r["sent_idx"]))
            except Exception:
                continue
    return done


def materialize_source(src_path: Path, splits: list[str]) -> None:
    if src_path.exists():
        return
    print(f"[source] materializing BoolQ → {src_path}", flush=True)
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
            print(f"[source]   {split}: {len(ds):,} rows", flush=True)


def rows_from(src_path: Path, done: set[tuple]):
    """Yield (row_idx, split, kind, sent_idx, en_orig) — no recasing yet."""
    with src_path.open() as f:
        for line in f:
            r = json.loads(line)
            i = r["orig_idx"]
            key = (i, "q", 0)
            if key not in done:
                yield (i, r["split"], "q", 0, r["question"])
            for si, s in enumerate(sentences(r["passage"])):
                key = (i, "p", si)
                if key not in done:
                    yield (i, r["split"], "p", si, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/data2/eo-mt-v12-bidir/final")
    ap.add_argument("--tokenizer", default="/mnt/data2/spm_v2/spm_eneo_48k_v3.model")
    ap.add_argument("--source", default="/mnt/data2/boolq_en.jsonl")
    ap.add_argument("--output", default="/mnt/data2/boolq_roundtrip_v2.jsonl")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--max-input-tokens", type=int, default=500)
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after this many sentences (0 = all)")
    args = ap.parse_args()

    src_path = Path(args.source)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    materialize_source(src_path, args.splits)
    done = load_done(out_path)
    print(f"[resume] already processed: {len(done):,} sentences", flush=True)

    print("[spacy] loading en_core_web_lg", flush=True)
    recase = _make_recaser()

    stop = {"flag": False}

    def _handle(signum, frame):
        if stop["flag"]:
            os._exit(1)
        print(f"\n[signal {signum}] finishing current batch then stopping",
              flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(f"[model] loading {args.checkpoint}", flush=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    tok = SPMTokenizer(args.tokenizer)
    model = MarianMTModel.from_pretrained(
        args.checkpoint, torch_dtype=dtype).cuda().eval()

    sb = _sacrebleu()

    def flush(batch_rows, fout):
        ens_pre = preproc_batch(recase, batch_rows)
        eos = batch_translate(model, tok, ens_pre, "eo",
                              args.max_input_tokens, args.max_output_tokens)
        back_ens = batch_translate(model, tok, eos, "en",
                                    args.max_input_tokens, args.max_output_tokens)
        for (i, split, kind, si, en_orig), en_pre, eo, back in zip(
                batch_rows, ens_pre, eos, back_ens):
            back_stripped = back.rstrip("?.! ").strip() if kind == "q" else back
            ref = en_orig.rstrip("?.! ").strip() if kind == "q" else en_orig
            chrf = sb.sentence_chrf(back_stripped, [ref]).score
            bleu = sb.sentence_bleu(back_stripped, [ref]).score
            fout.write(json.dumps({
                "orig_idx": i,
                "split": split,
                "kind": kind,
                "sent_idx": si,
                "en_orig": en_orig,
                "en_preproc": en_pre,
                "en_eo": eo,
                "en_eo_en": back_stripped,
                "chrf": chrf,
                "bleu": bleu,
            }, ensure_ascii=False) + "\n")

    total = 0
    t0 = time.time()
    batch: list[tuple] = []
    with out_path.open("a", buffering=1) as fout:
        for row in rows_from(src_path, done):
            batch.append(row)
            if len(batch) >= args.batch_size:
                flush(batch, fout)
                total += len(batch)
                batch = []
                el = time.time() - t0
                rate = total / max(el, 1e-6)
                print(f"[{total:>7,} sents]  {rate:.1f} sents/s", flush=True)
                if stop["flag"] or (args.limit and total >= args.limit):
                    break
        if batch and not stop["flag"]:
            flush(batch, fout)
            total += len(batch)

    el = time.time() - t0
    print(f"\n[done] {total:,} sentences in {el/60:.1f} min  "
          f"({total / max(el, 1e-6):.1f} sents/s)", flush=True)


if __name__ == "__main__":
    main()
