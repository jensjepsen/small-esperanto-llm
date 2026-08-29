"""Round-trip BoolQ EN → EO → EN via v12 MT locally to flag mistranslations.

For each sentence in each BoolQ passage (and each question), translate
EN→EO with v12, then EO→EN with the same model, and score the round-trip
with chrF. Low chrF ⇒ a mistranslation (either forward or backward hop
lost/added meaning).

Output JSONL rows (one per sentence):

    {"orig_idx", "split", "kind", "sent_idx",
     "en_orig", "en_eo", "en_eo_en", "chrf", "bleu"}

Sort ascending by chrf to see the worst mistranslations first. Interrupt-
safe append + resume via (orig_idx, kind, sent_idx) triple.
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
from sp_tokenizer import SPMTokenizer  # type: ignore  # noqa: E402
from transformers import MarianMTModel  # noqa: E402

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


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
    """Yield (row_idx, kind, sent_idx, en_text) tuples to process, skipping done."""
    with src_path.open() as f:
        for line in f:
            r = json.loads(line)
            i = r["orig_idx"]
            # question first
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
    ap.add_argument("--output", default="/mnt/data2/boolq_roundtrip.jsonl")
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

    stop = {"flag": False}
    def _handle(signum, frame):
        if stop["flag"]:
            os._exit(1)
        print(f"\n[signal {signum}] finishing current batch then stopping", flush=True)
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(f"[model] loading {args.checkpoint}", flush=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    tok = SPMTokenizer(args.tokenizer)
    model = MarianMTModel.from_pretrained(args.checkpoint, torch_dtype=dtype).cuda().eval()

    sb = _sacrebleu()

    def flush(batch_rows, fout):
        # batch_rows: list of (i, split, kind, sent_idx, en)
        ens = [r[4] for r in batch_rows]
        eos = batch_translate(model, tok, ens, "eo",
                              args.max_input_tokens, args.max_output_tokens)
        back_ens = batch_translate(model, tok, eos, "en",
                                    args.max_input_tokens, args.max_output_tokens)
        for (i, split, kind, si, en), eo, back in zip(batch_rows, eos, back_ens):
            chrf = sb.sentence_chrf(back, [en]).score
            bleu = sb.sentence_bleu(back, [en]).score
            fout.write(json.dumps({
                "orig_idx": i,
                "split": split,
                "kind": kind,
                "sent_idx": si,
                "en_orig": en,
                "en_eo": eo,
                "en_eo_en": back,
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
