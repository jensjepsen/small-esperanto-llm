"""Re-run BoolQ QUESTIONS ONLY through v12 with recase + trailing '?' preproc.

Passages already have proper caps + punctuation so we leave them alone
(the scored file `boolq_roundtrip_scored.jsonl` already covers them).

Output JSONL row (one per question):
  {orig_idx, split, en_orig, en_preproc, en_eo, en_eo_en, chrf, bleu}
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import spacy
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mt" / "scripts"))
from sp_tokenizer import SPMTokenizer  # type: ignore  # noqa: E402
from transformers import MarianMTModel  # noqa: E402

ENT_TAGS = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "PRODUCT",
            "EVENT", "FAC", "NORP", "LANGUAGE"}


def make_recaser():
    nlp = spacy.load("en_core_web_lg")

    def recase_many(texts: list[str], batch_size: int = 256) -> list[str]:
        lowered = [t.lower() for t in texts]
        out = []
        for doc in nlp.pipe(lowered, batch_size=batch_size):
            toks = []
            for t in doc:
                w = t.text
                if (t.i == 0 or t.is_sent_start
                        or t.ent_type_ in ENT_TAGS or t.pos_ == "PROPN"):
                    w = w[:1].upper() + w[1:] if w else w
                toks.append(w + t.whitespace_)
            out.append("".join(toks).strip())
        return out

    return recase_many


def batch_translate(model, tok, srcs, tgt_lang, max_input=500, max_output=192):
    if not srcs:
        return []
    ids_list = [tok.encode(s, lang=tgt_lang)[:max_input] for s in srcs]
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    sorted_ids = [ids_list[i] for i in order]
    be = tok.pad_batch(sorted_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids=be.input_ids.cuda(),
            attention_mask=be.attention_mask.cuda(),
            max_length=max_output,
            do_sample=False,
            num_beams=1,
        )
    decoded = [tok.decode(out[i]) for i in range(len(sorted_ids))]
    results = [""] * len(srcs)
    for sp, op in enumerate(order):
        results[op] = decoded[sp]
    return results


def load_done(out_path):
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["orig_idx"])
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/data2/eo-mt-v12-bidir/final")
    ap.add_argument("--tokenizer", default="/mnt/data2/spm_v2/spm_eneo_48k_v3.model")
    ap.add_argument("--source", default="/mnt/data2/boolq_en.jsonl")
    ap.add_argument("--output", default="/mnt/data2/boolq_q_roundtrip_v2.jsonl")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-append-q", action="store_true",
                    help="Skip appending '?' after recase")
    args = ap.parse_args()

    src_path = Path(args.source)
    out_path = Path(args.output)
    done = load_done(out_path)
    print(f"[resume] processed: {len(done):,}", flush=True)

    print("[spacy] loading en_core_web_lg", flush=True)
    recase_many = make_recaser()

    print(f"[model] loading {args.checkpoint}", flush=True)
    tok = SPMTokenizer(args.tokenizer)
    model = MarianMTModel.from_pretrained(
        args.checkpoint, torch_dtype=torch.float16).cuda().eval()

    import sacrebleu

    stop = {"flag": False}

    def _handle(signum, frame):
        if stop["flag"]:
            os._exit(1)
        print(f"\n[signal {signum}] stopping after current batch", flush=True)
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    # collect pending
    pending = []
    with src_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["orig_idx"] in done:
                continue
            pending.append((r["orig_idx"], r["split"], r["question"]))

    if args.limit:
        pending = pending[:args.limit]
    print(f"[pending] {len(pending):,} questions to process", flush=True)

    t0 = time.time()
    total = 0
    with out_path.open("a", buffering=1) as fout:
        for start in range(0, len(pending), args.batch_size):
            if stop["flag"]:
                break
            batch = pending[start:start + args.batch_size]
            en_orig = [r[2] for r in batch]
            en_pre = recase_many(en_orig)
            if not args.no_append_q:
                en_pre = [
                    p if p.endswith(("?", ".", "!")) else p + "?"
                    for p in en_pre
                ]
            eos = batch_translate(model, tok, en_pre, "eo")
            back = batch_translate(model, tok, eos, "en")
            for (idx, split, orig), pre, eo, b in zip(batch, en_pre, eos, back):
                b_strip = b.rstrip("?.! ").strip()
                chrf = sacrebleu.sentence_chrf(b_strip, [orig]).score
                bleu = sacrebleu.sentence_bleu(b_strip, [orig]).score
                fout.write(json.dumps({
                    "orig_idx": idx,
                    "split": split,
                    "en_orig": orig,
                    "en_preproc": pre,
                    "en_eo": eo,
                    "en_eo_en": b_strip,
                    "chrf": chrf,
                    "bleu": bleu,
                }, ensure_ascii=False) + "\n")
            total += len(batch)
            el = time.time() - t0
            rate = total / max(el, 1e-6)
            eta_min = (len(pending) - total) / max(rate, 1e-6) / 60
            print(f"[{total:>6,}/{len(pending):,}]  {rate:.1f} Qs/s   "
                  f"ETA {eta_min:.1f} min", flush=True)

    el = time.time() - t0
    print(f"\n[done] {total:,} Qs in {el/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
