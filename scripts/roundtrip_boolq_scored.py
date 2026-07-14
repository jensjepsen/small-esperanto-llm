"""Full BoolQ Q round-trip EN->EO->EN with LaBSE + chrF scoring.

Steps per question:
  1. spaCy en_core_web_lg recase (first-word cap + PROPN/entity title-case)
  2. Append '?' if missing terminal punctuation
  3. EN -> EO via MarianMT
  4. EO -> EN via same MarianMT
  5. Strip trailing '?' from back-translation before scoring
  6. sacrebleu chrF + BLEU on (back, orig)
  7. LaBSE cos_sim on (orig, back)

Output JSONL row:
  {orig_idx, split, en_orig, en_preproc, en_eo, en_eo_en, chrf, bleu, cos_sim}
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import sacrebleu
import spacy
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, MarianMTModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mt" / "scripts"))
from sp_tokenizer import SPMTokenizer  # type: ignore  # noqa: E402

ENT_TAGS = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "PRODUCT",
            "EVENT", "FAC", "NORP", "LANGUAGE"}


def make_recaser():
    nlp = spacy.load("en_core_web_lg")

    def recase_many(texts, batch_size=256):
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


def batch_translate(model, tok, srcs, tgt_lang,
                    max_input=500, max_output=192):
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


def load_done(out_path: Path):
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
    ap.add_argument("--checkpoint", required=True,
                    help="MarianMT checkpoint (HF id or local path)")
    ap.add_argument("--tokenizer", required=True,
                    help="SPM model path")
    ap.add_argument("--boolq-splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--labse-batch", type=int, default=256)
    ap.add_argument("--mt-batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output)
    print(f"[resume] processed: {len(done):,}", flush=True)

    print("[data] loading boolq...", flush=True)
    rows = []
    idx = 0
    for split in args.boolq_splits:
        ds = load_dataset("google/boolq", split=split)
        for r in ds:
            if idx not in done:
                rows.append({"orig_idx": idx, "split": split,
                             "question": r["question"]})
            idx += 1
        print(f"  {split}: {len(ds):,} rows (total idx now {idx})", flush=True)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[data] {len(rows):,} Qs to process", flush=True)
    if not rows:
        print("[done] nothing to do", flush=True)
        return

    print("[spacy] loading en_core_web_lg", flush=True)
    recase_many = make_recaser()

    print(f"[model] loading {args.checkpoint}", flush=True)
    tok = SPMTokenizer(args.tokenizer)
    model = MarianMTModel.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16).cuda().eval()

    print("[labse] loading sentence-transformers/LaBSE", flush=True)
    lt = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    lm = AutoModel.from_pretrained("sentence-transformers/LaBSE").cuda().eval().half()

    @torch.no_grad()
    def embed(texts, bs):
        E = []
        for i in range(0, len(texts), bs):
            enc = lt(texts[i:i + bs], padding=True, truncation=True,
                     max_length=128, return_tensors="pt").to("cuda")
            e = lm(**enc).last_hidden_state[:, 0]
            e = torch.nn.functional.normalize(e, dim=1)
            E.append(e.float().cpu())
        return torch.cat(E, 0)

    stop = {"flag": False}

    def _handle(signum, frame):
        if stop["flag"]:
            os._exit(1)
        print(f"\n[signal {signum}] finishing current batch", flush=True)
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    t0 = time.time()
    total = 0
    with args.output.open("a", buffering=1) as fout:
        for start in range(0, len(rows), args.mt_batch):
            if stop["flag"]:
                break
            batch = rows[start:start + args.mt_batch]
            origs = [r["question"] for r in batch]
            recased = recase_many(origs)
            preproc = [
                p if p.endswith(("?", ".", "!")) else p + "?"
                for p in recased
            ]
            eos = batch_translate(model, tok, preproc, "eo")
            backs = batch_translate(model, tok, eos, "en")
            backs_stripped = [b.rstrip("?.! ").strip() for b in backs]

            # LaBSE cos_sim
            e_o = embed(origs, args.labse_batch)
            e_b = embed(backs_stripped, args.labse_batch)
            cos = (e_o * e_b).sum(1).tolist()

            for r, pre, eo, back, c in zip(batch, preproc, eos, backs_stripped, cos):
                chrf = sacrebleu.sentence_chrf(back, [r["question"]]).score
                bleu = sacrebleu.sentence_bleu(back, [r["question"]]).score
                fout.write(json.dumps({
                    "orig_idx": r["orig_idx"],
                    "split": r["split"],
                    "en_orig": r["question"],
                    "en_preproc": pre,
                    "en_eo": eo,
                    "en_eo_en": back,
                    "chrf": chrf,
                    "bleu": bleu,
                    "cos_sim": c,
                }, ensure_ascii=False) + "\n")

            total += len(batch)
            el = time.time() - t0
            rate = total / max(el, 1e-6)
            eta_min = (len(rows) - total) / max(rate, 1e-6) / 60
            print(f"[{total:>6,}/{len(rows):,}]  {rate:.1f} Qs/s  ETA {eta_min:.1f} min",
                  flush=True)

    el = time.time() - t0
    print(f"\n[done] {total:,} Qs in {el/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
