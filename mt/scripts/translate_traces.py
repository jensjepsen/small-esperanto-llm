"""Translate existing English (question, CoT-answer) pairs to Esperanto.

Skips the LFM generation step — we already have well-formed reasoning chains
from a source dataset (Orca-Math, MetaMath, GSM8K gold, etc). Just clean, then
v5b en→eo. Yield is much higher than distilling from a small model since
there's no truncation, no wrong-answer drops; only budget filter losses.

Re-uses the helpers in distill_lfm_to_eo.py so preprocessing stays consistent.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sp_tokenizer import SPMTokenizer
from distill_lfm_to_eo import (
    strip_markdown, strip_latex, replace_final_marker, split_sentences,
    extract_final_number, HEDGE_PATTERNS,
)


def strip_gsm_calc_markers(text: str) -> str:
    """Strip the <<2+3=5>> calculator-trace markers from GSM-style answers."""
    return re.sub(r"<<[^>]*>>", "", text)


def load_orca_math(n: int, skip: int):
    """microsoft/orca-math-word-problems-200k: columns 'question', 'answer'.

    Answers are GPT-4 generated step-by-step prose, typically ending in
    'The answer is: N' or '#### N'.
    """
    from datasets import load_dataset
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    end = min(skip + n, len(ds))
    out = []
    for i in range(skip, end):
        out.append((ds[i]["question"].strip(), ds[i]["answer"].strip()))
    return out


def load_metamath(n: int, skip: int):
    """meta-math/MetaMathQA: columns 'query', 'response' (CoT)."""
    from datasets import load_dataset
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    end = min(skip + n, len(ds))
    out = []
    for i in range(skip, end):
        out.append((ds[i]["query"].strip(), ds[i]["response"].strip()))
    return out


def load_gsm8k_gold(n: int, skip: int):
    """openai/gsm8k train: gold human-written step-by-step answers."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    end = min(skip + n, len(ds))
    out = []
    for i in range(skip, end):
        out.append((ds[i]["question"].strip(), ds[i]["answer"].strip()))
    return out


SOURCES = {
    "orca_math": load_orca_math,
    "metamath": load_metamath,
    "gsm8k_gold": load_gsm8k_gold,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(SOURCES))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--mt-checkpoint", required=True)
    ap.add_argument("--mt-tokenizer", required=True)
    ap.add_argument("--student-tokenizer", default="tokenizer_morpheme")
    ap.add_argument("--mt-batch-size", type=int, default=256)
    ap.add_argument("--mt-num-beams", type=int, default=2)
    ap.add_argument("--mt-max-length", type=int, default=256)
    ap.add_argument("--max-q-chars", type=int, default=800)
    ap.add_argument("--max-a-chars", type=int, default=2500)
    ap.add_argument("--max-student-tokens", type=int, default=512)
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--require-answer-match", action="store_true",
                    help="Drop rows whose EO-translated answer's final number "
                         "doesn't match the EN gold. Cheap insurance.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-cache", default="/workspace/hf-cache")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_cache)
    from transformers import AutoTokenizer, MarianMTModel

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try: done_ids.add(json.loads(line)["i"])
                except Exception: pass
        print(f"Resume: {len(done_ids)} rows already in {out_path}")

    print(f"Loading {args.source} (n={args.n}, skip={args.skip})…")
    pairs = SOURCES[args.source](args.n, args.skip)
    todo = [(i, q, a) for i, (q, a) in enumerate(pairs) if i not in done_ids]
    print(f"  {len(pairs)} loaded, {len(todo)} to process")
    if not todo:
        return

    print(f"Loading MT {args.mt_checkpoint}…")
    mt_tok = SPMTokenizer(args.mt_tokenizer)
    # eager attn avoids a transformers-5.x SDPA bug that crashes on short
    # / single-token Marian batches with a device-side assert.
    mt_model = MarianMTModel.from_pretrained(
        args.mt_checkpoint, attn_implementation="eager"
    ).half().to("cuda").eval()
    mt_model.generation_config.no_repeat_ngram_size = 5

    print(f"Loading student tokenizer {args.student_tokenizer}…")
    student_tok = AutoTokenizer.from_pretrained(args.student_tokenizer)
    def stu(text): return len(student_tok(text, add_special_tokens=False).input_ids)

    def _sorted_batches(texts, bs):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        return [(order[i:i+bs], [texts[j] for j in order[i:i+bs]])
                for i in range(0, len(order), bs)]

    def mt_translate(texts, desc):
        out = [None] * len(texts)
        # Skip empties to avoid degenerate single-token batches; pad-out result.
        nonempty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not nonempty:
            return [""] * len(texts)
        order_texts = [t for _, t in nonempty]
        for idx, chunk in tqdm(_sorted_batches(order_texts, args.mt_batch_size),
                               desc=desc, leave=False):
            ids = [mt_tok.encode(t, lang="eo") for t in chunk]
            be = mt_tok.pad_batch(ids)
            with torch.no_grad():
                gen = mt_model.generate(
                    input_ids=be.input_ids.to("cuda"),
                    attention_mask=be.attention_mask.to("cuda"),
                    num_beams=args.mt_num_beams, max_length=args.mt_max_length,
                    early_stopping=True, no_repeat_ngram_size=5,
                )
            for orig_i, seq in zip(idx, gen):
                out[nonempty[orig_i][0]] = mt_tok.decode(seq)
        return [t if t is not None else "" for t in out]

    def clean_answer(a_en: str) -> str:
        a = strip_gsm_calc_markers(a_en)
        return strip_markdown(a)[: args.max_a_chars].strip()

    t0 = time.perf_counter()
    stats = dict(total=0, kept=0,
                 q_too_long=0, a_empty=0, hedge=0,
                 budget=0, wrong_eo=0)

    with out_path.open("a") as fout:
        for cs in tqdm(range(0, len(todo), args.chunk_size), desc="chunks"):
            chunk = todo[cs:cs+args.chunk_size]
            stats["total"] += len(chunk)
            # Pre-filter long Qs
            q_long = [len(q) > args.max_q_chars for _, q, _ in chunk]
            keep_idx = [j for j, m in enumerate(q_long) if not m]
            stats["q_too_long"] += len(chunk) - len(keep_idx)

            q_ens = [chunk[j][1] for j in keep_idx]
            a_ens_clean = [clean_answer(chunk[j][2]) for j in keep_idx]
            gold = [extract_final_number(chunk[j][2]) for j in keep_idx]

            # Hedge / empty filter
            ok = [bool(a) and not HEDGE_PATTERNS.search(a) for a in a_ens_clean]
            stats["a_empty"] += len(ok) - sum(ok)
            # Note: not separating hedge from empty — both treated as drop

            # Translate Qs (batched)
            q_eos = mt_translate(q_ens, desc="Q en→eo")

            # Translate As sentence-by-sentence (flatten, batch, rejoin)
            flat_sents, spans = [], []
            for j, (a, good) in enumerate(zip(a_ens_clean, ok)):
                if not good:
                    spans.append(None); continue
                sents = split_sentences(a) or [a]
                spans.append((len(flat_sents), len(flat_sents) + len(sents)))
                flat_sents.extend(sents)
            flat_eos = mt_translate(flat_sents, desc="A en→eo") if flat_sents else []

            # Reassemble + budget + answer check + write
            for j_kept, (idx_orig, _, a_en) in enumerate([chunk[j] for j in keep_idx]):
                global_i = chunk[keep_idx[j_kept]][0]
                if not ok[j_kept]:
                    fout.write(json.dumps({
                        "i": global_i, "skipped": True, "reason": "hedge_or_empty",
                        "q_en": q_ens[j_kept], "a_en": a_en,
                    }, ensure_ascii=False) + "\n")
                    continue
                a, b = spans[j_kept]
                a_eo = " ".join(flat_eos[a:b]).strip()
                q_eo = q_eos[j_kept]
                if stu(q_eo) + stu(a_eo) > args.max_student_tokens:
                    stats["budget"] += 1
                    fout.write(json.dumps({
                        "i": global_i, "skipped": True, "reason": "total_too_long",
                        "q_en": q_ens[j_kept], "q_eo": q_eo,
                        "a_en": a_en, "a_eo": a_eo,
                    }, ensure_ascii=False) + "\n")
                    continue
                if args.require_answer_match and gold[j_kept] is not None:
                    eo_pred = extract_final_number(strip_latex(a_eo))
                    if eo_pred != gold[j_kept]:
                        stats["wrong_eo"] += 1
                        fout.write(json.dumps({
                            "i": global_i, "skipped": True, "reason": "wrong_answer_eo",
                            "q_en": q_ens[j_kept], "q_eo": q_eo, "a_en": a_en,
                            "a_eo": a_eo, "gold": gold[j_kept], "eo_pred": eo_pred,
                        }, ensure_ascii=False) + "\n")
                        continue
                stats["kept"] += 1
                fout.write(json.dumps({
                    "i": global_i, "skipped": False,
                    "q_en": q_ens[j_kept], "q_eo": q_eo,
                    "a_en": a_en, "a_eo": a_eo, "gold": gold[j_kept],
                }, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"  chunk {cs//args.chunk_size+1} done — "
                  f"kept {stats['kept']}/{stats['total']} "
                  f"({stats['kept']/stats['total']*100:.1f}%)", flush=True)

    dt = time.perf_counter() - t0
    print(f"\n=== done in {dt:.0f}s ({stats['total']/dt:.1f} rows/s) ===")
    for k, v in stats.items():
        print(f"  {k:15s} {v:6d}  ({v/stats['total']*100:.1f}%)")


if __name__ == "__main__":
    main()
