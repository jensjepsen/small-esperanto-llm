"""Generic multi-dataset EN->EO->EN round-trip with LaBSE + chrF scoring.

Handles multiple HF datasets with configurable field mappings. Preserves
ALL original fields; adds `_eo`, `_roundtrip`, `_chrf`, `_cos_sim` per
translated field. Writes JSONL incrementally so we can resume on
interrupt.

Datasets are configured in a REGISTRY at the top. Each config specifies
which HF fields to translate and any per-dataset preproc (like splitting
the `input` field in e-SNLI).

Output: one JSONL per (dataset, split) at
`{output_dir}/{name}_{split}.jsonl`. Resumes by index.
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
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, MarianMTModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mt" / "scripts"))
from sp_tokenizer import SPMTokenizer  # type: ignore  # noqa: E402


def _split_esnli_input(row):
    """Split '<premise></s><hypothesis>' into two fields on the row."""
    parts = row["input"].split("</s>")
    row["premise"] = parts[0].strip()
    row["hypothesis"] = parts[1].strip() if len(parts) > 1 else ""
    return row


REGISTRY = {
    "ecqa": {
        "repo": "yangdong/ecqa",
        "splits": ["train", "validation", "test"],
        "translate_fields": [
            "q_text", "q_op1", "q_op2", "q_op3", "q_op4", "q_op5",
            "q_ans", "taskA_pos", "taskA_neg", "taskB",
        ],
        "keep_fields": ["q_no", "q_concept"],
    },
    "ecare": {
        "repo": "12ml/e-CARE",
        "splits": ["train", "validation"],
        "translate_fields": [
            "premise", "choice1", "choice2", "conceptual_explanation",
        ],
        "keep_fields": ["idx", "question", "label"],
    },
    "esnli": {
        "repo": "pwei07/esnli",
        "splits": ["train", "valid", "test"],
        "preproc": _split_esnli_input,
        "translate_fields": ["premise", "hypothesis", "rationale"],
        "keep_fields": ["label", "llm_label"],
    },
}


def batch_translate(model, tok, srcs, tgt_lang,
                    max_input=500, max_output=256, sub_batch=64):
    """Sort by length, split into fixed-size sub-batches for GPU efficiency.

    A single giant model.generate on 1000+ sequences with max_length=256
    KV-caches every seq at every step -> 5-10x slower than batched.
    """
    if not srcs:
        return []
    nonempty = [(i, s) for i, s in enumerate(srcs) if s and s.strip()]
    outs = [""] * len(srcs)
    if not nonempty:
        return outs
    ids_list = [tok.encode(s, lang=tgt_lang)[:max_input] for _, s in nonempty]
    # Sort by length across ALL sub-batches so each sub-batch is homogeneous.
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    sorted_ids = [ids_list[i] for i in order]

    decoded_sorted: list[str] = []
    for start in range(0, len(sorted_ids), sub_batch):
        chunk = sorted_ids[start:start + sub_batch]
        be = tok.pad_batch(chunk)
        with torch.no_grad():
            out = model.generate(
                input_ids=be.input_ids.cuda(),
                attention_mask=be.attention_mask.cuda(),
                max_length=max_output,
                do_sample=False,
                num_beams=1,
            )
        decoded_sorted.extend(tok.decode(out[i]) for i in range(len(chunk)))

    for sp, op in enumerate(order):
        outs[nonempty[op][0]] = decoded_sorted[sp]
    return outs


def load_done(out_path: Path) -> set[int]:
    done: set[int] = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["_row_idx"])
            except Exception:
                continue
    return done


def process_dataset(cfg_name, cfg, split, args, model, tok, lt, lm):
    """Translate one dataset+split. Streams rows, writes JSONL incrementally."""
    out_path = Path(args.output_dir) / f"{cfg_name}_{split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)

    print(f"\n[{cfg_name}/{split}] loading...", flush=True)
    ds = load_dataset(cfg["repo"], split=split)
    if "preproc" in cfg:
        ds = ds.map(cfg["preproc"])
    total = len(ds)
    remaining = total - len(done)
    print(f"[{cfg_name}/{split}] {total:,} rows total, {len(done):,} done, "
          f"{remaining:,} remaining", flush=True)
    if remaining == 0:
        return

    trans_fields = cfg["translate_fields"]
    keep_fields = cfg.get("keep_fields", [])

    @torch.no_grad()
    def embed(texts, bs=64):
        # tuple out empties
        nonempty_idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if not nonempty_idx:
            return torch.zeros(len(texts), 768)
        pool = [texts[i] for i in nonempty_idx]
        E = []
        for i in range(0, len(pool), bs):
            enc = lt(pool[i:i + bs], padding=True, truncation=True,
                     max_length=128, return_tensors="pt").to("cuda")
            e = lm(**enc).last_hidden_state[:, 0]
            e = torch.nn.functional.normalize(e, dim=1)
            E.append(e.float().cpu())
        E = torch.cat(E, 0)
        # scatter back
        full = torch.zeros(len(texts), E.shape[1])
        for j, oi in enumerate(nonempty_idx):
            full[oi] = E[j]
        return full

    t0 = time.time()
    processed = 0
    with out_path.open("a", buffering=1) as fout:
        for start in range(0, total, args.batch_rows):
            end = min(start + args.batch_rows, total)
            batch_rows = []
            for i in range(start, end):
                if i in done:
                    continue
                batch_rows.append((i, ds[i]))
            if not batch_rows:
                continue

            # Flatten (row_idx, field_name, text) tuples for batched translation
            flat = []
            for row_idx, row in batch_rows:
                for f in trans_fields:
                    flat.append((row_idx, f, row.get(f, "") or ""))

            en_texts = [t[2] for t in flat]
            # EN -> EO
            eo_texts = batch_translate(model, tok, en_texts, "eo",
                                        max_input=args.max_input,
                                        max_output=args.max_output,
                                        sub_batch=args.sub_batch)
            # EO -> EN back
            back_texts = batch_translate(model, tok, eo_texts, "en",
                                          max_input=args.max_input,
                                          max_output=args.max_output,
                                          sub_batch=args.sub_batch)

            # LaBSE cos_sim
            e_en = embed(en_texts)
            e_back = embed(back_texts)
            cos = (e_en * e_back).sum(1).tolist()

            # chrF per translation
            chrfs = [
                sacrebleu.sentence_chrf(b, [e]).score if e else 0.0
                for e, b in zip(en_texts, back_texts)
            ]

            # Reassemble by row
            per_row = {r: {} for r, _, _ in flat}
            for (row_idx, f, en), eo, back, c, cf in zip(
                    flat, eo_texts, back_texts, cos, chrfs):
                per_row[row_idx][f] = {
                    "eo": eo,
                    "roundtrip": back,
                    "cos_sim": float(c),
                    "chrf": float(cf),
                }

            for row_idx, row in batch_rows:
                out_row = {"_row_idx": row_idx, "_split": split,
                           "_source": cfg["repo"]}
                # keep all original fields
                for k, v in row.items():
                    out_row[k] = v
                # add translations
                for f in trans_fields:
                    t = per_row[row_idx].get(f)
                    if t is None:
                        continue
                    out_row[f + "_eo"] = t["eo"]
                    out_row[f + "_roundtrip"] = t["roundtrip"]
                    out_row[f + "_cos_sim"] = t["cos_sim"]
                    out_row[f + "_chrf"] = t["chrf"]
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            processed += len(batch_rows)
            el = time.time() - t0
            rate = processed / max(el, 1e-6)
            eta_min = (remaining - processed) / max(rate, 1e-6) / 60
            print(f"  [{cfg_name}/{split}] {start + len(batch_rows):>6,}/{total:,}  "
                  f"{rate:.1f} rows/s  ETA {eta_min:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(REGISTRY.keys()),
                    choices=list(REGISTRY.keys()))
    ap.add_argument("--batch-rows", type=int, default=64,
                    help="Rows per outer batch (each row has multiple fields)")
    ap.add_argument("--sub-batch", type=int, default=128,
                    help="Max sequences per model.generate call")
    ap.add_argument("--max-input", type=int, default=500)
    ap.add_argument("--max-output", type=int, default=256)
    args = ap.parse_args()

    print(f"[model] loading {args.checkpoint}", flush=True)
    tok = SPMTokenizer(args.tokenizer)
    model = MarianMTModel.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16).cuda().eval()

    print("[labse] loading sentence-transformers/LaBSE", flush=True)
    lt = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    lm = AutoModel.from_pretrained("sentence-transformers/LaBSE").cuda().eval().half()

    stop = {"flag": False}

    def _handle(signum, frame):
        if stop["flag"]:
            os._exit(1)
        print(f"\n[signal {signum}] stopping after current batch", flush=True)
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    for name in args.datasets:
        cfg = REGISTRY[name]
        for split in cfg["splits"]:
            if stop["flag"]:
                return
            process_dataset(name, cfg, split, args, model, tok, lt, lm)


if __name__ == "__main__":
    main()
