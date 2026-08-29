"""Resumable EN→EO translation of MetaMathQA — pod flavor.

Differences from `translate_metamath_local.py`:
  * Pulls the v11 model + SPM tokenizer + custom tokenizer class from HF Hub
    on first run (private repo: needs `HUGGING_FACE_HUB_TOKEN` or `huggingface-cli login`).
  * FP16 inference (5090/Blackwell has real fp16 tensor cores).
  * L2T-based ASCII math prep replaces the LaTeX with ASCII math the model
    already knows — cleans up `\\boxed`, `\\dbinom`, `\\sqrt`, `\\frac`, `\\neq`,
    `\\infty`, and everything else without needing sentinel tokens or a
    fine-tune.
  * Larger default batch size (192) — 32GB VRAM budget.
  * Same output schema as the local script so shards from either can concat.

On a fresh pod:
  bash scripts/setup_vastai.sh          # torch+cu128, SPM tokenizer, sanity checks
  huggingface-cli login                 # for the private v11 repo
  uv run python scripts/translate_metamath_pod.py \\
    --output /workspace/metamath_eo.jsonl \\
    --batch-size 192 --chunk-rows 768
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from pylatexenc.latex2text import LatexNodes2Text
from transformers import MarianMTModel

REPO = "jensjepsen/eo-mt-v11"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
EN_CHARS_PER_EO_TOK = 2.535  # calibrated on orca-math parallel

# ── LaTeX → ASCII math ──────────────────────────────────────────────────

L2T = LatexNodes2Text()
DBINOM_FIX = re.compile(r"\\d?binom\{([^{}]*)\}\{([^{}]*)\}")

# L2T outputs some math Unicode we want to fold to ASCII the LM was trained on.
UNICODE_TO_ASCII = {
    "√": "sqrt", "≠": "!=", "≥": ">=", "≤": "<=", "∞": "inf",
    "ℝ": "R", "ℕ": "N", "ℤ": "Z", "ℚ": "Q", "ℂ": "C",
    "∈": " in ", "∉": " notin ", "·": "*", "×": "*", "÷": "/",
    "…": "...", "→": "->", "±": "+-",
    "π": "pi", "α": "alpha", "β": "beta", "θ": "theta", "λ": "lambda",
    "²": "^2", "³": "^3",
}


def latex_to_ascii(text: str) -> str:
    if not text:
        return text
    # L2T juxtaposes binomial args ("165") — pre-substitute to keep them parseable.
    text = DBINOM_FIX.sub(r"C(\1,\2)", text)
    text = L2T.latex_to_text(text)
    for k, v in UNICODE_TO_ASCII.items():
        text = text.replace(k, v)
    return re.sub(r"\s+", " ", text).strip()


def sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


# ── Model bootstrap ─────────────────────────────────────────────────────


def fetch_model(cache_dir: Path) -> tuple[Path, Path]:
    """Download model + tokenizer + sp_tokenizer.py from HF Hub. Returns
    (model_dir, tokenizer_file)."""
    print(f"[fetch] snapshot {REPO} → {cache_dir}", flush=True)
    local = Path(snapshot_download(REPO, local_dir=cache_dir))
    tok_file = local / "tokenizer" / "spm_eneo_48k_v2.model"
    assert tok_file.exists(), f"tokenizer not in snapshot at {tok_file}"
    return local, tok_file


def load_translator(model_dir: Path, tok_file: Path, device: str, dtype: torch.dtype):
    # The custom SPM wrapper lives in the repo as `sp_tokenizer.py`. Add the
    # repo dir to sys.path so `from sp_tokenizer import SPMTokenizer` resolves
    # against the downloaded file.
    sys.path.insert(0, str(model_dir))
    from sp_tokenizer import SPMTokenizer  # type: ignore

    tok = SPMTokenizer(str(tok_file))
    model = MarianMTModel.from_pretrained(str(model_dir), torch_dtype=dtype).to(device).eval()
    return tok, model


# ── I/O helpers ─────────────────────────────────────────────────────────


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
    print(f"[source] materializing MetaMathQA → {src_path}", flush=True)
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
    print(f"[source] wrote {len(ds):,} rows", flush=True)


class Chunker:
    def __init__(self, src_path: Path, done: set[int], chunk_rows: int, max_en_chars: int):
        self.src_path = src_path
        self.done = done
        self.chunk_rows = chunk_rows
        self.max_en_chars = max_en_chars
        self.n_skip_done = 0
        self.n_skip_long = 0

    def __iter__(self):
        buf: list[dict] = []
        with self.src_path.open() as f:
            for line in f:
                r = json.loads(line)
                if r["orig_idx"] in self.done:
                    self.n_skip_done += 1
                    continue
                if len(r["query"]) + 1 + len(r["response"]) > self.max_en_chars:
                    self.n_skip_long += 1
                    continue
                buf.append(r)
                if len(buf) >= self.chunk_rows:
                    yield buf
                    buf = []
        if buf:
            yield buf


# ── Translation core ────────────────────────────────────────────────────


def raw_translate_batch(tok, model, device: str, dtype: torch.dtype, srcs: list[str],
                        max_input_tokens: int, max_output_tokens: int) -> list[str]:
    if not srcs:
        return []
    ids_list = [tok.encode(s, lang="eo")[:max_input_tokens] for s in srcs]
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    sorted_ids = [ids_list[i] for i in order]
    be = tok.pad_batch(sorted_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids=be.input_ids.to(device),
            attention_mask=be.attention_mask.to(device),
            max_length=max_output_tokens,
            do_sample=False,
            num_beams=1,
        )
    decoded_sorted = [tok.decode(out[i]) for i in range(len(sorted_ids))]
    results = [""] * len(srcs)
    for sp, op in enumerate(order):
        results[op] = decoded_sorted[sp]
    return results


def translate_chunk(rows: list[dict], tok, model, device, dtype, bs, max_in, max_out) -> list[dict]:
    tasks: list[tuple[int, str, int, str]] = []
    prepped_by_row: list[tuple[str, str]] = []
    for ri, r in enumerate(rows):
        q_ascii = latex_to_ascii(r["query"])
        a_ascii = latex_to_ascii(r["response"])
        prepped_by_row.append((q_ascii, a_ascii))
        for si, s in enumerate(sentences(q_ascii)):
            tasks.append((ri, "q", si, s))
        for si, s in enumerate(sentences(a_ascii)):
            tasks.append((ri, "a", si, s))

    unique_map: dict[str, str] = {}
    for _, _, _, s in tasks:
        unique_map.setdefault(s, "")
    uniques = list(unique_map.keys())

    for j in range(0, len(uniques), bs):
        chunk = uniques[j : j + bs]
        for src, dst in zip(chunk, raw_translate_batch(tok, model, device, dtype, chunk, max_in, max_out)):
            unique_map[src] = dst

    per_row: dict[int, dict[str, dict[int, str]]] = {i: {"q": {}, "a": {}} for i in range(len(rows))}
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


# ── Driver ──────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="/workspace/metamath_eo.jsonl")
    ap.add_argument("--source", default="/workspace/metamath_en.jsonl")
    ap.add_argument("--model-cache", default="/workspace/models/eo-mt-v11")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--chunk-rows", type=int, default=1024)
    ap.add_argument("--max-eo-tokens", type=int, default=512)
    ap.add_argument("--max-input-tokens", type=int, default=500)
    # Sentence-level output is short; capping the decode loop at 160 tokens
    # halves the wall time vs 256 with negligible truncation (99% of MetaMath
    # sentences emit <128 tokens; keeping headroom for outliers).
    ap.add_argument("--max-output-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src_path = Path(args.source)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    print(f"[gpu] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"[dtype] {args.dtype}", flush=True)

    materialize_source(src_path)
    done = load_done(out_path)
    print(f"[resume] already done: {len(done):,}", flush=True)

    model_dir, tok_file = fetch_model(Path(args.model_cache))
    tok, model = load_translator(model_dir, tok_file, "cuda", dtype)

    max_en_chars = int(args.max_eo_tokens * EN_CHARS_PER_EO_TOK)
    print(f"[filter] skip EN>{max_en_chars} chars (≈{args.max_eo_tokens} EO tokens)", flush=True)

    stop = {"flag": False}

    def _handle(signum, frame):
        if stop["flag"]:
            print("\n[double-interrupt] hard exit", flush=True)
            os._exit(1)
        print(f"\n[signal {signum}] finishing current chunk then stopping", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    chunker = Chunker(src_path, done, args.chunk_rows, max_en_chars)
    total = 0
    t0 = time.time()

    with out_path.open("a", buffering=1) as fout:
        for chunk in chunker:
            if stop["flag"]:
                break
            results = translate_chunk(chunk, tok, model, "cuda", dtype,
                                      args.batch_size, args.max_input_tokens, args.max_output_tokens)
            for row in results:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(results)
            el = time.time() - t0
            rate = total / max(el, 1e-6)
            print(
                f"[{total + len(done):>6,} written]"
                f" {rate:.2f} rows/s"
                f"  skip_done={chunker.n_skip_done:,}"
                f"  skip_long={chunker.n_skip_long:,}",
                flush=True,
            )
            if args.limit and total >= args.limit:
                print(f"[limit {args.limit} reached] stopping", flush=True)
                break

    el = time.time() - t0
    print(f"\n[done] wrote {total:,} rows in {el/60:.1f} min ({total / max(el, 1e-6):.2f} rows/s)", flush=True)
    print(f"[done] skipped_done={chunker.n_skip_done:,}  skipped_over={chunker.n_skip_long:,}", flush=True)


if __name__ == "__main__":
    main()
