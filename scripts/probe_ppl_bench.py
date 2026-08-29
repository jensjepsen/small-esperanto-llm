"""Per-category perplexity comparison with 50 passages each.

Loads N random samples of similar length from each HF source, computes
per-passage cross-entropy + ppl, reports mean / median / std per category.

Run on multiple ckpts back-to-back so means are directly comparable.

Usage:
    uv run python scripts/probe_ppl_bench.py \\
        --ckpts /mnt/data2/checkpoints/lm/v10_large/final \\
                /mnt/data2/checkpoints/lm/v11_h100/checkpoint-5000 \\
        --n 50
"""
import argparse
import math
import re
import sys
import statistics
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM

sys.path.insert(0, "src")
from esperanto_lm.data import load_tokenizer
from esperanto_lm.morphology import decompose


def morph_pp(text, tok):
    has_w = "<w>" in tok.get_vocab()
    words = re.findall(r"[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+|[^\s]", text)
    parts = []
    for w in words:
        if parts and has_w: parts.append("<w>")
        if w and w[0].isalpha(): parts.extend(decompose(w))
        else: parts.append(w)
    return parts


SOURCES = {
    # (repo, split, col, min_chars, max_chars, truncate_to)
    # Passages shorter than min_chars are skipped. Passages longer than
    # max_chars get truncated to truncate_to. Use this to normalize
    # passage length across sources without dropping long sources entirely.
    "wiki-hplt":   ("jensjepsen/esperanto-hplt-filtered", "train", "text", 400, 1200, 1200),
    "news":        ("jensjepsen/liberafolio",             "train", "content", 400, 1200, 1200),
    "gutenberg":   ("jensjepsen/esperanto-gutenberg",     "train", "text", 400, 1200, 1200),
    "factoids":    ("jensjepsen/esperanto-factoids",      "train", "text", 100, 1200, 1200),
    "algebra":     ("jensjepsen/esperanto-algebra-pretrain","train", "text", 30, 800, 800),
}


def sample_passages(source, n, seed=42):
    """Pull `n` random passages within a per-source length window."""
    import os
    repo, split, col, min_chars, max_chars, truncate_to = source
    print(f"  loading {repo}…", flush=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if not token:
        try:
            token = (Path.home() / ".cache/huggingface/token").read_text().strip()
        except Exception:
            token = None
    ds = load_dataset(repo, split=split, streaming=False, token=token)
    if col not in ds.column_names:
        col = ds.column_names[0]
    import random
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    passages = []
    for i in idxs:
        t = ds[i][col]
        if not isinstance(t, str) or len(t) < min_chars: continue
        if len(t) > max_chars:
            if truncate_to is not None:
                passages.append(t[:truncate_to])
            # else: skip
        else:
            passages.append(t)
        if len(passages) >= n:
            break
    return passages[:n]


@torch.no_grad()
def ppl_one(model, tok, text):
    morphs = morph_pp(text, tok)
    ids = tok(" ".join(morphs), return_tensors="pt",
              add_special_tokens=False).input_ids.cuda()
    if ids.shape[1] < 4: return None
    out = model(ids, labels=ids)
    return out.loss.item()  # per-token mean CE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    print(f"sampling {args.n} passages per source…", flush=True)
    cat_passages = {}
    for cat, src in SOURCES.items():
        try:
            cat_passages[cat] = sample_passages(src, args.n)
            ps = cat_passages[cat]
            if ps:
                print(f"    {cat:14s}  n={len(ps)}  avg_chars={sum(len(p) for p in ps)//len(ps)}",
                      flush=True)
            else:
                print(f"    {cat:14s}  NO PASSAGES FIT WINDOW", flush=True)
        except Exception as e:
            print(f"    {cat:14s}  FAILED: {e}", flush=True)
            cat_passages[cat] = []

    tok = load_tokenizer(Path("tokenizer_morpheme"))
    results = {}
    for ckpt in args.ckpts:
        ckpt_name = Path(ckpt).parent.name + "/" + Path(ckpt).name
        print(f"\n=== {ckpt_name} ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda().eval()
        results[ckpt_name] = {}
        for cat, passages in cat_passages.items():
            losses = []
            for p in passages:
                v = ppl_one(model, tok, p)
                if v is not None: losses.append(v)
            if not losses:
                print(f"  {cat:14s}  (no data)"); continue
            mean = statistics.mean(losses)
            median = statistics.median(losses)
            stdev = statistics.stdev(losses) if len(losses) > 1 else 0.0
            results[ckpt_name][cat] = (mean, median, stdev, len(losses))
            print(f"  {cat:14s}  loss mean={mean:.3f} median={median:.3f} std={stdev:.3f}  "
                  f"ppl_mean={math.exp(mean):.2f}  (n={len(losses)})")
        del model
        torch.cuda.empty_cache()

    # side-by-side summary
    if len(args.ckpts) >= 2:
        print(f"\n{'='*70}")
        print("side-by-side (mean ppl)")
        print(f"{'='*70}")
        names = list(results.keys())
        header = f"{'category':14s}  " + "  ".join(f"{n[:24]:>24s}" for n in names) + "  Δ(B-A)"
        print(header)
        for cat in SOURCES:
            row = f"{cat:14s}  "
            vals = []
            for n in names:
                v = results[n].get(cat)
                if v: vals.append(math.exp(v[0])); row += f"{math.exp(v[0]):>24.2f}  "
                else: vals.append(None); row += f"{'-':>24s}  "
            if len(vals) == 2 and vals[0] and vals[1]:
                row += f"{vals[1]-vals[0]:+.2f}"
            print(row)


if __name__ == "__main__":
    main()
