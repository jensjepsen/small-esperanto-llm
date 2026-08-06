"""Sanity check: does NTK-scaled RoPE extension work on the DA base
without any continued pretraining?

Strategy:
  - Load `jensjepsen/danish-lm-400m-base-ckpt310k` two ways:
      A) vanilla (max_position_embeddings=512, no rope_scaling)
      B) NTK-dynamic (max_position_embeddings=2048, factor=4)
  - Feed each a long (~2048-token) Danish text.
  - For A: only positions 0..511 are valid (model would crash beyond).
    Compute per-position-bucket NLL over positions 0..511 as the reference.
  - For B: compute per-position-bucket NLL over positions 0..2047.
  - Report NLL by position bucket. If NTK works well, B's NLL in the
    0..511 bucket should be close to A's (i.e., NTK didn't degrade the
    trained range), and NLL in higher buckets should not blow up.

If B's NLL at high positions is only ~1.2-1.5× the trained-range NLL,
NTK-at-inference is usable. If it's 3-10×, we need proper continued
pretraining to extend context.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

BASE = "jensjepsen/danish-lm-400m-base-ckpt310k"
BUCKETS = [(0, 256), (256, 512), (512, 1024), (1024, 1536), (1536, 2048)]


@torch.no_grad()
def per_position_nll(model, input_ids):
    """Return a 1-D tensor of NLL for each output position (len = seq_len-1)."""
    logits = model(input_ids=input_ids).logits[0]  # [seq, vocab]
    shift_logits = logits[:-1]                     # predict [1:]
    shift_labels = input_ids[0, 1:]
    return F.cross_entropy(shift_logits, shift_labels, reduction="none")


def bucketed_nll(nll: torch.Tensor, buckets):
    """Return {bucket_label: mean_nll} for each (lo, hi) bucket."""
    out = {}
    for lo, hi in buckets:
        if lo >= nll.shape[0]:
            continue
        segment = nll[lo:min(hi, nll.shape[0])]
        out[f"[{lo:>4d},{min(hi, nll.shape[0]):>4d})"] = segment.mean().item()
    return out


def load_vanilla():
    cfg = AutoConfig.from_pretrained(BASE)
    assert cfg.max_position_embeddings == 512
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).cuda().eval()
    return model


def load_ntk(factor=4.0):
    cfg = AutoConfig.from_pretrained(BASE)
    cfg.max_position_embeddings = 512 * int(factor)
    cfg.rope_scaling = {"type": "dynamic", "factor": factor}
    model = AutoModelForCausalLM.from_pretrained(
        BASE, config=cfg, torch_dtype=torch.float16, ignore_mismatched_sizes=False,
    ).cuda().eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", default=[
        "/tmp/da_probe_doc0.txt", "/tmp/da_probe_doc1.txt", "/tmp/da_probe_doc2.txt",
    ])
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--tokenizer", default="jensjepsen/danish-tokenizer")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Build a single long token sequence by concatenating docs until we hit
    # the target length. Simpler than sliding-window; probe is a smoke check
    # so a couple of long inputs is enough.
    texts = [Path(p).read_text() for p in args.docs]
    joined = "\n\n".join(texts)
    ids = tok(joined, add_special_tokens=False)["input_ids"][:args.tokens]
    if len(ids) < args.tokens:
        print(f"WARN: only {len(ids)} tokens available, wanted {args.tokens}", file=sys.stderr)
    input_ids = torch.tensor([ids], dtype=torch.long).cuda()
    print(f"sequence length: {input_ids.shape[1]} tokens", flush=True)

    # ── A: vanilla model, positions 0..511 ────────────────────────────────
    print(f"\nloading vanilla {BASE} (max_position=512) …", flush=True)
    model_a = load_vanilla()
    ids_512 = input_ids[:, :512]
    nll_a = per_position_nll(model_a, ids_512)
    print(f"  positions covered: 0..{nll_a.shape[0]}")
    a_bucketed = bucketed_nll(nll_a, [(0, 256), (256, 512)])
    del model_a
    torch.cuda.empty_cache()

    # ── B: NTK-extended model, positions 0..2047 ──────────────────────────
    print(f"\nloading NTK-scaled {BASE} (max_position=2048, dyn factor=4) …",
          flush=True)
    model_b = load_ntk(factor=4.0)
    nll_b = per_position_nll(model_b, input_ids)
    b_bucketed = bucketed_nll(nll_b, BUCKETS)
    del model_b
    torch.cuda.empty_cache()

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"vanilla model (positions 0..511):")
    for k, v in a_bucketed.items():
        print(f"  {k}  NLL={v:.4f}  ppl={torch.exp(torch.tensor(v)).item():.2f}")

    print(f"\nNTK-extended model (positions 0..2047):")
    ref = None
    for k, v in b_bucketed.items():
        ratio = ""
        if ref is None:
            ref = v
        else:
            ratio = f"  (×{v/ref:.2f} vs [0,256))"
        print(f"  {k}  NLL={v:.4f}  ppl={torch.exp(torch.tensor(v)).item():.2f}{ratio}")

    # Sanity delta: NTK's [0,256) NLL vs vanilla's — should be nearly identical.
    a0 = a_bucketed.get("[   0, 256)")
    b0 = b_bucketed.get("[   0, 256)")
    if a0 is not None and b0 is not None:
        print(f"\nNTK degradation on trained range [0,256): "
              f"Δ NLL = {b0 - a0:+.4f}  ({(b0/a0 - 1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
