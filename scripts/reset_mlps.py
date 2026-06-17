"""Reset MLP weights of a Llama checkpoint while preserving attention,
embeddings, layer norms, and lm_head.

Experiment: given pretrained attention patterns + embeddings, can SFT
re-grow the MLPs from scratch? Tests whether attention's structural
priors are sufficient, or whether the pretrained MLPs are doing the
real lifting.

Re-initializes gate_proj, up_proj, down_proj in every layer using the
same N(0, initializer_range) Llama uses for fresh weights. Saves as a
new checkpoint dir.

Usage:
    uv run python scripts/reset_mlps.py \\
        --src runs/large/checkpoint-44000 \\
        --dst runs/large/checkpoint-44000-mlp-reset
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source checkpoint dir")
    ap.add_argument("--dst", required=True, help="Output checkpoint dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading {args.src} ...")
    model = AutoModelForCausalLM.from_pretrained(args.src, torch_dtype=torch.float32)
    cfg = model.config
    std = cfg.initializer_range
    print(f"Re-initializing {cfg.num_hidden_layers} layers' MLPs "
          f"with N(0, {std}) ...")

    mlp_projs = ("gate_proj", "up_proj", "down_proj")

    before_stats = {}
    after_stats = {}
    n_reset = 0
    for i, layer in enumerate(model.model.layers):
        for name in mlp_projs:
            proj = getattr(layer.mlp, name)
            w = proj.weight
            if i == 0:
                before_stats[name] = (w.mean().item(), w.std().item())
            with torch.no_grad():
                w.normal_(mean=0.0, std=std)
                if proj.bias is not None:
                    proj.bias.zero_()
            if i == 0:
                after_stats[name] = (w.mean().item(), w.std().item())
            n_reset += w.numel() + (proj.bias.numel() if proj.bias is not None else 0)

    print(f"Reset {n_reset:,} params across {cfg.num_hidden_layers} layers.")
    print(f"Layer-0 weight stats (mean, std):")
    for name in mlp_projs:
        b = before_stats[name]
        a = after_stats[name]
        print(f"  {name:12s} before=({b[0]:+.4f}, {b[1]:.4f})  "
              f"after=({a[0]:+.4f}, {a[1]:.4f})")

    # Sanity-check: attention + embeddings + lm_head untouched.
    untouched = {
        "embed_tokens": model.model.embed_tokens.weight,
        "lm_head":      model.lm_head.weight,
        "layer-0 q_proj": model.model.layers[0].self_attn.q_proj.weight,
        "layer-0 k_proj": model.model.layers[0].self_attn.k_proj.weight,
        "layer-0 v_proj": model.model.layers[0].self_attn.v_proj.weight,
        "layer-0 o_proj": model.model.layers[0].self_attn.o_proj.weight,
        "layer-0 input_layernorm": model.model.layers[0].input_layernorm.weight,
    }
    print("Preserved weights — sample mean/std (should look 'trained', not init):")
    for k, w in untouched.items():
        print(f"  {k:30s} mean={w.mean().item():+.4f}  std={w.std().item():.4f}")

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {dst} ...")
    model.save_pretrained(dst)
    print("Done.")


if __name__ == "__main__":
    main()
