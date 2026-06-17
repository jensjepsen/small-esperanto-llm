"""Build a small MarianMT encoder-decoder.

Defaults: 6+6 layers, d_model=512, ffn=2048, 8 heads. ~70M params
with tied input/output embeddings on a 32k joint vocab.
"""
from __future__ import annotations

import argparse

from transformers import MarianConfig, MarianMTModel


def build_model(
    vocab_size: int,
    d_model: int = 512,
    encoder_layers: int = 6,
    decoder_layers: int = 6,
    encoder_attention_heads: int = 8,
    decoder_attention_heads: int = 8,
    encoder_ffn_dim: int = 2048,
    decoder_ffn_dim: int = 2048,
    max_position_embeddings: int = 512,
    dropout: float = 0.1,
    pad_token_id: int = 0,
    bos_token_id: int = 2,
    eos_token_id: int = 3,
    decoder_start_token_id: int | None = None,
) -> MarianMTModel:
    cfg = MarianConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        encoder_attention_heads=encoder_attention_heads,
        decoder_attention_heads=decoder_attention_heads,
        encoder_ffn_dim=encoder_ffn_dim,
        decoder_ffn_dim=decoder_ffn_dim,
        max_position_embeddings=max_position_embeddings,
        dropout=dropout,
        activation_function="gelu",
        share_encoder_decoder_embeddings=True,
        tie_word_embeddings=True,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        decoder_start_token_id=decoder_start_token_id if decoder_start_token_id is not None else pad_token_id,
        forced_eos_token_id=eos_token_id,
        scale_embedding=True,
    )
    return MarianMTModel(cfg)


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def param_summary(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {_fmt(total)} ({total:,})")
    print(f"Trainable params: {_fmt(trainable)} ({trainable:,})")
    by_module = {}
    for name, p in model.named_parameters():
        top = name.split(".")[1] if "." in name else name
        by_module[top] = by_module.get(top, 0) + p.numel()
    for k in sorted(by_module, key=lambda k: -by_module[k]):
        print(f"  {k:30s} {_fmt(by_module[k]):>10s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--encoder-layers", type=int, default=6)
    ap.add_argument("--decoder-layers", type=int, default=6)
    ap.add_argument("--ffn-dim", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()

    model = build_model(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_attention_heads=args.heads,
        decoder_attention_heads=args.heads,
        encoder_ffn_dim=args.ffn_dim,
        decoder_ffn_dim=args.ffn_dim,
    )
    param_summary(model)


if __name__ == "__main__":
    main()
