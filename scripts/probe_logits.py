"""Probe what a checkpoint 'knows' via top-k next-token logits on cloze prompts.

Mirrors the morpheme preprocessing in scripts/generate.py so prompts hit the
tokenizer the same way training data does. For each prompt:
  - Show top-k next-morpheme distribution (with probability)
  - Greedy-extend N more tokens to show the surface-form continuation

Useful for commonsense / factual probing without the noise of full sampling.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from esperanto_lm.data import load_tokenizer
from esperanto_lm.morphology import decompose


# Default cloze prompts. Each one isolates a single piece of commonsense /
# factual knowledge whose continuation should be near-deterministic.
DEFAULT_PROMPTS = [
    ("Pluvis, do li alportis",                "ombrelon"),
    ("La patro de mia patro estas mia",       "avo"),
    ("La suno leviĝas en la",                 "oriento"),
    ("Birdoj povas",                          "flugi"),
    ("Glacio estas frosta, sed fajro estas",  "varma"),
    ("Se mi estas malsata, mi",               "manĝas"),
    ("Hundoj diras boj-boj, kaj katoj diras", "miaŭ"),
    ("Tri plus du estas",                     "kvin"),
    ("Post lundo venas",                      "mardo"),
    ("La ĉefurbo de Francio estas",           "Parizo"),
]


def morpheme_preprocess(text: str, tokenizer) -> str:
    """Same preprocessing as scripts/generate.py — splits words to morphemes
    with <w> boundaries when the tokenizer supports them."""
    has_w_token = "<w>" in tokenizer.get_vocab()
    words = re.findall(r"[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+|[^\s]", text)
    parts = []
    for word in words:
        if parts and has_w_token:
            parts.append("<w>")
        if word and word[0].isalpha():
            parts.extend(decompose(word))
        else:
            parts.append(word)
    return " ".join(parts)


@torch.no_grad()
def _topk_at(model, ctx_ids, top_k: int) -> list[tuple[str, float, int]]:
    """Return [(token_str, prob, vocab_id)] for top-k next tokens given context."""
    out = model(input_ids=ctx_ids)
    probs = torch.softmax(out.logits[0, -1], dim=-1)
    top_p, top_i = probs.topk(top_k)
    return list(zip(top_p.tolist(), top_i.tolist()))


@torch.no_grad()
def _logp_sequence(model, ctx_ids, target_ids: list[int]) -> list[float]:
    """Return per-token log-prob of `target_ids` continuing from `ctx_ids`.

    Single forward pass: feed ctx + target, read off log-probs of each target
    position from the corresponding output position.
    """
    full = torch.cat([ctx_ids, torch.tensor([target_ids], device=ctx_ids.device)],
                     dim=1)
    out = model(input_ids=full)
    logp = torch.log_softmax(out.logits[0], dim=-1)
    # Position i of logits predicts token at i+1, so target token j sits at
    # position (ctx_len - 1 + j) in the logits.
    ctx_len = ctx_ids.size(1)
    return [float(logp[ctx_len - 1 + j, tid]) for j, tid in enumerate(target_ids)]


@torch.no_grad()
def probe(model, tokenizer, device, prompt: str, expected: str,
          top_k: int, extend: int):
    pp = morpheme_preprocess(prompt, tokenizer)
    inputs = tokenizer(pp, return_tensors="pt",
                       return_token_type_ids=False).to(device)
    ctx = inputs.input_ids

    # (1) Top-k at the raw last position — usually dominated by <w>.
    raw_top = _topk_at(model, ctx, top_k)

    # (2) Append <w> deterministically (if it's the natural next token) and
    # show top-k of the *content* morpheme that follows. Most informative.
    w_id = tokenizer.convert_tokens_to_ids("<w>")
    if w_id is not None and w_id != tokenizer.unk_token_id:
        ctx_after_w = torch.cat(
            [ctx, torch.tensor([[w_id]], device=device)], dim=1)
        post_w_top = _topk_at(model, ctx_after_w, top_k)
    else:
        ctx_after_w = ctx
        post_w_top = raw_top

    # (3) Greedy extension to show surface continuation.
    extension_ids: list[int] = []
    cur = ctx
    for _ in range(extend):
        o = model(input_ids=cur)
        nid = int(o.logits[0, -1].argmax())
        extension_ids.append(nid)
        cur = torch.cat([cur, torch.tensor([[nid]], device=device)], dim=1)
    extension = tokenizer.decode(extension_ids, skip_special_tokens=False)

    # (4) If expected answer given, score it: tokenize via the same morpheme
    # preprocessor (with leading <w> so it joins onto the prompt naturally),
    # then compute per-token log-probs.
    expected_score = None
    if expected:
        # Preprocess as if it were a continuation — leading <w> is what would
        # naturally separate it from the prompt's last word.
        exp_pp = morpheme_preprocess(expected, tokenizer)
        if w_id is not None and not exp_pp.startswith("<w>"):
            exp_pp = "<w> " + exp_pp
        exp_ids = tokenizer(exp_pp, add_special_tokens=False)["input_ids"]
        per_tok_logp = _logp_sequence(model, ctx, exp_ids)
        exp_tokens = tokenizer.convert_ids_to_tokens(exp_ids)

        # Rank of the *first content morpheme* (after <w>) in post-<w> dist.
        first_content_id = exp_ids[1] if (exp_ids and exp_ids[0] == w_id
                                          and len(exp_ids) > 1) else exp_ids[0]
        out = model(input_ids=ctx_after_w)
        logits_post = out.logits[0, -1]
        # rank = number of tokens with strictly higher logit
        rank = int((logits_post > logits_post[first_content_id]).sum().item()) + 1

        expected_score = {
            "tokens": exp_tokens,
            "per_tok_logp": per_tok_logp,
            "total_logp": sum(per_tok_logp),
            "first_content_rank": rank,
            "first_content_token": tokenizer.convert_ids_to_tokens(
                [first_content_id])[0],
        }

    return {
        "raw_top": raw_top,
        "post_w_top": post_w_top,
        "extension": extension,
        "expected_score": expected_score,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to model checkpoint directory")
    ap.add_argument("--tokenizer", default="tokenizer_morpheme",
                    help="Path to tokenizer (default: tokenizer_morpheme)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--extend", type=int, default=4,
                    help="Greedy-extend this many tokens after the prompt")
    ap.add_argument("--prompts-file", type=Path,
                    help="JSONL of {prompt, expected?} per line; overrides defaults")
    ap.add_argument("--prompt", action="append",
                    help="Add a single prompt; repeatable; overrides defaults")
    args = ap.parse_args()

    if args.prompts_file:
        import json
        prompts = []
        with open(args.prompts_file) as f:
            for line in f:
                d = json.loads(line)
                prompts.append((d["prompt"], d.get("expected", "")))
    elif args.prompt:
        prompts = [(p, "") for p in args.prompt]
    else:
        prompts = DEFAULT_PROMPTS

    tokenizer = load_tokenizer(Path(args.tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.checkpoint} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)
    model.eval()

    summary = []  # (prompt, expected, total_logp, rank, hit)

    for prompt, expected in prompts:
        r = probe(model, tokenizer, device, prompt, expected,
                  args.top_k, args.extend)
        header = f"PROMPT: {prompt!r}"
        if expected:
            header += f"   (expected: {expected!r})"
        print()
        print(header)
        print(f"  greedy +{args.extend}: {r['extension']!r}")

        # Skip raw top-k (always <w>-dominated). Show post-<w> content dist.
        print(f"  top-{args.top_k} content morphemes (after <w>):")
        for p, i in r["post_w_top"]:
            tok = tokenizer.convert_ids_to_tokens([i])[0]
            bar = "█" * min(40, int(40 * p))
            print(f"    {p*100:5.2f}%  {tok!r:<20}  {bar}")

        if r["expected_score"]:
            es = r["expected_score"]
            print(f"  expected score:")
            print(f"    tokens:        {es['tokens']}")
            print(f"    per-tok log P: {[f'{x:.2f}' for x in es['per_tok_logp']]}")
            print(f"    total log P:   {es['total_logp']:.2f}  "
                  f"(P = {torch.exp(torch.tensor(es['total_logp'])).item():.2e})")
            print(f"    first-content morpheme {es['first_content_token']!r} "
                  f"ranked #{es['first_content_rank']} in post-<w> dist")
            hit = es["first_content_rank"] == 1
            summary.append((prompt, expected, es["total_logp"],
                            es["first_content_rank"], hit))

    if summary:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'expected':<12} {'rank':>5} {'total log P':>13}  prompt")
        for prompt, expected, lp, rank, hit in summary:
            mark = "✓" if hit else " "
            print(f"{mark} {expected!r:<10} {rank:>5} {lp:>13.2f}  {prompt!r}")
        n_hit = sum(1 for *_, h in summary if h)
        print(f"\nTop-1 hits: {n_hit}/{len(summary)}")

if __name__ == "__main__":
    main()
