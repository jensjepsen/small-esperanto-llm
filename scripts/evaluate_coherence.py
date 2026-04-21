"""Coherence evaluation for a trained Esperanto LM.

Samples continuations from a checkpoint on a standard prompt set, then
scores each output with:
  - Verifier diagnostics (grammar / morphology / agreement)
  - LexiconCheck (confabulated-word penalty, counted separately)
  - LaBSE sentence-embedding signals (validated against qwen-2b vs qwen-4b
    to give positive discrimination on real model outputs, d ≈ +0.6):
      * prompt_sent_min   — worst sentence-to-prompt cosine
      * inter_sent_min    — worst pairwise cosine across output sentences
  - Repetition rate (sanity guard; inter_sent_min could be reward-hacked
    by degenerate loops, so we track it and optionally penalise)

Writes per-sample JSONL and an aggregate summary table. With
``--compare-checkpoint``, also runs a paired A/B against a second model
with Cohen's d per metric.

Example:
  python scripts/evaluate_coherence.py \
      --checkpoint models/grpo_step_5000 \
      --num-samples 4 \
      --output runs/eval_step5000.jsonl

  # A/B vs baseline:
  python scripts/evaluate_coherence.py \
      --checkpoint models/grpo_step_5000 \
      --compare-checkpoint models/pretrained_baseline \
      --num-samples 4
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM

from esperanto_lm.data import load_tokenizer, _morpheme_preprocess
from esperanto_lm.verify import (
    DEFAULT_CHECKS,
    LexiconCheck,
    Verifier,
    unknown_word_rate,
)

console = Console()

DEFAULT_PROMPTS = [
    "Hieraŭ mi iris al la urbo kaj",
    "La malnova biblioteko estis",
    "Mia amiko rakontis al mi, ke",
    "Dum la vintro, la infanoj",
    "La kuracisto en la hospitalo",
    "En la ĝardeno de mia avino",
    "La studentoj aŭskultis atente dum",
    "Kiam mi vekiĝis matene,",
    "La trajno alvenis al la stacidomo",
    "Antaŭ multaj jaroj, en malgranda vilaĝo,",
    "La muziko plenigis la ĉambron kaj",
    "Post longa tago de laboro, mi",
    "La vojaĝanto eniris la gastejon kaj",
    "Sur la tablo kuŝis",
    "La birdoj kantis en la arboj dum",
    "Mia patrino ĉiam diris, ke",
]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SKIP_TOKENS = {"<s>", "</s>", "<pad>", "<unk>", "<|user|>", "<|assistant|>", "<|end|>"}


# ---------- sampling -------------------------------------------------------

def decode_generated(tokenizer, token_ids: list[int]) -> str:
    """Decode new tokens to text. Handles <w> word boundary tokens."""
    gen = tokenizer.convert_ids_to_tokens(token_ids)
    gen = [t for t in gen if t not in SKIP_TOKENS]
    has_w = "<w>" in tokenizer.get_vocab()
    if has_w:
        return "".join(t if t != "<w>" else " " for t in gen).strip()
    parts: list[str] = []
    for t in gen:
        if t in (".", ",", ";", ":", "!", "?", ")", "]"):
            parts.append(t)
        elif parts and t not in ("(", "["):
            parts.append(" " + t)
        else:
            parts.append(t)
    return "".join(parts).strip()


@torch.no_grad()
def sample_continuations(model, tokenizer, device, prompt: str, *,
                         n: int, max_new_tokens: int, temperature: float,
                         top_p: float, top_k: int,
                         repetition_penalty: float) -> list[str]:
    """Sample n independent continuations for a prompt."""
    prompt_pp = _morpheme_preprocess(prompt)
    inputs = tokenizer(prompt_pp, return_tensors="pt").to(device)
    n_prompt = inputs["input_ids"].shape[1]
    outputs: list[str] = []
    for _ in range(n):
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        new_ids = out[0][n_prompt:].tolist()
        text = decode_generated(tokenizer, new_ids)
        outputs.append(text)
    return outputs


# ---------- scoring --------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]


def repetition_rate(text: str, n: int = 4) -> float:
    words = re.findall(r"[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+", text.lower())
    if len(words) < n + 1:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def labse_scores(labse, prompt: str, continuation: str) -> dict:
    """LaBSE-based coherence scores.

    Returns prompt_sent_min (worst drift) and inter_sent_min (worst
    pairwise across sentences). NaN when fewer than 2 sentences.
    """
    sents = split_sentences(continuation)
    if not sents:
        return {"prompt_sent_min": float("nan"), "inter_sent_min": float("nan"),
                "n_sentences": 0}
    p_emb = labse.encode([prompt], normalize_embeddings=True, show_progress_bar=False)[0]
    s_embs = labse.encode(sents, normalize_embeddings=True, show_progress_bar=False)
    prompt_sims = [float(np.dot(p_emb, s)) for s in s_embs]
    inter_min = float("nan")
    if len(s_embs) >= 2:
        inter_sims = [float(np.dot(s_embs[i], s_embs[j]))
                      for i in range(len(s_embs))
                      for j in range(i + 1, len(s_embs))]
        inter_min = float(min(inter_sims))
    return {
        "prompt_sent_min": float(min(prompt_sims)),
        "inter_sent_min":  inter_min,
        "n_sentences":     len(sents),
    }


@dataclass
class SampleScore:
    prompt: str
    continuation: str
    # syntactic / lexical
    n_diagnostics: int
    n_grammar: int
    n_lexicon: int
    unk_rate: float
    # semantic (LaBSE)
    prompt_sent_min: float
    inter_sent_min: float
    n_sentences: int
    # surface
    n_tokens: int
    repetition_rate: float


def score_sample(verifier: Verifier, labse, prompt: str, continuation: str) -> SampleScore:
    diags = verifier.verify(continuation)
    n_lexicon = sum(1 for d in diags if d.check == "lexicon")
    labse_s = labse_scores(labse, prompt, continuation)
    return SampleScore(
        prompt=prompt,
        continuation=continuation,
        n_diagnostics=len(diags),
        n_grammar=len(diags) - n_lexicon,
        n_lexicon=n_lexicon,
        unk_rate=unknown_word_rate(continuation),
        prompt_sent_min=labse_s["prompt_sent_min"],
        inter_sent_min=labse_s["inter_sent_min"],
        n_sentences=labse_s["n_sentences"],
        n_tokens=len(continuation.split()),
        repetition_rate=repetition_rate(continuation),
    )


# ---------- aggregation + reporting ----------------------------------------

METRIC_DIRECTION = {
    # higher = better
    "prompt_sent_min": +1,
    "inter_sent_min":  +1,
    # lower = better
    "n_grammar":       -1,
    "n_lexicon":       -1,
    "unk_rate":        -1,
    "repetition_rate": -1,
    # purely descriptive
    "n_sentences":      0,
    "n_tokens":         0,
    "n_diagnostics":    0,
}


def summarize(scores: list[SampleScore]) -> dict[str, tuple[float, float, int]]:
    """Return mean, std, n for each metric."""
    out = {}
    for field_name in METRIC_DIRECTION:
        vals = [getattr(s, field_name) for s in scores]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if vals:
            out[field_name] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        else:
            out[field_name] = (float("nan"), float("nan"), 0)
    return out


def render_summary(title: str, summary: dict) -> Table:
    t = Table(title=title)
    t.add_column("metric"); t.add_column("mean", justify="right")
    t.add_column("std", justify="right"); t.add_column("n", justify="right")
    t.add_column("dir", justify="center")
    for metric, (mean, std, n) in summary.items():
        direction = {1: "↑", -1: "↓", 0: "·"}[METRIC_DIRECTION[metric]]
        t.add_row(metric, f"{mean:+.3f}", f"{std:.3f}", str(n), direction)
    return t


def cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d for 'a is better than b' on same-direction metric."""
    a = [x for x in a if isinstance(x, (int, float)) and not math.isnan(x)]
    b = [x for x in b if isinstance(x, (int, float)) and not math.isnan(x)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def render_paired(a_name: str, b_name: str,
                  a_scores: list[SampleScore], b_scores: list[SampleScore]) -> Table:
    t = Table(title=f"Paired comparison: {a_name} vs {b_name}")
    t.add_column("metric"); t.add_column(a_name, justify="right")
    t.add_column(b_name, justify="right"); t.add_column("d (A−B)*dir", justify="right")
    t.add_column("A better", justify="right")
    for metric, direction in METRIC_DIRECTION.items():
        if direction == 0:
            continue
        va = [getattr(s, metric) for s in a_scores]
        vb = [getattr(s, metric) for s in b_scores]
        # Sign d so positive always means "A is better"
        d = cohen_d(va, vb) * direction
        ma = np.nanmean([v for v in va if not (isinstance(v, float) and math.isnan(v))])
        mb = np.nanmean([v for v in vb if not (isinstance(v, float) and math.isnan(v))])
        a_wins = sum(
            1 for x, y in zip(va, vb)
            if isinstance(x, (int, float)) and isinstance(y, (int, float))
            and not math.isnan(x) and not math.isnan(y)
            and (x - y) * direction > 0
        )
        paired = sum(
            1 for x, y in zip(va, vb)
            if isinstance(x, (int, float)) and isinstance(y, (int, float))
            and not math.isnan(x) and not math.isnan(y)
        )
        t.add_row(metric, f"{ma:+.3f}", f"{mb:+.3f}", f"{d:+.2f}",
                  f"{a_wins}/{paired}" if paired else "—")
    return t


# ---------- main flow ------------------------------------------------------

def load_model(checkpoint: Path, tokenizer_path: Path | None, device: str):
    tokenizer = load_tokenizer(tokenizer_path or checkpoint)
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint))
    model.to(device)
    model.eval()
    return model, tokenizer


def run_checkpoint(args, ck: Path, labse, verifier, prompts: list[str],
                   device: str) -> list[SampleScore]:
    console.print(f"[bold green]Loading {ck}[/]")
    model, tokenizer = load_model(ck, Path(args.tokenizer) if args.tokenizer else None, device)

    all_scores: list[SampleScore] = []
    for i, prompt in enumerate(prompts):
        conts = sample_continuations(
            model, tokenizer, device, prompt,
            n=args.num_samples, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        for cont in conts:
            all_scores.append(score_sample(verifier, labse, prompt, cont))
        console.print(f"  [{i+1}/{len(prompts)}] {prompt[:48]:<50s}  "
                      f"({len(conts)} samples)")

    # release model before next checkpoint
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return all_scores


def write_jsonl(scores: list[SampleScore], out_path: Path, label: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in scores:
            record = {"checkpoint": label, **s.__dict__}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(f"  wrote {len(scores)} samples → {out_path}")


def main():
    p = argparse.ArgumentParser(description="Coherence evaluation for Esperanto LM")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to the model checkpoint to evaluate")
    p.add_argument("--compare-checkpoint", type=Path, default=None,
                   help="Optional second checkpoint for paired A/B comparison")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Tokenizer path (default: loaded from checkpoint)")
    p.add_argument("--prompts-file", type=Path, default=None,
                   help="Optional path to a file of prompts, one per line. "
                        "If omitted, uses the bundled 16-prompt default set.")
    p.add_argument("--num-samples", type=int, default=4,
                   help="Samples per prompt")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--labse-model", type=str, default="sentence-transformers/LaBSE",
                   help="Sentence transformer for coherence scoring")
    p.add_argument("--output", type=Path, default=None,
                   help="JSONL output path for per-sample scores "
                        "(default: no file written, only stdout)")
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu (default: auto)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for sampling")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Device:[/] {device}")
    torch.manual_seed(args.seed)

    if args.prompts_file:
        prompts = [line.strip() for line in args.prompts_file.read_text().splitlines()
                   if line.strip()]
        console.print(f"[bold]Prompts:[/] {len(prompts)} from {args.prompts_file}")
    else:
        prompts = DEFAULT_PROMPTS
        console.print(f"[bold]Prompts:[/] {len(prompts)} (bundled default)")

    console.print(f"[bold green]Loading {args.labse_model}[/]")
    labse = SentenceTransformer(args.labse_model, device=device)
    labse.eval()

    verifier = Verifier(list(DEFAULT_CHECKS) + [LexiconCheck()])

    a_scores = run_checkpoint(args, args.checkpoint, labse, verifier, prompts, device)

    summary_a = summarize(a_scores)
    console.print()
    console.print(render_summary(f"Checkpoint A: {args.checkpoint.name}", summary_a))

    if args.output:
        write_jsonl(a_scores, args.output, args.checkpoint.name)

    if args.compare_checkpoint:
        b_scores = run_checkpoint(args, args.compare_checkpoint, labse, verifier,
                                  prompts, device)
        summary_b = summarize(b_scores)
        console.print()
        console.print(render_summary(f"Checkpoint B: {args.compare_checkpoint.name}",
                                     summary_b))

        # Paired comparison: align by (prompt_idx, sample_idx)
        console.print()
        console.print(render_paired(args.checkpoint.name,
                                    args.compare_checkpoint.name,
                                    a_scores, b_scores))
        if args.output:
            b_out = args.output.with_name(args.output.stem + "_B" + args.output.suffix)
            write_jsonl(b_scores, b_out, args.compare_checkpoint.name)


if __name__ == "__main__":
    main()
