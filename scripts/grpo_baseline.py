"""Score the current policy's completions on the GRPO reward function,
to verify the signal has useful spread before kicking off training.

What you want to see:
  - Mean reward roughly mid-range (e.g. 0.2-0.6), not saturated.
  - Per-prompt spread (best - worst > 0.1) for at least most prompts —
    GRPO advantage is group-relative; if all G generations score the
    same, the gradient is zero.
  - Component breakdown shows each reward term contributes — if one
    dominates (e.g. unknown rate is always 0), retune weights.

Usage:
    uv run python scripts/grpo_baseline.py \\
        --checkpoint runs/large/checkpoint-44000-sft/checkpoint-26000 \\
        --benchmark benchmarks/factual_qa.json \\
        --num-prompts 30 --num-generations 4
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM

from esperanto_lm.data import load_tokenizer
from esperanto_lm.morphology import decompose
from esperanto_lm.verify import (
    Verifier, LexiconCheck, DEFAULT_CHECKS,
    unknown_word_rate, claim_overlap, claim_entity_overlap, extract_claims,
)


USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"


def _morph(text, has_w):
    words = re.findall(r"[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+|[^\s]", text)
    parts = []
    for word in words:
        if parts and has_w:
            parts.append("<w>")
        if word[0].isalpha():
            parts.extend(decompose(word))
        else:
            parts.append(word)
    return " ".join(parts)


def _decode(tokens, has_w):
    out = [t for t in tokens if t not in ("<s>", "</s>", "<pad>", "<unk>",
                                            USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN)]
    if has_w:
        return "".join(t if t != "<w>" else " " for t in out)
    parts = []
    for t in out:
        if t in (".", ",", ";", ":", "!", "?", ")", "]"):
            parts.append(t)
        elif parts and t not in ("(", "["):
            parts.append(" " + t)
        else:
            parts.append(t)
    return "".join(parts)


# Two-tier reward:
#  - structural: full claim_overlap match against a structured gold sentence
#                (e.g. "La ĉefurbo de Francio estas Parizo.")
#  - presence:   the gold answer word(s) appear anywhere in the generation
#                — picks up partial credit when the model says the right thing
#                but in different syntactic form
W = {"overlap": 0.4, "entity": 0.2, "presence": 0.3,
     "unk": 0.20, "grammar": 0.10, "short": 0.5}
MAX_GRAM = 8

_STOPWORDS = {"la", "de", "kaj", "estas", "respondo", "en", "al", "el",
              "kun", "por", "pri", "pro", "sur", "sub", "kio", "kiu",
              "kien", "kiam", "kie", "ĉu"}


def _content_words(text: str) -> set[str]:
    """Lowercase content words, minus stopwords."""
    words = re.findall(r"[a-zĉĝĥĵŝŭ]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def gold_word_presence(generation: str, gold_answers: list[str]) -> float:
    """Fraction of gold answers whose content words all appear in the
    generation. For multi-word gold like 'Nov-Delio', either piece counts.
    """
    if not gold_answers:
        return 0.0
    gen_words = _content_words(generation)
    hits = 0
    for ans in gold_answers:
        ans_words = _content_words(ans)
        if not ans_words:
            continue
        if ans_words & gen_words:
            hits += 1
    return hits / len(gold_answers)


def score_components(text: str, gold: str | None,
                      gold_answers: list[str], verifier):
    """Return (total_reward, breakdown_dict)."""
    parts = {"overlap": 0.0, "entity": 0.0, "presence": 0.0,
             "unk": 0.0, "grammar": 0.0, "short": 0.0}
    if gold:
        parts["overlap"] = W["overlap"] * claim_overlap(text, gold)
        parts["entity"]  = W["entity"]  * claim_entity_overlap(text, gold)
    parts["presence"] = W["presence"] * gold_word_presence(text, gold_answers)
    parts["unk"]      = -W["unk"] * unknown_word_rate(text, freq_threshold=3)
    n_gram = len(verifier.verify(text))
    parts["grammar"]  = -W["grammar"] * min(1.0, n_gram / MAX_GRAM)
    if len(text.split()) < 5:
        parts["short"] = -W["short"]
    total = sum(parts.values())
    return total, parts


# Per-category gold templates: build a structured gold sentence from the
# question + answer so claim_overlap has something to match against.
def build_structured_gold(item: dict) -> tuple[str | None, list[str]]:
    cat = item["category"]
    q = item["question"]
    ans = item["answer"]
    answers = ans if isinstance(ans, list) else [ans]
    primary = answers[0]

    # Extract the question's entity for templating. Fall back to a generic
    # "La respondo estas X." if we can't.
    if cat == "ĉefurbo":
        # "Kio estas la ĉefurbo de X?"
        m = re.search(r"ĉefurbo de (\w[\w\-ĉĝĥĵŝŭĈĜĤĴŜŬ]*)", q)
        ent = m.group(1) if m else None
        gold = f"La ĉefurbo de {ent} estas {primary}." if ent else f"La ĉefurbo estas {primary}."
    elif cat == "lando":
        # "En kiu lando troviĝas X?"
        m = re.search(r"troviĝas (\w[\w\-ĉĝĥĵŝŭĈĜĤĴŜŬ]*)", q)
        ent = m.group(1) if m else None
        gold = f"{ent} troviĝas en {primary}." if ent else f"Ĝi troviĝas en {primary}."
    elif cat == "profesio":
        # "Kiu estas X?"
        m = re.search(r"Kiu estas (.+?)\?", q)
        ent = m.group(1) if m else None
        gold = f"{ent} estas {primary}." if ent else f"Estas {primary}."
    elif cat == "aritmetiko":
        # "Kio estas A + B?" or "A * B?"
        m = re.search(r"(?:Kio estas|Kalkulu:)\s*(.+?)\?", q)
        expr = m.group(1).strip() if m else q
        gold = f"{expr} = {primary}."
    elif cat == "estas":
        m = re.search(r"Kio estas (.+?)\?", q)
        ent = m.group(1) if m else None
        gold = f"{ent} estas {primary}." if ent else f"Ĝi estas {primary}."
    elif cat == "devenlando":
        m = re.search(r"devenas (.+?)\?", q)
        ent = m.group(1) if m else None
        gold = f"{ent} devenas el {primary}." if ent else f"Ĝi devenas el {primary}."
    elif cat == "ĝenro":
        m = re.search(r"ĝenro de (.+?)\?", q)
        ent = m.group(1) if m else None
        gold = f"La ĝenro de {ent} estas {primary}." if ent else f"La ĝenro estas {primary}."
    elif cat == "laborkampo":
        m = re.search(r"laboris (.+?)\?", q)
        ent = m.group(1) if m else None
        gold = f"{ent} laboris en {primary}." if ent else f"Laboris en {primary}."
    else:
        gold = f"La respondo estas {primary}."
    return gold, answers


def _hist(values, n_bins=20, lo=None, hi=None):
    if not values:
        return ""
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    if hi == lo:
        return f"all values = {lo:.3f}"
    bins = [0] * n_bins
    for v in values:
        i = min(n_bins - 1, max(0, int((v - lo) / (hi - lo) * n_bins)))
        bins[i] += 1
    peak = max(bins) or 1
    out = []
    for i, c in enumerate(bins):
        edge = lo + (hi - lo) * i / n_bins
        bar = "█" * int(c * 32 / peak)
        out.append(f"  {edge:+6.2f}  {bar} {c}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/factual_qa.json"))
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--show-examples", type=int, default=3,
                        help="Show top-N best and worst per prompt")
    parser.add_argument("--prompt-style", choices=["chat", "continuation"],
                        default="chat",
                        help="`chat` wraps with <|user|>/<|assistant|> for SFT models; "
                             "`continuation` just feeds the question + answer-stem "
                             "for base models")
    args = parser.parse_args()

    print(f"Loading {args.checkpoint}...", flush=True)
    tokenizer = load_tokenizer(Path(args.checkpoint))
    has_w = "<w>" in tokenizer.get_vocab()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.float16).to(device)
    model.eval()
    verifier = Verifier(DEFAULT_CHECKS + [LexiconCheck(freq_threshold=3)])

    with open(args.benchmark) as f:
        items = json.load(f)
    items = items[:args.num_prompts]
    end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)

    all_rewards = []
    all_breakdowns = []
    spreads = []
    examples = []  # (reward, prompt, gen, gold)

    for pi, item in enumerate(items, 1):
        question = item["question"]
        gold, gold_answers = build_structured_gold(item)

        if args.prompt_style == "chat":
            chat = f"{USER_TOKEN} {question} {ASSISTANT_TOKEN}"
        else:
            chat = f"{question} La respondo estas"
        inp = tokenizer(_morph(chat, has_w), return_tensors="pt").to(device)
        prompt_len = inp["input_ids"].shape[1]

        rewards = []
        per_gen = []
        for _ in range(args.num_generations):
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=50,
                    repetition_penalty=1.15,
                    do_sample=True,
                    eos_token_id=end_id,
                )
            gen_ids = out[0, prompt_len:].tolist()
            gen_text = _decode(tokenizer.convert_ids_to_tokens(gen_ids), has_w).strip()
            r, parts = score_components(gen_text, gold, gold_answers, verifier)
            rewards.append(r)
            per_gen.append((r, gen_text, parts))
            all_rewards.append(r)
            all_breakdowns.append(parts)

        spread = max(rewards) - min(rewards)
        spreads.append(spread)
        per_gen.sort(key=lambda x: -x[0])

        print(f"\n━━ {pi}/{len(items)}  Q: {question[:70]}", flush=True)
        print(f"   gold: {gold}", flush=True)
        print(f"   r: best={max(rewards):+.3f}  mean={statistics.mean(rewards):+.3f}  "
              f"worst={min(rewards):+.3f}  spread={spread:.3f}", flush=True)
        if args.show_examples:
            for r, txt, _ in per_gen[: args.show_examples]:
                print(f"   [{r:+.3f}] {txt[:140]}", flush=True)

    # Aggregate report
    print("\n\n========================================================", flush=True)
    print(" AGGREGATE REWARD DISTRIBUTION", flush=True)
    print("========================================================\n", flush=True)
    print(f"  total samples: {len(all_rewards)}", flush=True)
    print(f"  mean   = {statistics.mean(all_rewards):+.3f}", flush=True)
    print(f"  stdev  = {statistics.stdev(all_rewards):+.3f}", flush=True)
    qs = statistics.quantiles(all_rewards, n=4)
    print(f"  Q25    = {qs[0]:+.3f}", flush=True)
    print(f"  median = {qs[1]:+.3f}", flush=True)
    print(f"  Q75    = {qs[2]:+.3f}", flush=True)
    print(f"  min    = {min(all_rewards):+.3f}", flush=True)
    print(f"  max    = {max(all_rewards):+.3f}", flush=True)
    print(f"\n  histogram:")
    print(_hist(all_rewards))

    # Spread per prompt — GRPO needs nonzero spread to learn anything
    print(f"\n  per-prompt spread (max - min within group of {args.num_generations}):")
    print(f"    mean spread = {statistics.mean(spreads):.3f}")
    print(f"    fraction of prompts with spread < 0.05: "
          f"{sum(1 for s in spreads if s < 0.05) / len(spreads) * 100:.0f}%")

    # Component contributions
    print(f"\n  reward component means (positive = bonus, negative = penalty):")
    keys = ["overlap", "entity", "presence", "unk", "grammar", "short"]
    for k in keys:
        vals = [b[k] for b in all_breakdowns]
        nonzero = sum(1 for v in vals if abs(v) > 1e-9)
        print(f"    {k:8s}  mean={statistics.mean(vals):+.4f}   "
              f"nonzero={nonzero}/{len(vals)} ({nonzero/len(vals)*100:.0f}%)")


if __name__ == "__main__":
    main()
