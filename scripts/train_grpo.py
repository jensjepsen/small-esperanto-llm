"""GRPO training for Esperanto LM with verifier-based reward.

Two-tier reward (matches `scripts/grpo_baseline.py`): structural claim
overlap + entity overlap + answer-word presence + regularizers (unk word
rate, grammar flags, short-completion penalty).

Reads SFT-format JSONL: each line `{"messages": [{role, content}, ...]}`.
The first user turn is the prompt; the first assistant turn is BOTH the
structural gold (`claim_overlap` matches against it directly) and the
source for `gold_answers` (presence picks up content-word overlap).

Usage:
    uv run python scripts/train_grpo.py \\
        --checkpoint runs/large/checkpoint-44000 \\
        --output-dir runs/large/checkpoint-44000-grpo \\
        --epochs 1 --num-generations 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from datasets import Dataset

from esperanto_lm.data import load_tokenizer
from esperanto_lm.morphology import decompose
from esperanto_lm.verify import (
    Verifier, LexiconCheck, DEFAULT_CHECKS,
    unknown_word_rate, claim_overlap, claim_entity_overlap,
    tokenize, extract_claims,
)


USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"


# overlap ⇒ entity ⇒ presence is a quality hierarchy: presence means the
# answer word appears; entity means the answer-entity pair appears in a
# claim; overlap means the full (subj, rel, obj) claim matches. Weight
# them accordingly so the structural signal drives late learning.
# Fluency: parseable rewards well-formed sentence structure (positive),
# repetition penalises n-gram loops (the "X estas X estas X" failure mode
# we saw in late-collapse runs).
W = {"overlap": 0.5, "entity": 0.3, "presence": 0.2,
     "parseable": 0.2, "math": 1.0,
     "unk": 0.20, "grammar": 0.10, "repetition": 0.30, "foreign": 0.5,
     "length": 0.30}
LENGTH_GRACE = 50         # words before length penalty kicks in
LENGTH_FULL_PENALTY = 150  # words at which penalty saturates to 1.0
MAX_GRAM = 8
REPETITION_NGRAM = 3

# Esperanto alphabet (case-insensitive). Anything outside this set in an
# alphabetic position is a foreign script: Cyrillic, Arabic, Greek, CJK,
# IPA, math symbols. Catches the multi-script Unicode salad mode-collapse
# we saw at step 4600 (e.g. "tekstἦ lampirἕ pivotĉarpio联 idiotismoὼ").
_EO_LETTERS = set("abcdefghijklmnopqrstuvwxyzĉĝĥĵŝŭABCDEFGHIJKLMNOPQRSTUVWXYZĈĜĤĴŜŬ")

# `est` (copula) appears in nearly every Esperanto sentence so it's
# always filtered. Other question-vocab echoes are handled per-prompt.
_PRESENCE_SKIP_ROOTS = {"est"}
_CONTENT_POS = {"N", "A", "Adv", "V", "INF"}


def _content_stems(text: str) -> set[str]:
    """Lemmas-as-stems: prefix+root joined, restricted to content POS.
    `Parizo`/`Parizaj`/`Parizanoj` all collapse to `pariz`; `malĝoja` to
    `malĝoj`. Closed-class words (la, de, kaj, kun, etc.) are filtered by
    POS, so no explicit stopword list is needed.
    """
    out = set()
    for tok in tokenize(text):
        if tok.pos not in _CONTENT_POS or not tok.root:
            continue
        if tok.root in _PRESENCE_SKIP_ROOTS:
            continue
        stem = "".join(tok.prefixes) + tok.root
        if len(stem) >= 2:
            out.add(stem)
    return out


def gold_word_presence(generation: str, gold_answers: list[str],
                       prompt: str = "") -> float:
    """Stem-level presence. Reward only the stems the gold ADDS beyond
    the question — that's the actual answer content. If the gold shares
    all its stems with the question (rare; gold is pure echo), fall back
    to the full gold set.
    """
    if not gold_answers:
        return 0.0
    gen_stems = _content_stems(generation)
    q_stems = _content_stems(prompt)
    hits, scoreable = 0, 0
    for ans in gold_answers:
        a_stems = _content_stems(ans)
        target = a_stems - q_stems  # answer-specific stems
        if not target:
            target = a_stems  # fallback: gold is entirely question vocab
        if not target:
            continue
        scoreable += 1
        if target & gen_stems:
            hits += 1
    return hits / scoreable if scoreable else 0.0


# ─── Reward function ────────────────────────────────────────────────────────

def demorph(text: str) -> str:
    """Undo morpheme tokenization: `<w>` separates words, morphemes within a
    word are concatenated. Tokenizer.decode produces e.g.
    'la<w>ĉefurbo<w>de<w>francio' which becomes 'la ĉefurbo de francio'.
    Also drops chat/control tokens that may appear in completions.
    """
    for tok in (USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN, "<s>", "</s>", "<pad>", "<unk>"):
        text = text.replace(tok, " ")
    text = text.replace("<w>", " ")
    return re.sub(r"\s+", " ", text).strip()


def repetition_rate(text: str, n: int = REPETITION_NGRAM) -> float:
    """Fraction of n-grams that repeat. 0.0 = no repetition, 1.0 = pure loop.
    Catches degenerate "X estas X estas X" outputs that GRPO can fall into."""
    toks = text.lower().split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def parseable_rate(text: str, verifier=None) -> float:
    """Fraction of period-delimited sentences that are well-formed: extract
    at least one claim AND have zero verifier diagnostics (NPAgreement,
    PredicateAdjAgreement, MissingAccusative, WrongEndings, lexicon, etc).
    A pure structure check would let "Hundoj manĝas pomo." (wrong case)
    pass; coupling with the verifier makes "well-formed" actually mean
    well-formed. If verifier is None, falls back to claim-only check."""
    sents = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    if not sents:
        return 0.0
    if verifier is None:
        clean = sum(1 for s in sents if extract_claims(s))
    else:
        clean = sum(1 for s in sents
                    if extract_claims(s) and not verifier.verify(s))
    return clean / len(sents)


_MATH_HASH_PATTERN = re.compile(r"####\s*(-?\d+(?:[.,]\d+)?)")
_MATH_RESPONDO_PATTERN = re.compile(
    r"(?:respondo|rezulto|sumo|valoro|estas?|=)\s*[:=]?\s*(-?\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_MATH_ANY_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def _parse_num(s: str) -> float | None:
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def extract_math_answer(text: str) -> float | None:
    """Lenient final-answer extraction for GSM8K-style chain-of-thought.
    Tries (in order): #### N (canonical), 'respondo|estas|= N' near the end,
    last number anywhere. Returns None if no number found."""
    m = _MATH_HASH_PATTERN.search(text)
    if m:
        return _parse_num(m.group(1))
    matches = _MATH_RESPONDO_PATTERN.findall(text)
    if matches:
        return _parse_num(matches[-1])
    nums = _MATH_ANY_NUMBER.findall(text)
    if nums:
        return _parse_num(nums[-1])
    return None


def math_close(generation: str, gold: str | None) -> float:
    """Smoothly-decaying correctness for GSM8K-style numeric answers.
    Exact match → 1.0; partial credit via exp(-relative_error). Gives the
    policy gradient signal even on near-misses (e.g. 17 when gold is 18
    scores ~0.95, while 1000 scores ~0.0). Returns 0.0 if either side is
    non-numeric so it stays dormant on factoid/atomic data."""
    if not gold:
        return 0.0
    ga = extract_math_answer(gold)
    ca = extract_math_answer(generation)
    if ga is None or ca is None:
        return 0.0
    if abs(ga - ca) < 1e-6:
        return 1.0
    rel_err = abs(ca - ga) / max(abs(ga), 1.0)
    import math as _math
    return _math.exp(-min(rel_err, 10.0))


def length_excess(text: str) -> float:
    """0..1 length penalty. Free below LENGTH_GRACE words; linear ramp to 1.0
    at LENGTH_FULL_PENALTY words. Discourages unbounded chain-of-thought
    rambling and the parseable-reward farming pattern (each extra sentence
    adds parseable bonus regardless of math correctness)."""
    n = len(text.split())
    if n <= LENGTH_GRACE:
        return 0.0
    return min(1.0, (n - LENGTH_GRACE) / (LENGTH_FULL_PENALTY - LENGTH_GRACE))


def foreign_char_rate(text: str) -> float:
    """Fraction of words containing any non-Esperanto alphabetic char.
    Word-level (not char-level) so a single foreign rune in a token
    poisons the whole word, catching multi-script salads like
    'tekstἦ lampirἕ pivotĉarpio联 idiotismoὼ'. Cyrillic, Arabic,
    Greek, CJK, IPA, math symbols all count as foreign."""
    words = re.findall(r"\S+", text)
    if not words:
        return 0.0
    bad = sum(1 for w in words
              if any(c.isalpha() and c not in _EO_LETTERS for c in w))
    return bad / len(words)


def make_reward_components(max_gram=MAX_GRAM):
    """Return `(reward_funcs, reward_weights)` for GRPOTrainer.

    Each reward function returns a raw 0..1 score per completion. Weights
    carry the sign (penalties are negative). TRL logs each function under
    `rewards/<fn_name>` so the breakdown shows up in the step logs.
    """
    verifier = Verifier(DEFAULT_CHECKS + [LexiconCheck(freq_threshold=3)])

    def _texts(completions):
        return [demorph(c) for c in completions]

    def reward_overlap(prompts, completions, gold=None, **_):
        n = len(completions)
        refs = gold if gold is not None else [None] * n
        return [claim_overlap(t, r) if r else 0.0
                for t, r in zip(_texts(completions), refs)]

    def reward_entity(prompts, completions, gold=None, **_):
        n = len(completions)
        refs = gold if gold is not None else [None] * n
        return [claim_entity_overlap(t, r) if r else 0.0
                for t, r in zip(_texts(completions), refs)]

    def reward_presence(prompts, completions, gold_answers=None, **_):
        n = len(completions)
        gas = gold_answers if gold_answers is not None else [[]] * n
        q_texts = [demorph(p) for p in prompts]
        return [gold_word_presence(t, a, q)
                for t, a, q in zip(_texts(completions), gas, q_texts)]

    def reward_parseable(prompts, completions, **_):
        return [parseable_rate(t, verifier) for t in _texts(completions)]

    def reward_unk(prompts, completions, **_):
        return [unknown_word_rate(t, freq_threshold=3) for t in _texts(completions)]

    def reward_grammar(prompts, completions, **_):
        return [min(1.0, len(verifier.verify(t)) / max_gram)
                for t in _texts(completions)]

    def reward_repetition(prompts, completions, **_):
        return [repetition_rate(t) for t in _texts(completions)]

    def reward_foreign(prompts, completions, **_):
        return [foreign_char_rate(t) for t in _texts(completions)]

    def reward_length(prompts, completions, **_):
        return [length_excess(t) for t in _texts(completions)]

    def reward_math(prompts, completions, gold=None, **_):
        n = len(completions)
        refs = gold if gold is not None else [None] * n
        return [math_close(t, r) for t, r in zip(_texts(completions), refs)]

    funcs = [reward_overlap, reward_entity, reward_presence, reward_parseable,
             reward_math,
             reward_unk, reward_grammar, reward_repetition, reward_foreign,
             reward_length]
    weights = [W["overlap"], W["entity"], W["presence"], W["parseable"],
               W["math"],
               -W["unk"], -W["grammar"], -W["repetition"], -W["foreign"],
               -W["length"]]
    return funcs, weights


# ─── Prompt formatting ──────────────────────────────────────────────────────

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


def format_prompt(question: str, has_w: bool, style: str) -> str:
    if style == "chat":
        # Chat tokens are untrained on the base model — only use this for
        # SFT checkpoints that have actually seen these tokens.
        text = f"{USER_TOKEN} {question} {ASSISTANT_TOKEN}"
    else:
        # Bare question: the base model has seen "Q? A" style continuations
        # in Wikipedia pretraining; let the pretrain prior pick answer shape.
        # A "La respondo estas" suffix grammatically fits only single-noun
        # gold (factual_qa's capitals) — it breaks for sentence-style gold
        # ("Jericho situas en Germanio.") and ellipsis gold ("Trinki ĝin.").
        text = f"{question}"
    return _morph(text, has_w)


# ─── Data loading ───────────────────────────────────────────────────────────

def _iter_messages(source: str, max_n: int = 0):
    """Yield message-lists from a local JSONL file or an HF Hub dataset id.
    Same dispatch as scripts/train_sft.py."""
    src_path = Path(source)
    if src_path.exists():
        with open(src_path) as f:
            for i, line in enumerate(f):
                if max_n and i >= max_n:
                    break
                yield json.loads(line)["messages"]
    else:
        from datasets import load_dataset as hf_load
        ds = hf_load(source, split="train")
        for i, row in enumerate(ds):
            if max_n and i >= max_n:
                break
            yield row["messages"]


def load_chat_dataset(sources: list[str], has_w: bool, prompt_style: str,
                      max_examples: int = 0, presence_target: str = "assistant"):
    """SFT-format conversations: each row has `messages: [{role, content}, ...]`.

    Each `source` is either a local JSONL path or an HF Hub dataset id (same
    convention as scripts/train_sft.py). First user/assistant turn becomes
    (prompt, gold). Assistant text is the structural gold (claim_overlap
    matches it directly).

    `presence_target` controls what `gold_answers` contains:
      "assistant"    — full assistant text (default; presence rewards any
                       gold content stem in the generation).
      "math-answer"  — just the `#### N` numeric value (use for GSM8K-style
                       math gold; prevents the model from gaming presence
                       by sprinkling gold-vocab words around).
    """
    rows = []
    per_source_cap = max_examples // len(sources) if max_examples else 0
    for source in sources:
        for msgs in _iter_messages(source, per_source_cap):
            user = next(m["content"] for m in msgs if m["role"] == "user")
            assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
            if presence_target == "math-answer":
                # Restrict presence to just the `#### N` value so the model
                # can't game it by sprinkling gold-vocab words around.
                num = extract_math_answer(assistant)
                gold_answers = [str(int(num) if num == int(num) else num)] if num is not None else []
            else:
                gold_answers = [assistant.strip()]
            rows.append({
                "prompt": format_prompt(user.strip(), has_w, prompt_style),
                "gold": assistant.strip(),
                "gold_answers": gold_answers,
            })
    return Dataset.from_list(rows)


# ─── Training entry ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/large/checkpoint-44000",
                        help="Default: base model. Pass an SFT checkpoint to fine-tune that.")
    parser.add_argument("--dataset", type=str, nargs="+",
                        default=["jensjepsen/esperanto-sft-atomic-qa",
                                 "jensjepsen/esperanto-sft-factoid"],
                        help="One or more SFT-format sources. Each entry can be a local "
                             "JSONL path or an HF Hub dataset id (same dispatch as "
                             "scripts/train_sft.py). Factoids give overlap/entity a "
                             "workout (full-sentence gold); atomic-QA exercises presence "
                             "on short answers.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-style", choices=["chat", "continuation"],
                        default="continuation")
    parser.add_argument("--presence-target", choices=["assistant", "math-answer"],
                        default="assistant",
                        help="What presence reward scores against. 'assistant' "
                             "(default): all gold content stems. 'math-answer': "
                             "just the `#### N` numeric value — use for GSM8K to "
                             "stop the model from gaming presence with gold-vocab.")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-prompt-len", type=int, default=128)
    parser.add_argument("--max-completion-len", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL coefficient against reference policy")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Higher temp → more diversity per group → bigger "
                             "reward_std → cleaner GRPO advantage signal.")
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3,
                        help="Keep only the N most recent checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in --output-dir")
    parser.add_argument("--wandb-project", default="jepsen/espllm",
                        help="`entity/project` for Weights & Biases. "
                             "Pass empty string to disable wandb logging.")
    parser.add_argument("--wandb-run-name", default=None,
                        help="Optional run name (default: auto from output-dir).")
    parser.add_argument("--wandb-tags", nargs="*", default=None,
                        help="Optional tags for the wandb run.")
    args = parser.parse_args()

    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoModelForCausalLM

    if args.wandb_project:
        import os
        import wandb
        if "/" in args.wandb_project:
            entity, project = args.wandb_project.split("/", 1)
        else:
            entity, project = None, args.wandb_project
        os.environ.setdefault("WANDB_PROJECT", project)
        if entity:
            os.environ.setdefault("WANDB_ENTITY", entity)
        wandb.init(
            entity=entity,
            project=project,
            name=args.wandb_run_name or Path(args.output_dir).name,
            tags=args.wandb_tags,
            config={
                "checkpoint": args.checkpoint,
                "dataset": [str(p) for p in args.dataset],
                "reward_weights": W,
                "num_generations": args.num_generations,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "learning_rate": args.learning_rate,
                "beta": args.beta,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "prompt_style": args.prompt_style,
                "max_prompt_len": args.max_prompt_len,
                "max_completion_len": args.max_completion_len,
            },
        )

    print(f"Loading {args.checkpoint}...", flush=True)
    tokenizer = load_tokenizer(Path(args.checkpoint))
    has_w = "<w>" in tokenizer.get_vocab()

    # The project tokenizer marks every morpheme as a special token, so
    # TRL's `batch_decode(..., skip_special_tokens=True)` would erase the
    # whole completion. Force False just for the trainer's decode path.
    _orig_batch_decode = tokenizer.batch_decode
    def _keep_specials(ids, **kw):
        kw["skip_special_tokens"] = False
        return _orig_batch_decode(ids, **kw)
    tokenizer.batch_decode = _keep_specials

    # If the tokenizer has chat tokens (chat-format SFT models), use them
    # as eos. TRL builds GenerationConfig from `processing_class.eos_token_id`,
    # which is a single int derived from `tokenizer.eos_token`. We set the
    # primary eos to `<|end|>` (the SFT termination token); `<|user|>` is
    # added as an extra stop to catch the model spawning a fake follow-up
    # turn for parseable-reward farming. `<|assistant|>` is intentionally
    # NOT added — it sits at the answer boundary in SFT data, and adding
    # it can cause length-1 completions and unstable gradients.
    chat_stops = []
    end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
    if end_id is not None and end_id != tokenizer.unk_token_id:
        original_eos = tokenizer.eos_token
        tokenizer.eos_token = END_TOKEN
        chat_stops.append(end_id)
        print(f"tokenizer.eos_token: {original_eos!r} → {END_TOKEN!r} "
              f"(id={end_id})", flush=True)
    user_id = tokenizer.convert_tokens_to_ids(USER_TOKEN)
    if user_id is not None and user_id != tokenizer.unk_token_id:
        chat_stops.append(user_id)

    # Load fp32 master weights — AMP (fp16=True below) handles cast.
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)

    dataset = load_chat_dataset(args.dataset, has_w, args.prompt_style,
                                args.max_examples,
                                presence_target=args.presence_target)
    srcs = ", ".join(args.dataset)
    print(f"Dataset: {len(dataset)} prompts from [{srcs}] "
          f"(style={args.prompt_style}, presence={args.presence_target})", flush=True)

    reward_funcs, reward_weights = make_reward_components()

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        bf16=False, fp16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        reward_weights=reward_weights,
        report_to="wandb" if args.wandb_project else "none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    # Replace the trainer's GenerationConfig with one that stops on any of
    # the chat tokens — primary `<|end|>` plus `<|user|>`/`<|assistant|>`.
    # The latter two catch the model trying to spawn a fake multi-turn
    # chain mid-completion (a parseable-reward exploit we saw at step 60).
    if len(chat_stops) > 1:
        from transformers import GenerationConfig
        gc = trainer.generation_config
        cfg = gc.to_dict()
        cfg["eos_token_id"] = chat_stops
        trainer.generation_config = GenerationConfig(**cfg)
    print(f"trainer.generation_config.eos_token_id = "
          f"{trainer.generation_config.eos_token_id}", flush=True)
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
