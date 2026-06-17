"""Evaluate an SFT'd model on a hand-crafted ICL Q/A eval set.

For each eval example: build the chat-template prompt the trainer
used, generate up to a short cap, strip the <|end|> token, compare
to the gold answer.

Reports:
  - overall accuracy (exact-match, case-insensitive, whitespace-
    normalized)
  - accuracy by question-template heuristic (color / state /
    counting / etc.) — keyed by simple substrings in the question.

Usage:
    python scripts/eval_icl.py \\
        --checkpoint runs/large/checkpoint-44000-causal-icl-sft/final \\
        --eval data/causal_corpus/eval_handcrafted_v27.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
)

from esperanto_lm.data import _morpheme_preprocess


USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"
SPECIAL_TOKENS = (USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN)


def preprocess_chat(text: str) -> str:
    """Mirror `train_sft.py:preprocess_and_tokenize`: split on chat
    special tokens, morpheme-preprocess the content parts, rejoin
    with spaces. Keeps the special tokens atomic in the tokenizer's
    output."""
    pat = "(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")"
    parts = re.split(pat, text)
    out = []
    for p in parts:
        if p in SPECIAL_TOKENS:
            out.append(p)
        elif p.strip():
            out.append(_morpheme_preprocess(p.strip()))
        else:
            out.append(p)
    return " ".join(out)


def detokenize_morphemes(tokens: list[str]) -> str:
    """Reverse the morpheme tokenizer: `<w>` → space, concatenate
    the rest. Mirrors the postprocess in `scripts/generate.py`."""
    return "".join(" " if t == "<w>" else t for t in tokens)


def normalize(s: str) -> str:
    """Lenient match: case-fold, drop punctuation/spaces around words,
    strip leading articles, prepositions, and accusative endings.
    Causal-connector unification: "pro tio, ke X" and "pro tio ke X"
    rewrite to "ĉar X" — both are standard Esperanto for "because X"."""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    s = s.rstrip(".,;:!?")
    s = re.sub(r"^pro tio\s*,?\s*ke\s+", "ĉar ", s)
    for prefix in ("en la ", "en ", "sur la ", "sur ", "el la ", "el ",
                    "tra la ", "tra ", "apud la ", "apud ",
                    "ĉe la ", "ĉe ", "al la ", "al ",
                    "per la ", "per ", "de la ", "de ",
                    "la "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = re.sub(r'\bla ', '', s).strip()
    s = re.sub(r'\b(\w+)ojn\b', r'\1oj', s)
    s = re.sub(r'\b(\w+)on\b', r'\1o', s)
    return s


# Wiki-fact substrings: the eval's last block (idx 212-231) tests
# pretraining knowledge — capitals, scientific facts, historical
# dates. Detected by anchor terms in the question rather than index
# range so the tagger stays robust to eval-file edits.
_WIKI_ANCHORS = (
    "ĉefurbo", "profesio de zamenhof", "naskiĝis ŝekspiro",
    "naskiĝis einsteino", "konsistas akvo", "orbitas ĉirkaŭ la suno",
    "plej granda surtera besto", "unua mondmilito",
    "mortis mozarto", "amazona pluvarbo", "fluas la nilo",
    "malkovris marie curie", "natura satelito de la tero",
    "kreis linukson", "granda muro", "unue trinkis kafon",
    "strukturon de dna", "plej granda planedo",
)

# Heuristic question-type tagger from the question text. Used to
# break down accuracy by template.
QUESTION_TAGS = [
    ("wiki",       lambda q: any(a in q.lower() for a in _WIKI_ANCHORS)),
    ("color",      lambda q: "koloro" in q),
    ("posture",    lambda q: "pozici" in q),
    ("openness",   lambda q: "malfermita aŭ fermita" in q),
    ("fullness",   lambda q: "plena aŭ malplena" in q),
    ("lock_state", lambda q: "ŝlosita aŭ malŝlosita" in q),
    ("power_state", lambda q: "aktiva aŭ neaktiva" in q),
    ("cleanliness", lambda q: "pura aŭ malpura" in q),
    ("first",      lambda q: "okazis unue" in q),
    ("last",       lambda q: "okazis laste" in q or "laste en la rakonto" in q),
    ("state_after", lambda q: "post kiam" in q.lower() or "post la " in q.lower()),
    ("location_start", lambda q: "komence" in q),
    ("count",      lambda q: "kiom" in q.lower()),
    ("ordering",   lambda q: ("antaŭ" in q or "post la"
                              in q or "post kio" in q.lower())),
    ("instrument", lambda q: "per kio" in q.lower()),
    ("who",        lambda q: q.lower().startswith("kiu")),
    ("what",       lambda q: q.lower().startswith("kion")),
    ("where",      lambda q: q.lower().startswith("kie")
                              or q.lower().startswith("kien")),
    ("why",        lambda q: q.lower().startswith("kial")),
]


def tag_question(q: str) -> str:
    for name, pred in QUESTION_TAGS:
        if pred(q):
            return name
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--tokenizer", type=str, default="tokenizer_morpheme")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = all examples")
    p.add_argument("--pass-k", type=int, default=1,
                   help="Generate K samples per question, count correct "
                        "if any match (pass@K). K=1 is greedy.")
    p.add_argument("--temperature", type=float, default=None,
                   help="If set, override the greedy pass@1 with sampling "
                        "at this temperature. pass@k retries still use 0.7.")
    p.add_argument("--top-p", type=float, default=None,
                   help="Nucleus sampling cutoff (only used when sampling).")
    p.add_argument("--top-k", type=int, default=None,
                   help="Top-k truncation (only used when sampling).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading tokenizer + model from {args.checkpoint}…", flush=True)
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    # Add chat special tokens if not present (base tokenizer
    # doesn't have them; SFT adds them at training time).
    tok.add_special_tokens(
        {"additional_special_tokens":
         [USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN]})
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.float16
    ).to(device).eval()
    model.resize_token_embeddings(len(tok))

    end_id = tok.convert_tokens_to_ids(END_TOKEN)
    assert end_id is not None and end_id != tok.unk_token_id, \
        f"<|end|> token not in tokenizer"

    results: list[dict] = []
    with open(args.eval) as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            rec = json.loads(line)
            user = rec["messages"][0]["content"]
            gold = rec["messages"][1]["content"]
            accepted = rec.get("accepted_answers") or [gold]
            if gold not in accepted:
                accepted = [gold] + list(accepted)
            q = user.split("Demando:", 1)[-1].strip()

            # Build the chat-template prompt the trainer used, then
            # split-and-morpheme-preprocess: special tokens stay
            # atomic, content gets `<w>`-boundary preprocessing.
            prompt_text = preprocess_chat(
                f"{USER_TOKEN} {user} {ASSISTANT_TOKEN}")
            inputs = tok(
                prompt_text, return_tensors="pt",
                return_token_type_ids=False).to(device)
            def _generate_one(do_sample=False, temperature=1.0):
                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    eos_token_id=end_id,
                    pad_token_id=tok.pad_token_id or end_id,
                )
                if do_sample:
                    if args.top_p is not None:
                        gen_kwargs["top_p"] = args.top_p
                    if args.top_k is not None:
                        gen_kwargs["top_k"] = args.top_k
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                prompt_len = inputs["input_ids"].shape[-1]
                gen_ids = out[0][prompt_len:].tolist()
                gen_toks = tok.convert_ids_to_tokens(gen_ids)
                cleaned: list[str] = []
                for t in gen_toks:
                    if t == END_TOKEN:
                        break
                    if t in ("<s>", "</s>", "<pad>", "<unk>",
                             USER_TOKEN, ASSISTANT_TOKEN):
                        continue
                    cleaned.append(t)
                return detokenize_morphemes(cleaned).strip()

            def _matches(answer, gold):
                na, ng = normalize(answer), normalize(gold)
                if na == ng:
                    return True
                if (len(ng.split()) > 1
                        and re.search(rf"\b{re.escape(ng)}\b", na)):
                    return True
                # Unit-less acceptance: gold "kvin kilogramoj" (number
                # + unit), pred "kvin" — accept. The model gave the
                # right count even without the unit. Same when gold
                # has trailing acc-marker: "du horojn" vs pred "du".
                if len(ng.split()) == 2 and " " not in na:
                    gold_first = ng.split()[0]
                    if gold_first == na:
                        return True
                # Single-word gold: accept if it appears in pred at a
                # word boundary within the FIRST sentence, AND pred
                # doesn't carry an inversion prefix. First-sentence
                # restriction prevents matching the gold word inside
                # an echo loop (e.g. model says "kvar fruktoj ." then
                # regurgitates "se oni forprenas la tri fruktojn" from
                # the prompt — the matcher would see "tri" and call
                # it ✓).
                na_first = na.split(".", 1)[0].strip()
                # Arithmetic CoT shape ("A + B = C" or "A plus B egalas C"):
                # the gold must appear in the RESULT (after the last
                # "=" or "egalas"), not as an addend earlier in the
                # equation. Without this, "tri pomoj + du oranĝoj =
                # ses pomoj" gets ✓ for gold="tri" via the leading
                # addend, even though the model's stated answer is
                # "ses".
                #
                # For multi-clause chains ("8-3=5. 5-2=3. maria havas
                # tri krajonojn."), check (a) the FINAL "= ..." in the
                # whole prediction and (b) the FINAL sentence — the
                # answer is usually in the last sentence as
                # "X havas Y" or in the trailing equation.
                arith_result = None
                if " = " in na_first:
                    arith_result = na_first.rsplit(" = ", 1)[1].strip()
                elif " egalas " in na_first:
                    arith_result = na_first.rsplit(" egalas ", 1)[1].strip()
                if arith_result is not None:
                    if (ng and " " not in ng
                            and re.search(rf"\b{re.escape(ng)}\b",
                                          arith_result)):
                        return True
                    # Multi-clause chain: scan the FINAL equation result.
                    last_eq = max(na.rfind(" = "), na.rfind(" egalas "))
                    if last_eq >= 0:
                        sep_len = 3 if na.rfind(" = ") > na.rfind(" egalas ") else 8
                        final_tail = na[last_eq + sep_len:].split(".", 1)[0].strip()
                        if (ng and " " not in ng
                                and re.search(rf"\b{re.escape(ng)}\b", final_tail)):
                            return True
                    # Final-sentence check: "maria havas tri krajonojn"
                    # at the end of the chain, with gold "tri". Only the
                    # tail AFTER a "= ..." / "havas ..." / "restas ..."
                    # counts as the answer — front addends ("du etaĝoj +
                    # tri etaĝoj = kvin etaĝoj") don't.
                    sentences = [s.strip() for s in na.split(".") if s.strip()]
                    if sentences and ng and " " not in ng:
                        last = sentences[-1]
                        # Strip prefix up to last answer-bearing keyword.
                        tail = last
                        for kw in (" = ", " egalas ", " havas ", " restas "):
                            idx = tail.rfind(kw)
                            if idx >= 0:
                                tail = tail[idx + len(kw):]
                        if (tail != last  # found at least one keyword
                                and re.search(rf"\b{re.escape(ng)}\b", tail)):
                            return True
                    # Explicit reject for arith-shape answers — don't
                    # fall through to lenient rules below.
                    return False
                if (ng and " " not in ng
                        and not na.startswith("ne ,")
                        and not na.startswith("ne ")
                        and re.search(rf"\b{re.escape(ng)}\b", na_first)):
                    return True
                if " estas " in na and na.split(" estas ", 1)[1] == ng:
                    return True
                if na.startswith(ng + " "):
                    return True
                for vp in ("estis ", "restas ", "havas "):
                    if na.startswith(vp):
                        if na[len(vp):] == ng or na[len(vp):].startswith(ng + " "):
                            return True
                if "ĝin" in ng:
                    ng_no_pro = re.sub(r'\bĝin\b', '', ng).strip()
                    na_no_obj = re.sub(r'\bla \w+o\b', '', na).strip()
                    if ng_no_pro and ng_no_pro == na_no_obj:
                        return True
                    ng_verb = ng.split("ĝin")[0].strip()
                    if ng_verb and na.startswith(ng_verb):
                        return True
                # Symmetric pronoun-resolution: pred uses "ĝin" where
                # gold spells out the noun (e.g. pred "por malŝlosi ĝin"
                # vs gold "por malŝlosi la kofron" — same fact).
                if "ĝin" in na:
                    na_verb = na.split("ĝin")[0].strip()
                    if na_verb and ng.startswith(na_verb):
                        return True
                return False

            def _matches_any(pred):
                return any(_matches(pred, g) for g in accepted)

            if args.temperature is not None:
                answer = _generate_one(
                    do_sample=True, temperature=args.temperature)
            else:
                answer = _generate_one(do_sample=False)
            ok = _matches_any(answer)
            if not ok and args.pass_k > 1:
                for _ in range(args.pass_k - 1):
                    alt = _generate_one(do_sample=True, temperature=0.7)
                    if _matches_any(alt):
                        answer = alt
                        ok = True
                        break
            results.append({
                "tag": tag_question(q),
                "q": q,
                "gold": gold,
                "pred": answer,
                "ok": ok,
            })
            mark = "✓" if ok else "✗"
            print(f"  {mark} [{results[-1]['tag']}] gold={gold!r} "
                  f"pred={answer!r}", flush=True)

    n = len(results)
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n=== Overall: {n_ok}/{n} = {n_ok/n:.1%} ===")

    by_tag = defaultdict(lambda: [0, 0])
    for r in results:
        by_tag[r["tag"]][0] += 1
        by_tag[r["tag"]][1] += int(r["ok"])
    print("\nBy template:")
    for tag, (total, ok) in sorted(by_tag.items(), key=lambda x: -x[1][0]):
        print(f"  {tag:>16}: {ok}/{total} = {ok/total:.0%}")


if __name__ == "__main__":
    main()
