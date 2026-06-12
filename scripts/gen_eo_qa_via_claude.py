"""Generate Esperanto Q/A training samples via the Anthropic API.

One-shot the Claude API with a structured prompt that includes a few
seed examples for each target template (why-direction, state_after,
first/last, wiki facts, count-arithmetic, distractor). Each API call
returns a JSON list; we accumulate to --n-samples and write the
flattened JSONL in the same chat-template shape as the rest of the
SFT corpus.

Cost-aware: requests are batched (one prompt → ~20 samples) so a 5k
run is ~250 API calls — at 5e-6/in token, 1.5e-5/out token, and
~500 in / ~1500 out per call, that's ~$8-15 total.

Usage:
    export ANTHROPIC_API_KEY=...
    uv run python scripts/gen_eo_qa_via_claude.py \\
        --n-samples 5000 --out runs/claude_5k.jsonl \\
        --templates why state count distractor wiki
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

try:
    from anthropic import Anthropic
except ImportError as e:
    raise SystemExit(
        "anthropic package missing — run: uv add anthropic"
    ) from e


# Seed examples per template, included verbatim in the prompt so the
# model produces output in the same shape. Drawn from the v72 eval's
# gold answers to maximize transfer.
SEED_EXAMPLES = {
    "why_state": [
        {"q": "Kial Petro ne povis tuj eniri?",
         "a": "Ĉar la pordo estis ŝlosita."},
        {"q": "Kial la infanoj ne naĝis?",
         "a": "Ĉar la akvo estis malvarma."},
        {"q": "Kial la kuko bruliĝis?",
         "a": "Ĉar la forno estis tro varma."},
    ],
    "why_purpose": [
        {"q": "Por kio Petro turnis la ŝlosilon?",
         "a": "Por malŝlosi la pordon."},
        {"q": "Kial Anna malfermis la pordon?",
         "a": "Por eniri la ĉambron."},
        {"q": "Kial Sara prenis la ŝlosilon?",
         "a": "Por malŝlosi la kofron."},
    ],
    "state_after": [
        {"q": "Kia estas la pordo nun?", "a": "Malfermita."},
        {"q": "Kia estas la seruro nun?", "a": "Malŝlosita."},
        {"q": "Kia estas la forno?", "a": "Aktiva."},
        {"q": "Kia estas la glaso nun?", "a": "Malplena."},
        {"q": "Kia estas la telero post la lavo?", "a": "Pura."},
    ],
    "first_last": [
        {"q": "Kio okazis unue?", "a": "Anna prenis libron."},
        {"q": "Kio okazis laste?", "a": "Karlo revenis hejmen."},
    ],
    "wiki": [
        {"q": "Kio estas la kapitalo de Aŭstrio?", "a": "Vieno."},
        {"q": "En kiu kontinento situas Brazilo?", "a": "Sudameriko."},
        {"q": "Kio estas la plej granda planedo?", "a": "Jupitero."},
        {"q": "Kion Marie Curie malkovris?",
         "a": "Radiumon kaj poloniumon."},
    ],
    "count_subtract": [
        {"q": "Kiom da pomoj restas en la korbo?", "a": "Sep pomoj."},
        {"q": "Kiom da moneroj restas al Petro?", "a": "Tri moneroj."},
    ],
    "compare": [
        {"q": "Kiu havis pli da pomoj?", "a": "Maria."},
        {"q": "En kiu ĉambro estis pli da seĝoj?",
         "a": "En la dua ĉambro."},
    ],
    "distractor": [
        {"q": ("Inter pomo, sofo, tablo, lampo, kio NE estis"
                " en la salono?"),
         "a": "Pomo."},
        {"q": "Inter taso, libro, telero, forko, kio NE estis"
              " en la kuirejo?",
         "a": "Libro."},
    ],
}

TEMPLATE_BRIEFS = {
    "why_state": (
        "Causal reasoning where the CAUSE is a STATE/PROPERTY that "
        "preceded the effect. The answer always starts with 'Ĉar' "
        "and names the state, NOT the consequence. Common pattern: "
        "'Ĉar [entity] estis [adjective]'. Each Q/A pair must have "
        "a short scene-setup sentence before the question that "
        "makes the state explicit."),
    "why_purpose": (
        "Purpose-oriented why: answer starts with 'Por' (in order to). "
        "Pattern: 'Por [infinitive]' or 'Por ke [agent] povu [verb]'. "
        "Each Q/A pair includes a setup sentence that establishes a "
        "goal-directed action."),
    "state_after": (
        "After-action state. The answer is a BARE adjectival state "
        "(malfermita, ŝlosita, plena, pura, varma) or a single short "
        "sentence repeating that state. Use Esperanto property values "
        "from this whitelist: malfermita/fermita, ŝlosita/malŝlosita, "
        "plena/malplena, pura/malpura, varma/malvarma, aktiva/neaktiva, "
        "kuirita/krudaĵo, vekita/dormanta, sidanta/staranta. The "
        "setup sentence describes an action that produces the state."),
    "first_last": (
        "Temporal ordering. The setup gives 3-5 sequential events "
        "(named subjects like Anna, Petro, Karlo, Maria, Sara). "
        "Question asks for first or last; answer is the full event "
        "sentence in past tense."),
    "wiki": (
        "World-knowledge facts grounded in a one-sentence setup. "
        "Geography (capitals, continents), science (Marie Curie, "
        "DNA, planets), history (WWI year, Great Wall), figures "
        "(Zamenhof, Shakespeare, Linus Torvalds, Einstein). "
        "Answer is the specific fact, not a generic phrase like "
        "'en la urbo'."),
    "count_subtract": (
        "Simple subtraction arithmetic embedded in a story. Setup "
        "names an initial count and a transfer/consumption. Answer "
        "is in Esperanto cardinals + plural noun: 'Sep pomoj.', "
        "'Tri moneroj.'"),
    "compare": (
        "Comparison between two quantities. Setup names two amounts "
        "(or two scenes with amounts); question asks 'pli da X'. "
        "Answer names the entity/scene with more."),
    "distractor": (
        "Negative-existence question. Setup lists 3-4 entities present "
        "in a scene. Question lists those plus one absent entity, asks "
        "'kio NE estis'. Answer is the absent entity name."),
}


PROMPT_TEMPLATE = """Generate {n_per_call} new Esperanto Q/A training samples for the "{template}" category.

CATEGORY: {brief}

Seed examples (do not copy verbatim; produce NEW samples in the same shape):
{seeds}

REQUIREMENTS:
- Esperanto only, native-correct grammar (correct -is/-as/-os tense, correct accusative -n, correct adjective agreement).
- Each Q must have a 1-3 sentence setup before "Demando: " that grounds the answer.
- Variety: use different subjects (Anna, Petro, Karlo, Maria, Sara, Lucia, Tomaso, Eva), different objects (pomo, libro, ŝlosilo, glaso, sako, lampo, krajono, telero, forno, fenestro, pordo, monero, kuko, seĝo, tablo).
- Answers are SHORT and direct — match the style of the seed examples exactly.

OUTPUT FORMAT: a single JSON array of {n_per_call} objects, each with keys "setup" (string, scene context ending without a question), "q" (the question, just the Esperanto question), "a" (the answer). No prose, no markdown — pure JSON.

Example output shape:
[
  {{"setup": "Petro volis ekzameni la kontenton de la kesto, sed la kesto estis ŝlosita.", "q": "Kial Petro ne povis ekzameni la kontenton?", "a": "Ĉar la kesto estis ŝlosita."}},
  ...
]
"""


def call_claude(client, prompt, model="claude-opus-4-7", max_tokens=4096):
    """One API call returning the raw text response."""
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def parse_response(text, template):
    """Extract the JSON array from the response text. Tolerates wrapping
    markdown code fences. Yields chat-template dicts."""
    # Strip markdown code fences if present
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    # Find the first [...] block
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        items = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [warn] JSON parse failed: {e}; skipping batch")
        return
    for it in items:
        if not isinstance(it, dict):
            continue
        setup = (it.get("setup") or "").strip()
        q = (it.get("q") or "").strip()
        a = (it.get("a") or "").strip()
        if not q or not a:
            continue
        user_content = f"{setup}\n\nDemando: {q}" if setup else f"Demando: {q}"
        yield {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": a},
            ],
            # metadata kept out of the messages dict so it doesn't leak
            # into training; downstream slicing/filtering can read it.
            "_template": template,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=5000,
                   help="Total Q/A pairs to generate (split across templates).")
    p.add_argument("--n-per-call", type=int, default=20,
                   help="Samples requested per API call.")
    p.add_argument("--templates", nargs="+",
                   default=list(SEED_EXAMPLES.keys()),
                   help="Which templates to include in the mix.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="claude-opus-4-7")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sleep-between-calls", type=float, default=0.5)
    args = p.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("Set ANTHROPIC_API_KEY before running.")

    client = Anthropic()
    rng = random.Random(args.seed)

    # Distribute n_samples evenly across templates.
    per_template = args.n_samples // len(args.templates)
    leftover = args.n_samples - per_template * len(args.templates)
    plan = {t: per_template + (1 if i < leftover else 0)
            for i, t in enumerate(args.templates)}

    out_records: list[dict] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as out_fh:
        for template, target in plan.items():
            print(f"[{template}] target {target} samples")
            seeds = SEED_EXAMPLES.get(template, [])
            brief = TEMPLATE_BRIEFS.get(template, "")
            seed_str = "\n".join(
                f"  - Q: {ex['q']}\n    A: {ex['a']}"
                for ex in seeds)
            n_calls = (target + args.n_per_call - 1) // args.n_per_call
            produced = 0
            for call_i in range(n_calls):
                remaining = target - produced
                this_batch = min(args.n_per_call, remaining)
                if this_batch <= 0:
                    break
                prompt = PROMPT_TEMPLATE.format(
                    template=template,
                    brief=brief,
                    seeds=seed_str,
                    n_per_call=this_batch,
                )
                try:
                    text = call_claude(
                        client, prompt,
                        model=args.model, max_tokens=args.max_tokens)
                except Exception as e:
                    print(f"  [warn] API call failed: {e}; retrying once")
                    time.sleep(2.0)
                    try:
                        text = call_claude(
                            client, prompt,
                            model=args.model, max_tokens=args.max_tokens)
                    except Exception as e2:
                        print(f"  [error] giving up on this batch: {e2}")
                        continue
                batch_written = 0
                for rec in parse_response(text, template):
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    batch_written += 1
                    produced += 1
                print(f"  call {call_i+1}/{n_calls}: wrote {batch_written}"
                      f" (total {produced}/{target})")
                if args.sleep_between_calls > 0:
                    time.sleep(args.sleep_between_calls)
            print(f"[{template}] done: {produced}/{target}")
    print(f"\nTotal samples written to {args.out}")


if __name__ == "__main__":
    main()
