"""Build a GRPO IF training set by REWRITING v4 rows with a new mix of rules.

For each source row from `danish-instruction-following-v4`:
  1. Extract the ORIGINAL prompt (task + old rules bundled)
  2. Sample 1-5 NEW rules from a unified pool: our 46 (Danish) + ~15 Google
     IFEval types (English descriptions — Gemma translates on the fly)
  3. Sample params for each new rule
  4. Ask gemma-3-12b via OpenRouter structured output to (a) extract the
     RAW base task from the source prompt, and (b) produce a NEW prompt
     that weaves the NEW rules into that task using varied surface form
     (inline / before / after / list / comma-run / etc.)
  5. Emit an HF-friendly row:
        {task, prompt, constraints, params, source, old_constraints}

Output = HF dataset via save_to_disk. Constraint names use `google:...`
prefix for Google-schema rules so the GRPO reward can dispatch by prefix.

Usage:
  uv run python scripts/build_grpo_if_rewrite.py \\
    --out data/grpo_if_rewrite_v1 \\
    --n 10000 --concurrency 12 --seed 42

Cost estimate: 10k * ~800in + 400out tokens on gemma-3-12b ≈ $0.80.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset, Dataset

from if_constraints import ALL as OUR_ALL, sample_combo as sample_ours
from ifeval_google import instructions_registry as _reg
from constraint_compat import is_valid as _combo_is_valid, merge_combo_params as _merge_combo_params

MODEL = "google/gemini-2.5-flash-lite"


# ── keys / http ──────────────────────────────────────────────────────────────

def _read_key(names):
    for name in names:
        for p in [Path.home() / name, Path.home() / f".{name}"]:
            if p.exists():
                return p.read_text().strip()
    return None


_SESSION = None


async def _get_session():
    global _SESSION
    if _SESSION is None:
        import aiohttp
        key = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY")
               or _read_key(["or", "openrouter"]))
        if not key:
            raise SystemExit("No OPENROUTER_API_KEY set and no ~/or key file.")
        _SESSION = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://claude-code-if",
                     "X-Title": "grpo-if-rewrite"},
            timeout=aiohttp.ClientTimeout(total=90))
    return _SESSION


_SCHEMA = {
    "type": "object",
    "properties": {
        "base_task": {"type": "string"},
        "new_prompt": {"type": "string"},
    },
    "required": ["base_task", "new_prompt"],
    "additionalProperties": False,
}


async def call_gemma(prompt: str, max_retries: int = 4) -> dict | None:
    session = await _get_session()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 1400,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "rewrite", "strict": True, "schema": _SCHEMA},
        },
    }
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions", json=body
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(backoff); backoff *= 2; continue
                if resp.status != 200:
                    text = await resp.text()
                    print(f"  http {resp.status}: {text[:150]}", file=sys.stderr)
                    return None
                data = await resp.json()
                raw = data["choices"][0]["message"]["content"]
                return json.loads(raw)
        except Exception as e:
            print(f"  err {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            await asyncio.sleep(backoff); backoff *= 2
    return None


# ── Google constraint pool ───────────────────────────────────────────────────
# Each entry: (name_with_prefix, sample_fn(rng)->params, describe_fn(params)->english_desc)
# Uses Google's own Instruction.build_description() where practical.

_KW_POOL = ["Danmark", "København", "sommer", "vinter", "havet", "bog",
            "musik", "kaffe", "familie", "arbejde", "skole", "rejse",
            "kunst", "historie", "videnskab", "natur", "byen", "landet"]
_FORBID_POOL = [
    # Filler adverbs
    "meget", "faktisk", "altså", "sådan", "godt", "typisk",
    "generelt", "grundlæggende", "især", "nemlig",
    # Mid-difficulty connectives — harder to avoid without breaking flow.
    "men", "eller", "hvis", "fordi", "derfor", "også", "når", "mens",
]


def _google_pool():
    reg = _reg.INSTRUCTION_DICT
    out = []

    # keywords:existence
    def _s_kw_exist(rng):
        return {"keywords": rng.sample(_KW_POOL, k=rng.randint(1, 3))}
    def _d_kw_exist(p):
        return f"Include the following keywords in your response: {', '.join(p['keywords'])}."
    out.append(("google:keywords:existence", _s_kw_exist, _d_kw_exist))

    # keywords:frequency
    def _s_kw_freq(rng):
        return {"keyword": rng.choice(_KW_POOL),
                "frequency": rng.randint(2, 5),
                "relation": rng.choice(["at least", "less than"])}
    def _d_kw_freq(p):
        return f"In your response, the word '{p['keyword']}' should appear {p['relation']} {p['frequency']} times."
    out.append(("google:keywords:frequency", _s_kw_freq, _d_kw_freq))

    # keywords:forbidden_words
    def _s_forbid(rng):
        return {"forbidden_words": rng.sample(_FORBID_POOL, k=rng.randint(2, 4))}
    def _d_forbid(p):
        return f"Do not include any of these words in your response: {', '.join(p['forbidden_words'])}."
    out.append(("google:keywords:forbidden_words", _s_forbid, _d_forbid))

    # keywords:letter_frequency
    def _s_letter(rng):
        return {"letter": rng.choice("aehilnorst"),
                "let_frequency": rng.randint(5, 25),
                "let_relation": rng.choice(["at least", "less than"])}
    def _d_letter(p):
        return f"The letter '{p['letter']}' should appear {p['let_relation']} {p['let_frequency']} times in your response."
    out.append(("google:keywords:letter_frequency", _s_letter, _d_letter))

    # length_constraints:number_sentences
    def _s_sents(rng):
        return {"num_sentences": rng.randint(3, 12),
                "relation": rng.choice(["at least", "less than"])}
    def _d_sents(p):
        return f"Your response should contain {p['relation']} {p['num_sentences']} sentences."
    out.append(("google:length_constraints:number_sentences", _s_sents, _d_sents))

    # length_constraints:number_words
    def _s_words(rng):
        return {"num_words": rng.randint(50, 400),
                "relation": rng.choice(["at least", "less than"])}
    def _d_words(p):
        return f"Answer with {p['relation']} {p['num_words']} words."
    out.append(("google:length_constraints:number_words", _s_words, _d_words))

    # length_constraints:number_paragraphs
    def _s_paras(rng):
        return {"num_paragraphs": rng.randint(2, 5)}
    def _d_paras(p):
        return (f"Your response should contain exactly {p['num_paragraphs']} paragraphs, "
                f"separated with a markdown divider: ***")
    out.append(("google:length_constraints:number_paragraphs", _s_paras, _d_paras))

    # length_constraints:nth_paragraph_first_word
    def _s_nth(rng):
        n = rng.randint(2, 4)
        return {"num_paragraphs": rng.randint(n, n + 2),
                "nth_paragraph": n,
                "first_word": rng.choice(["Faktisk", "Desuden", "Endvidere",
                                          "Alligevel", "Derfor"])}
    def _d_nth(p):
        return (f"There should be {p['num_paragraphs']} paragraphs. Paragraphs "
                f"are separated with a double line break. "
                f"Paragraph {p['nth_paragraph']} must start with the word '{p['first_word']}'.")
    out.append(("google:length_constraints:nth_paragraph_first_word", _s_nth, _d_nth))

    # detectable_content:number_placeholders
    def _s_ph(rng):
        return {"num_placeholders": rng.randint(2, 5)}
    def _d_ph(p):
        return (f"Your response must contain at least {p['num_placeholders']} placeholders "
                f"represented by square brackets, such as [address].")
    out.append(("google:detectable_content:number_placeholders", _s_ph, _d_ph))

    # detectable_content:postscript
    def _s_post(rng):
        return {"postscript_marker": rng.choice(["P.S.", "P.P.S", "NB:"])}
    def _d_post(p):
        return (f"At the end of your response, please explicitly add a postscript "
                f"starting with {p['postscript_marker']}.")
    out.append(("google:detectable_content:postscript", _s_post, _d_post))

    # detectable_format:number_bullet_lists
    def _s_bul(rng):
        return {"num_bullets": rng.randint(2, 6)}
    def _d_bul(p):
        return (f"Your answer must contain exactly {p['num_bullets']} bullet points. "
                f"Use markdown bullets like: * this is a bullet.")
    out.append(("google:detectable_format:number_bullet_lists", _s_bul, _d_bul))

    # detectable_format:constrained_response
    def _s_cr(rng):
        return {}
    def _d_cr(p):
        return "Answer with one of the following options: 'My answer is yes.', 'My answer is no.', 'My answer is maybe.'"
    out.append(("google:detectable_format:constrained_response", _s_cr, _d_cr))

    # detectable_format:number_highlighted_sections
    def _s_hi(rng):
        return {"num_highlights": rng.randint(2, 4)}
    def _d_hi(p):
        return f"Highlight at least {p['num_highlights']} sections in your answer with markdown, i.e. *highlighted section*."
    out.append(("google:detectable_format:number_highlighted_sections", _s_hi, _d_hi))

    # detectable_format:multiple_sections
    def _s_ms(rng):
        return {"section_spliter": rng.choice(["Section", "SECTION"]),
                "num_sections": rng.randint(2, 4)}
    def _d_ms(p):
        return (f"Your response must have {p['num_sections']} sections. "
                f"Mark the beginning of each section with {p['section_spliter']} X, "
                f"such as: {p['section_spliter']} 1")
    out.append(("google:detectable_format:multiple_sections", _s_ms, _d_ms))

    # detectable_format:json_format
    out.append(("google:detectable_format:json_format",
                lambda rng: {},
                lambda p: "Entire output should be wrapped in JSON format. You can use markdown ticks such as ```."))

    # detectable_format:title
    out.append(("google:detectable_format:title",
                lambda rng: {},
                lambda p: "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>."))

    # combination:two_responses
    out.append(("google:combination:two_responses",
                lambda rng: {},
                lambda p: "Give two different responses. Responses and only responses should be separated by 6 asterisk symbols: ******."))

    # combination:repeat_prompt
    out.append(("google:combination:repeat_prompt",
                lambda rng: {"prompt_to_repeat": "__ORIGINAL_TASK__"},  # filled at render
                lambda p: "First, repeat the request without change, then give your answer (do not say anything before repeating the request; the request you need to repeat does not include this sentence)."))

    # startend:end_checker
    def _s_end(rng):
        return {"end_phrase": rng.choice(["Det er mit svar.",
                                          "Andre spørgsmål?",
                                          "Jeg håber, du fandt det nyttigt."])}
    def _d_end(p):
        return f"Finish your response with this exact phrase: {p['end_phrase']} No other words should follow this phrase."
    out.append(("google:startend:end_checker", _s_end, _d_end))

    # startend:quotation
    out.append(("google:startend:quotation",
                lambda rng: {},
                lambda p: "Wrap your entire response with double quotation marks."))

    # change_case:capital_word_frequency
    def _s_cap(rng):
        return {"capital_frequency": rng.randint(1, 5),
                "capital_relation": rng.choice(["at least", "less than"])}
    def _d_cap(p):
        return f"In your response, words with all capital letters should appear {p['capital_relation']} {p['capital_frequency']} times."
    out.append(("google:change_case:capital_word_frequency", _s_cap, _d_cap))

    # change_case:english_capital / english_lowercase — verifier is language-
    # agnostic (just `text == text.upper()` / `.lower()`), but Google's
    # original description forces English and thus collides with every
    # Danish task. Swap to Danish descriptions; keep constraint names +
    # verifier untouched so downstream code stays compatible.
    out.append(("google:change_case:english_capital",
                lambda rng: {},
                lambda p: "Skriv hele svaret med STORE BOGSTAVER (alle bogstaver skal være store)."))
    out.append(("google:change_case:english_lowercase",
                lambda rng: {},
                lambda p: "Skriv hele svaret med små bogstaver (ingen store bogstaver må forekomme)."))

    # punctuation:no_comma
    out.append(("google:punctuation:no_comma",
                lambda rng: {},
                lambda p: "In your entire response, refrain from the use of any commas."))

    return out


GOOGLE_POOL = _google_pool()


# ── Sample a combined rule combo ─────────────────────────────────────────────

def _kind_of(name: str) -> str:
    """Coarse kind for conflict dedup. Rules of same kind can contradict each other."""
    n = name.lower()
    if any(k in n for k in ["paragraph", "single_paragraph", "at_most_n_paragraphs"]):
        return "paragraph"
    if any(k in n for k in ["number_sentences", "sentence", "at_least_n_sentences",
                            "at_most_n_sentences", "exact_sentences"]):
        return "sentence"
    if any(k in n for k in ["number_words", "min_words", "max_words", "exact_words",
                            "word_count", "number_of_words"]):
        return "wordcount"
    if any(k in n for k in ["bullet", "bulleted_list"]):
        return "bullet"
    if any(k in n for k in ["json", "constrained_response"]):
        return "format_lock"
    if any(k in n for k in ["capital_word_frequency"]):
        return "caps_freq"
    if any(k in n for k in ["english_capital", "all_uppercase"]):
        return "uppercase"
    if any(k in n for k in ["english_lowercase", "all_lowercase"]):
        return "lowercase"
    if any(k in n for k in ["no_comma"]):
        return "no_comma"
    if any(k in n for k in ["quotation"]):
        return "quotation"
    if any(k in n for k in ["title"]):
        return "title"
    if any(k in n for k in ["end_checker", "ends_with"]):
        return "end_phrase"
    if any(k in n for k in ["postscript"]):
        return "postscript"
    if any(k in n for k in ["placeholder"]):
        return "placeholder"
    if any(k in n for k in ["multiple_sections", "sections"]):
        return "sections"
    if any(k in n for k in ["letter_frequency", "letter_exactly", "letter_freq"]):
        return "letter_freq"
    return name  # unique-per-name for anything else


def sample_combined_combo(rng: random.Random, min_size: int, max_size: int,
                          google_frac: float) -> list[dict]:
    """Return list of {name, params, describe, source_pool} where source_pool ∈ {ours, google}.
    Deduplicated by coarse kind across pools so paragraph/word-count/format rules
    from different pools don't contradict."""
    # Start with ours (has tag-conflict handling built in for our-46 pool)
    ours = sample_ours(rng, min_size=min_size, max_size=max_size)
    n_ours = len(ours)
    n_google = int(round(n_ours * google_frac))
    n_google = min(n_google, n_ours - 1) if n_ours > 1 else 0
    n_google = max(0, n_google)
    keep_ours = ours[:n_ours - n_google]

    used_names: set[str] = {r["name"] for r in keep_ours}
    used_kinds: set[str] = {_kind_of(r["name"]) for r in keep_ours}

    google_picks = []
    for _ in range(n_google * 6):
        if len(google_picks) >= n_google:
            break
        name, sample_fn, desc_fn = rng.choice(GOOGLE_POOL)
        if name in used_names:
            continue
        kind = _kind_of(name)
        if kind in used_kinds:
            continue
        used_names.add(name); used_kinds.add(kind)
        params = sample_fn(rng)
        google_picks.append({
            "name": name, "params": params,
            "describe": desc_fn(params), "source_pool": "google",
        })

    out = [{"name": r["name"], "params": r["params"],
            "describe": r["render"], "source_pool": "ours"} for r in keep_ours]
    out.extend(google_picks)
    rng.shuffle(out)
    return out


# Rejection-sampling wrapper over sample_combined_combo — enforces
# constraint_compat's alias + pair + param checks. The naive sampler's
# _kind_of() dedup catches some conflicts but misses namespace duplicates
# (e.g. entire_in_quotes ≡ google:startend:quotation) and cross-kind
# format conflicts (e.g. json_format × two_responses). See
# scripts/constraint_compat.py for the full list.
_COMBO_GIVEUP_CT = 0

def sample_valid_combo(rng, min_size, max_size, google_frac, max_tries=20):
    global _COMBO_GIVEUP_CT
    for _ in range(max_tries):
        combo = sample_combined_combo(rng, min_size, max_size, google_frac)
        params = _merge_combo_params(combo)
        ok, _reasons = _combo_is_valid([r["name"] for r in combo], params)
        if ok:
            return combo
    _COMBO_GIVEUP_CT += 1
    return combo  # fall back to last attempt after max_tries


# ── Prompt template ──────────────────────────────────────────────────────────

# ── Style axes (cartesian: sample one option per axis per row) ───────────────

_PLACEMENT = [
    "alle regler ØVERST, før grundopgaven",
    "alle regler NEDERST, efter grundopgaven",
    "reglerne er delt op: nogle før grundopgaven, resten efter",
    "reglerne interleavet: første regel før opgaven, midterste regler indlejret i opgaveteksten, sidste regel efter",
    "hver regel som separat linje, indkredset omkring grundopgaven i en rimelig rækkefølge",
    "alle regler indlejret i grundopgavens sætninger — ingen adskilt liste",
    "reglerne som en enkelt afsluttende paragraf efter opgaven",
    "reglerne som en indledende paragraf før opgaven",
    "reglerne som en punktopstilling MIDT i teksten, mellem to sætninger af grundopgaven",
]

_FORM = [
    "nummereret liste (1. 2. 3. ...)",
    "bulleted med *",
    "bulleted med -",
    "bulleted med •",
    "komma-separeret enkeltlinje ('X, Y, Z')",
    "semikolon-separeret enkeltlinje ('X; Y; Z')",
    "naturlige separate sætninger, én regel per sætning",
    "én lang prosa-paragraph med reglerne vævet ind",
    "telegramstil: hver regel som ét kort udbrud ('MAX 50 ORD.'), adskilt af punktum",
    "spørgsmål-stil: hver regel formuleret som et ønske eller høfligt spørgsmål",
    "en blanding af form (én bullet, én komma-liste, én sætning)",
    "romertal (I. II. III.)",
    "bogstaver (a) b) c))",
]

_REPHRASE = [
    "PRÆCIS den givne formulering — kun oversæt til dansk hvis reglen er på engelsk. Ingen omskrivning.",
    "let omskrevet — samme betydning men lidt naturligere dansk. Bevar tal og nøgleord.",
    "kraftig omskrivning — omformuler helt i din egen stil, men bevar ALLE tal, nøgleord, og krav.",
    "let indslag af ord som 'nemlig', 'jo', 'altså' hist og her uden at ændre reglens indhold",
]

_TONE = [
    "neutral", "formelt", "høfligt (kunne du venligst...)", "direkte imperativt",
    "mundret og hverdagsagtigt", "kortfattet og nøgternt",
    "let sarkastisk eller humoristisk", "meget alvorligt",
    "kollegialt og venligt", "bureaukratisk (som et formelt dokument)",
    "poetisk eller pyntet", "kort-og-godt slang",
    "professionelt og teknisk", "afslappet som en ven der skriver",
]

_OPENER = [
    "Krav:", "Betingelser:", "Regler:", "Instruktioner:", "Sørg for at:",
    "Husk:", "OBS:", "OBS!", "NB:", "NB!", "Bemærk:", "Vigtigt:",
    "Vær opmærksom på:", "Overhold følgende:", "Følgende gælder:",
    "Formkrav:", "Restriktioner:", "Rammer:", "Konditioner:",
    "Retningslinjer:", "Følg disse retningslinjer:", "Følg venligst:",
    "Du bedes:", "Man skal:", "PS:", "PS!", "P.S.:", "P.S.",
    "Kort sagt:", "Til orientering:", "For god ordens skyld:",
    "Vær nøje:", "Vær venlig at:", "Vigtige krav:", "Vigtige oplysninger:",
    "Note:", "Til dig:", "Læs først:", "Læg mærke til:",
    "Krav til svaret:", "Format:", "Betingelser til opgaven:",
    "Hør her:", "Small print:", "Advarsel:", "Advarsel!",
    "Det er vigtigt at:", "Vær sikker på at:",
    "Punkter du skal huske:", "Din opgave inkluderer:",
    "Vær opmærksom:", "Der er nogle betingelser:",
    "Betingelserne er:", "Vil du:",
    "Jeg vil bede dig om at:", "Kan du:",
    None, None, None, None,  # sometimes no opener
]

_TERMINATOR = [".", ";", "!", ""]  # sometimes no terminator

_CASE_MODIFIER = [
    "normalt (sætningscase)", "normalt (sætningscase)", "normalt (sætningscase)",
    "normalt (sætningscase)", "normalt (sætningscase)",  # bias toward normal
    "alle regler i små bogstaver (lower)",
    "enkelte nøgleord i CAPS for at fremhæve",
    "hele regelsektionen i CAPS (sjældent)",
    "regel-teksten starter med lille bogstav",
]

_QUIRKS = [
    None, None, None, None, None,  # often none
    "tilføj en emoji eller to (fx 📝 eller ✍️) tæt på reglerne",
    "brug parenteser omkring en eller flere regler ('(max 100 ord)')",
    "sæt en regel i anførselstegn ('regel er: \"start med Faktisk\"')",
    "brug streger (---) til at afgrænse regelsektionen fra opgaven",
    "sæt regelnummeret i firkantede parenteser ([1], [2])",
    "efterlad ét ekstra linjeskift mellem opgave og regler",
    "inkluder en kort forklaring til én af reglerne (fx 'da det er vigtigt for læsbarheden')",
]


def sample_style_hint(rng: random.Random) -> tuple[str, dict]:
    """Sample cartesian combo across style axes; return (hint_text, tags_dict)."""
    tags = {
        "placement": rng.choice(_PLACEMENT),
        "form": rng.choice(_FORM),
        "rephrase": rng.choice(_REPHRASE),
        "tone": rng.choice(_TONE),
        "opener": rng.choice(_OPENER),
        "terminator": rng.choice(_TERMINATOR),
        "case": rng.choice(_CASE_MODIFIER),
        "quirk": rng.choice(_QUIRKS),
    }
    opener_text = f"brug præcis '{tags['opener']}' som indledning til regelsektionen" if tags["opener"] else "ingen introducerende overskrift til reglerne — bare reglerne selv"
    term_text = {".": "afslut hver regel med punktum",
                 ";": "afslut hver regel med semikolon",
                 "!": "afslut hver regel med udråbstegn",
                 "":  "ingen tegnsætning mellem regler — bare linjeskift eller mellemrum"}[tags["terminator"]]
    quirk_text = f"Særtræk: {tags['quirk']}." if tags["quirk"] else ""
    hint = (
        f"Placering: {tags['placement']}. "
        f"Form: {tags['form']}. "
        f"Genfortolkning af reglerne: {tags['rephrase']} "
        f"Tone: {tags['tone']}. "
        f"Opener: {opener_text}. "
        f"Terminator: {term_text}. "
        f"Case: {tags['case']}. "
        f"{quirk_text}"
    )
    return hint, tags


_GEMMA_TMPL = """Du får en dansk brugerforespørgsel med nogle GAMLE regler indbygget, plus en NY LISTE regler.

Din opgave:
1. Uddrag GRUNDOPGAVEN fra kildeforespørgslen — det som brugeren egentlig vil have (uden nogen af de gamle regler eller instruktioner om format).
2. Konstruér en NY dansk forespørgsel, der bundler GRUNDOPGAVEN sammen med DE NYE REGLER nedenfor.

STIL FOR DENNE ROW: {style_hint}

ABSOLUT KRITISKE KRAV:
- Skriv HELE outputtet på dansk. Hvis en regel er skrevet på engelsk, OVERSÆT den til dansk (bevar alle tal, nøgleord, forbudte ord, startord, afsluttende sætning eksakt — kun det bærende sprog skal skifte).
- GRUNDOPGAVEN skal fremgå TYDELIGT i den nye forespørgsel — som en genkendelig sætning eller spørgsmål. Den må IKKE bare skrives ind i reglerne eller forsvinde.
- ALLE {n_rules} regler skal være til stede i den nye forespørgsel — ingen må udelades. Læseren skal kunne genfinde hver eneste regel.
- Bevar HVERT indholdsmæssigt krav EKSAKT: samme tal, samme nøgleord, samme forbud, samme afslutningsfrase, samme startord. Reglerne må omformuleres sprogligt, men kravet må IKKE ændres.
- Tilføj ABSOLUT INGEN nye regler eller krav som ikke står på listen. Ingen ekstra "brug procenttegn", "afslut med spørgsmålstegn", osv. — kun præcis de {n_rules} regler der er givet.
- Din output skal indeholde præcis de {n_rules} regler, hverken flere eller færre.
- Returnér KUN JSON med feltene {{"base_task": "...", "new_prompt": "..."}}.

--- KILDEFORESPØRGSEL ---
{orig_prompt}

--- NYE REGLER (præcis {n_rules} stk, i tilfældig rækkefølge — placér dem ifølge stilen ovenfor) ---
{rules_block}

Producér JSON nu."""


def build_rules_block(combo: list[dict]) -> str:
    lines = []
    for i, r in enumerate(combo, 1):
        lines.append(f"{i}. [{r['name']}] {r['describe']}")
    return "\n".join(lines)


# ── Constraint-in-prompt verification ────────────────────────────────────────
# Keyword-based check that each requested constraint has some paraphrase in the
# generated prompt. Used to trigger a retry when Gemini drops a rule during
# rewriting. Kept small and Danish-first (with diacritics stripped to match
# both "inkludér" and "inkluder").
_CONSTRAINT_KEYWORDS = {
    "google:keywords:existence":            ["nogleord","inkluder","brug ord","include","brug ordet","hvert af folgende"],
    "google:keywords:frequency":            ["nojagtig","praecis","gange","exactly","times"],
    "google:keywords:letter_frequency":     ["bogstav","letter","gange"],
    "google:keywords:forbidden_words":      ["forbudt","ikke bruge","ma ikke","forbidden","avoid","uden at bruge","undga","undlad"],
    "google:length_constraints:number_words":       ["ord","words"],
    "google:length_constraints:number_sentences":   ["saetning","sentence","punktum"],
    "google:length_constraints:number_paragraphs":  ["afsnit","paragraph"],
    "google:length_constraints:nth_paragraph_first_word": ["afsnit","starte","begynde","first word"],
    "google:detectable_format:number_bullet_lists": ["punktopstil","bullet","*"],
    "google:detectable_format:number_highlighted_sections": ["fremhaev","kursiv","*","italic","highlight"],
    "google:detectable_format:multiple_sections":   ["sektion","section","SECTION"],
    "google:detectable_format:json_format":         ["json"],
    "google:detectable_format:title":               ["titel","title","<<"],
    "google:detectable_format:constrained_response":["muligheder","'My answer","'Mit svar"],
    "google:detectable_content:number_placeholders":["pladsholder","placeholder","["],
    "google:detectable_content:postscript":         ["p.s.","postscript","nb:"],
    "google:combination:repeat_prompt":             ["gentag","repeat"],
    "google:combination:two_responses":             ["to svar","two responses","******","to versioner","to udkast"],
    "google:startend:quotation":                    ['"',"«","»","anforselstegn","quotation"],
    "google:startend:end_checker":                  ["afslut","slutter","end with"],
    "google:change_case:english_capital":           ["store bogstaver","capital"],
    "google:change_case:english_lowercase":         ["sma bogstaver","lowercase"],
    "google:change_case:capital_word_frequency":    ["store bogstaver","capital"],
    "google:punctuation:no_comma":                  ["komma","commas","kommaer"],
    "in_second_person":                             ["du ","dig","din"],
    "no_first_person":                              ["jeg","vi","forste person","first person"],
    "first_sentence_max_words":                     ["forste saetning","first sentence"],
    "contains_year":                                ["ar","arstal","year"],
    "contains_percentage":                          ["procent","percent","%"],
    "markdown_table":                               ["tabel","table","|"],
    "starts_with_phrase":                           ["start","begin","begynd"],
    "ends_with_punctuation":                        ["tegn","punctuation"],
}

def _norm_for_verify(s: str) -> str:
    import unicodedata as _u
    s = _u.normalize("NFKD", s or "")
    return "".join(c for c in s if _u.category(c) != "Mn").lower()

def _find_missing_constraints(new_prompt: str, combo: list[dict]) -> list[dict]:
    """Return sublist of combo entries whose keyword hint isn't found in new_prompt."""
    p_norm = _norm_for_verify(new_prompt)
    missing = []
    for r in combo:
        kws = _CONSTRAINT_KEYWORDS.get(r["name"])
        if kws is None:
            continue  # unknown constraint, skip check
        if not any(_norm_for_verify(kw) in p_norm for kw in kws):
            missing.append(r)
    return missing


# ── main worker ──────────────────────────────────────────────────────────────

async def process_row(idx: int, row: dict, rng: random.Random, args) -> dict | None:
    orig_prompt = row["messages"][0]["content"]
    combo = sample_valid_combo(rng, args.min_rules, args.max_rules, args.google_frac)
    if not combo:
        return None
    rules_block = build_rules_block(combo)
    style_hint, style_tags = sample_style_hint(rng)
    prompt = _GEMMA_TMPL.format(orig_prompt=orig_prompt[:2500],
                                rules_block=rules_block,
                                style_hint=style_hint,
                                n_rules=len(combo))
    result = await call_gemma(prompt)
    if not result:
        return None
    base_task = (result.get("base_task") or "").strip()
    new_prompt = (result.get("new_prompt") or "").strip()
    if not base_task or not new_prompt or len(new_prompt) < 30:
        return None

    # Verification retry: if Gemini dropped any constraint, retry ONCE with
    # explicit reminders. Bounded to a single retry to keep cost predictable
    # (worst-case 2× LLM calls per row; expected ~30% × 2 = 1.3× on average
    # based on the 29% first-pass drop rate).
    missing = _find_missing_constraints(new_prompt, combo)
    if missing:
        missing_block = "\n".join(f"- [{r['name']}] {r['describe']}" for r in missing)
        retry_prompt = (
            prompt
            + "\n\n=== ADVARSEL — REGLER DER MANGLER I DIT SIDSTE FORSØG ===\n"
            + f"Du udelod følgende {len(missing)} regel(er) fra new_prompt:\n"
            + missing_block
            + "\n\nSkriv new_prompt igen med ALLE regler klart til stede. "
              "Behold din struktur, men tilføj de manglende regler eksplicit."
        )
        result2 = await call_gemma(retry_prompt)
        if result2:
            bt2 = (result2.get("base_task") or "").strip()
            np2 = (result2.get("new_prompt") or "").strip()
            if bt2 and np2 and len(np2) >= 30:
                still_missing = _find_missing_constraints(np2, combo)
                # Accept retry only if it's an improvement (missing set shrank)
                if len(still_missing) < len(missing):
                    base_task, new_prompt = bt2, np2
    return {
        "task": base_task,
        "prompt": new_prompt,
        "constraints": [r["name"] for r in combo],
        "params": [r["params"] for r in combo],
        "n_rules": len(combo),
        "style_tags": json.dumps(style_tags, ensure_ascii=False),
        "source": row.get("source", "v4"),
        "old_constraints": row.get("constraints", []),
        "row_idx": idx,
    }


async def bounded(sem, coro):
    async with sem:
        return await coro


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output HF dataset dir")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--min-rules", type=int, default=1)
    ap.add_argument("--max-rules", type=int, default=5)
    ap.add_argument("--google-frac", type=float, default=0.5,
                    help="Approx fraction of google-schema rules per combo (0=none, 1=all)")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source-cap", type=int, default=None,
                    help="Cap v4 rows to sample from (default all)")
    args = ap.parse_args()

    print("Loading v4 source rows...", flush=True)
    ds = load_dataset("jensjepsen/danish-instruction-following-v4",
                      "default", split="train")
    print(f"  loaded {len(ds)} v4 rows", flush=True)

    rng = random.Random(args.seed)
    idx_pool = list(range(len(ds)))
    rng.shuffle(idx_pool)
    if args.source_cap:
        idx_pool = idx_pool[:args.source_cap]

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path / "rows.jsonl"

    n_target = args.n
    n_ok = 0
    n_err = 0
    t0 = time.time()

    sem = asyncio.Semaphore(args.concurrency)

    # Process in chunks so we can flush progress + write jsonl streaming
    CHUNK = 500
    with open(jsonl_path, "w", encoding="utf-8") as fout:
        pool_i = 0
        while n_ok < n_target and pool_i < len(idx_pool):
            batch_ids = []
            while len(batch_ids) < CHUNK and pool_i < len(idx_pool) and n_ok + len(batch_ids) < n_target * 2:
                batch_ids.append(idx_pool[pool_i]); pool_i += 1
            tasks = []
            for i in batch_ids:
                row = ds[i]
                sub_rng = random.Random(args.seed + i)
                tasks.append(bounded(sem, process_row(i, row, sub_rng, args)))
            results = await asyncio.gather(*tasks)
            for r in results:
                if r is None:
                    n_err += 1
                    continue
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_ok += 1
                if n_ok >= n_target:
                    break
            fout.flush()
            dt = time.time() - t0
            rate = n_ok / max(1e-6, dt)
            eta = (n_target - n_ok) / max(1e-6, rate)
            print(f"  {n_ok}/{n_target}  ok={n_ok} err={n_err}  "
                  f"rate={rate:.1f}/s  elapsed={dt:.0f}s  eta={eta:.0f}s",
                  flush=True)

    print(f"\nTotal ok={n_ok} err={n_err} in {time.time()-t0:.0f}s", flush=True)
    print(f"jsonl at {jsonl_path}", flush=True)

    print("Loading jsonl → HF Dataset ...", flush=True)
    def _iter():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    ds_out = Dataset.from_list(list(_iter()))
    save_dir = str(out_path / "hf")
    ds_out.save_to_disk(save_dir)
    print(f"Saved HF dataset to {save_dir}  n={len(ds_out)}", flush=True)
    print(f"[combo] rejection-sampler give-up count: {_COMBO_GIVEUP_CT} "
          f"(fell back to invalid combo after 20 retries)", flush=True)

    if _SESSION is not None:
        await _SESSION.close()


if __name__ == "__main__":
    asyncio.run(main())
