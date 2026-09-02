"""Generate diverse Danish prompt-template pools via Gemini.

For each pool, ask Gemini to produce N variants covering register (formal ↔
casual), length (terse ↔ verbose), politeness (imperative ↔ polite), and
structure (leading vs trailing task, delimiter choice). Templates are Python
.format() strings with the specified placeholders.

Composition (used later by the flattener):
    prompt = wrapper.format(title=title, text=text, TASK=<task_variant>)
where <task_variant> = task_pool[subtype].format(**subtype_fields).

Output: data/task_expansion_v1/prompt_templates.json (dict of pool→list[str]).

Usage:
    uv run --no-project --with aiohttp python scripts/gen_prompt_templates.py \\
        --out data/task_expansion_v1/prompt_templates.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import aiohttp


MODEL = "google/gemini-2.5-flash-lite"
API = "https://openrouter.ai/api/v1/chat/completions"


POOLS = {
    "wrapper": dict(
        n=25,
        placeholders="{title}, {text}, {TASK}",
        semantic=(
            "Wrapper skabelon der placerer en dansk Wikipedia-artikel og en TASK-"
            "instruktion i én prompt. TASK kan komme FØR eller EFTER teksten. Titel er "
            "OPTIONAL. Brug forskellige overskrifter (ARTIKEL, Tekst, Passage, Uddrag, "
            "ingen overskrift), delimiter-stile (===, ---, tomme linjer, ingen), og "
            "sproglige registre (formel, neutral, letlæst, direkte)."
        ),
        example='"ARTIKEL: {text}\\n\\n{TASK}"',
    ),
    "rc_qa": dict(
        n=20,
        placeholders="{q}",
        semantic=(
            "Måde at introducere en læseforståelses-spørgsmål på. Varierer fra bare "
            "spørgsmålet selv, over 'Q:'-prefix, til høflige forespørgsler."
        ),
        example='"Besvar spørgsmålet: {q}"',
    ),
    "reason_qa": dict(
        n=20,
        placeholders="{q}",
        semantic=(
            "Måde at bede om et resoneret svar (kausal, analogi, ranking, multi-step, "
            "argumentation). Kan bede eksplicit om at 'vise trin', 'forklare hvorfor', "
            "eller bare stille spørgsmålet."
        ),
        example='"Reflektér over følgende og svar: {q}"',
    ),
    "reason_fact_check": dict(
        n=20,
        placeholders="{claim}",
        semantic=(
            "Måde at bede om en SAND/FALSK-vurdering af en påstand med begrundelse "
            "ifølge artiklen. Varier fra kort 'Sand eller falsk: {claim}' til "
            "længere formuleringer."
        ),
        example='"Er følgende påstand sand eller falsk ifølge teksten? Begrund. Påstand: {claim}"',
    ),
    "textman_summary": dict(
        n=20,
        placeholders="",
        semantic=(
            "Måde at bede om et 3-bullet resumé på. Varier ordlyd ('resumér', "
            "'sammenfat', 'lav bullet points', 'kort opsummering'), længde og "
            "punktumformater."
        ),
        example='"Skriv 3 bulletpoints der opsummerer artiklens hovedpointer."',
    ),
    "textman_rewrite": dict(
        n=20,
        placeholders="",
        semantic=(
            "Måde at bede om en omskrivning af artiklens første afsnit i egne ord "
            "(bevar information, skift formulering). Varier længde og register."
        ),
        example='"Omskriv de første par afsnit i dine egne ord."',
    ),
    "textman_style_transfer": dict(
        n=20,
        placeholders="{style}",
        semantic=(
            "Måde at bede om at omskrive artiklens første afsnit i en bestemt stil "
            "(f.eks. formel akademisk, afslappet talesprog, letlæst). Stil-navnet "
            "kommer i {style}."
        ),
        example='"Omskriv artiklens indledning i {style} stil."',
    ),
    "textman_elaborate": dict(
        n=20,
        placeholders="",
        semantic=(
            "Måde at bede om at vælge en kort passage fra artiklen og udvide den til "
            "et længere afsnit med baggrund/eksempler. Svaret skal angive kilde-"
            "passagen separat fra den udvidede version."
        ),
        example='"Vælg en kort passage i artiklen og udvid den med baggrund og eksempler. Angiv først passagen, derefter den udvidede version."',
    ),
    "textman_genre_transform": dict(
        n=20,
        placeholders="{genre}",
        semantic=(
            "Måde at bede om at omskrive artiklens essens som en anden genre "
            "(nyhedsoverskrift, tweet, email, brev). Genre-navnet kommer i {genre}."
        ),
        example='"Omskriv artiklens essens som en {genre}."',
    ),
}


PROMPT_TEMPLATE = """Du er en dansk prompt-designer. Generér præcis {n} DIVERSE skabelon-strings der alle udfører den samme rolle.

Rolle: {semantic}

Tilladte pladsholdere (Python .format-style): {placeholders}
Eksempel: {example}

STRENGE REGLER:
  * Nøjagtig {n} unikke skabeloner. Ingen duplikater eller minor variationer af samme skabelon.
  * Variér: længde (kort ↔ verbose), register (formel, neutral, letlæst, direkte, høflig), struktur, tegnsætning, ordvalg.
  * Brug KUN de tilladte pladsholdere. Ingen andre. Hvis pladsholdere er tomme, brug ingen pladsholdere.
  * Alle skabeloner skal være på flydende dansk.
  * Output KUN gyldig JSON i formatet: {{"templates": ["...", "...", ...]}}. Ingen markdown, ingen preamble.
"""


PARSE_RE = re.compile(r"\{.*\}", re.S)


def _relax_json(s):
    out, in_str, esc = [], False, False
    for c in s:
        if esc: out.append(c); esc = False; continue
        if c == "\\": out.append(c); esc = True; continue
        if c == '"': in_str = not in_str; out.append(c); continue
        if in_str and c in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[c])
        else:
            out.append(c)
    return "".join(out)


def parse(raw):
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0].strip()
    m = PARSE_RE.search(t)
    if not m: return None
    js = m.group(0)
    try:
        obj = json.loads(js)
    except json.JSONDecodeError:
        try: obj = json.loads(_relax_json(js))
        except json.JSONDecodeError: return None
    ts = obj.get("templates")
    if not isinstance(ts, list): return None
    return [t for t in ts if isinstance(t, str) and t.strip()]


async def gen_pool(session, key, name, spec):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(**spec)}],
        "temperature": 0.9,  # high — want diversity
        "max_tokens": 2500,
        "provider": {"order": ["Google AI Studio", "Google"], "allow_fallbacks": True},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://claude-code-prompt-gen",
        "X-Title": "DA-Prompt-Templates",
    }
    for attempt in range(3):
        async with session.post(API, headers=headers, json=body, timeout=90) as resp:
            data = await resp.json()
        if "choices" not in data:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt); continue
            return name, None, f"api:{json.dumps(data)[:180]}"
        raw = data["choices"][0]["message"]["content"]
        parsed = parse(raw)
        if parsed is None:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt); continue
            return name, None, f"parse_fail: {raw[:200]}"
        return name, parsed, None
    return name, None, "retry_exhausted"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    args = ap.parse_args()

    key = args.key_file.read_text().strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    async with aiohttp.ClientSession() as session:
        tasks = [gen_pool(session, key, name, spec) for name, spec in POOLS.items()]
        for coro in asyncio.as_completed(tasks):
            name, templates, err = await coro
            if err:
                print(f"  {name}: ERROR {err}", flush=True)
            else:
                print(f"  {name}: {len(templates)} templates", flush=True)
                results[name] = templates

    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
