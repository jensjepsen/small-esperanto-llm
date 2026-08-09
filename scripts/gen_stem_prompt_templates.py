"""Generate diverse Danish prompt-template pools for the STEM SFT build.

Two wrapper flavors (open-book with article, closed-book without) + one
task-instruction pool per subtype. Each pool is 20-25 Gemini-generated
variants covering register, length, structure, politeness.

Composition later (in the flattener):
    open  : wrapper_open.format(title=..., text=..., TASK=task_pool.format(**fields))
    closed: wrapper_closed.format(TASK=task_pool.format(**fields))
"""
from __future__ import annotations
import argparse, asyncio, json, re
from pathlib import Path
import aiohttp

MODEL = "google/gemini-2.5-flash-lite"
API = "https://openrouter.ai/api/v1/chat/completions"

POOLS = {
    "wrapper_open": dict(
        n=25, placeholders="{title}, {text}, {TASK}",
        semantic=(
            "Wrapper der placerer en dansk Wikipedia-artikel og en TASK-"
            "instruktion i én prompt. TASK kan komme FØR eller EFTER teksten. "
            "Titel er OPTIONAL. Brug forskellige overskrifter (ARTIKEL, Tekst, "
            "Passage, Uddrag, ingen overskrift), delimitere (===, ---, tom linje, ingen), "
            "og sproglige registre (formel, neutral, letlæst, direkte)."
        ),
        example='"ARTIKEL: {text}\\n\\n{TASK}"',
    ),
    "wrapper_closed": dict(
        n=20, placeholders="{TASK}",
        semantic=(
            "Wrapper til CLOSED-BOOK opgaver — INGEN artikel medfølger. "
            "Kun opgaven/spørgsmålet. Kan indeholde høflighed, kontekstuel "
            "framing ('som fysiker', 'hjælp mig'), eller være helt direkte."
        ),
        example='"{TASK}"',
    ),
    "stem_worked_calc": dict(
        n=20, placeholders="{q}",
        semantic=(
            "KRITISK: {q} er ALLEREDE et komplet, velformuleret problem — "
            "typisk med tal og en implicit opgave. Skabelonen skal blot rammesætte "
            "det (fx bede om trin, formler eller bare præsentere det). "
            "MÅ IKKE tilføje interrogativer som 'hvad er', 'hvor mange' — de er "
            "allerede i {q}. Varier fra bar '{q}' til høflige framings."
        ),
        example='"Løs følgende og vis dine trin: {q}"',
    ),
    "stem_mechanism": dict(
        n=20, placeholders="{q}",
        semantic=(
            "KRITISK: {q} er ALLEREDE et 'hvordan'-spørgsmål ('Hvordan virker X?', "
            "'Hvordan foregår Y?'). Skabelonen skal rammesætte anmodningen om en "
            "trin-forklaring — MÅ IKKE tilføje 'hvordan' igen (det giver dobbelt-"
            "interrogativ). Kan tilføje 'trin for trin', 'som en proces' eller bare "
            "vise {q} bart."
        ),
        example='"Besvar dette med en trin-for-trin forklaring: {q}"',
    ),
    "stem_counterfactual": dict(
        n=20, placeholders="{q}",
        semantic=(
            "KRITISK: {q} er ALLEREDE et 'hvad hvis'-spørgsmål med et konkret scenarie. "
            "Skabelonen skal rammesætte anmodningen om et ræsoneret svar med kausal "
            "analyse — MÅ IKKE tilføje 'hvad ville ske hvis' igen. Kan bede om 'forklar "
            "årsagen', 'analysér effekterne', eller bare vise {q} bart."
        ),
        example='"Analysér dette scenarie: {q}"',
    ),
    "stem_fact_check": dict(
        n=20, placeholders="{claim}",
        semantic=(
            "Måde at bede om SAND/FALSK-verifikation af en påstand med begrundelse. "
            "Variér længde og register. Nogle skal være meget direkte ('SAND eller "
            "FALSK: {claim}'), andre mere formelle."
        ),
        example='"Er følgende påstand sand eller falsk? Begrund. Påstand: {claim}"',
    ),
}

PROMPT_TEMPLATE = """Du er en dansk prompt-designer. Generér præcis {n} DIVERSE skabelon-strings der alle udfører den samme rolle.

Rolle: {semantic}

Tilladte pladsholdere (Python .format-style): {placeholders}
Eksempel: {example}

STRENGE REGLER:
  * Nøjagtig {n} unikke skabeloner. Ingen duplikater eller minor variationer.
  * Variér: længde (kort ↔ verbose), register (formel/neutral/letlæst/direkte/høflig), struktur, tegnsætning, ordvalg.
  * Brug KUN de tilladte pladsholdere. Ingen andre curly-braces.
  * Alle på flydende dansk.
  * Output KUN gyldig JSON: {{"templates": ["...", "...", ...]}}. Ingen markdown.
"""

PARSE_RE = re.compile(r"\{.*\}", re.S)


def _relax(s):
    valid = set('"\\/bfnrtu')
    out, in_str, esc = [], False, False
    for c in s:
        if esc:
            if in_str and c not in valid: out.append("\\")
            out.append(c); esc = False; continue
        if c == "\\": out.append(c); esc = True; continue
        if c == '"': in_str = not in_str; out.append(c); continue
        if in_str and c in "\n\r\t":
            out.append({"\n":"\\n","\r":"\\r","\t":"\\t"}[c])
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
    try: obj = json.loads(js)
    except json.JSONDecodeError:
        try: obj = json.loads(_relax(js))
        except json.JSONDecodeError: return None
    ts = obj.get("templates")
    if not isinstance(ts, list): return None
    return [t for t in ts if isinstance(t, str) and t.strip()]


async def gen_pool(session, key, name, spec):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(**spec)}],
        "temperature": 0.9, "max_tokens": 2500,
        "provider": {"order": ["Google AI Studio", "Google"], "allow_fallbacks": True},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://claude-code-stem-tpl", "X-Title": "DA-STEM-Templates"}
    for attempt in range(3):
        async with session.post(API, headers=headers, json=body, timeout=90) as resp:
            data = await resp.json()
        if "choices" not in data:
            if attempt < 2: await asyncio.sleep(2**attempt); continue
            return name, None, f"api:{json.dumps(data)[:150]}"
        raw = data["choices"][0]["message"]["content"]
        parsed = parse(raw)
        if parsed is None:
            if attempt < 2: await asyncio.sleep(2**attempt); continue
            return name, None, f"parse_fail: {raw[:200]}"
        # Filter: keep only templates with allowed placeholders
        allowed = set(re.findall(r"\{(\w+)\}", spec["placeholders"]))
        clean = []
        for t in parsed:
            used = set(re.findall(r"\{(\w+)\}", t))
            if used - allowed:  # has invalid placeholder
                continue
            clean.append(t)
        return name, clean, None
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
                print(f"  {name}: {len(templates)} templates (allowed placeholders)",
                      flush=True)
                results[name] = templates

    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
