"""Generate Danish tool-call SFT data.

Two-axis design:
  Axis 1 — structural template (grammar over n_params, types, constraints,
           nesting). Guarantees coverage over schema shapes.
  Axis 2 — domain scenario (large seeded pool). Guarantees semantic breadth.

Pipeline per row:
  1. Sample (structural_template, scenario) independently.
  2. Gemini → invent a Danish tool matching that shape for that scenario.
  3. Sample 0-4 distractor tools from cached pool.
  4. Sample difficulty bucket (verbatim / inference / defaults / clarify /
     refuse / multi-chain).
  5. Gemini → user utterance + reasoning + tool call JSON.
  6. Validate: parse JSON, function in catalog, args pass jsonschema.
  7. Reject & retry on failure.

Output row shape (JSONL):
  {"messages": [{"role":"user","content":<catalog + utterance>},
                {"role":"assistant","content":<reasoning + JSON call>}],
   "tool_catalog": [<full catalog>],
   "target_tool": "book_bord",
   "difficulty": "inference",
   "scenario": "restaurant",
   "structural_hash": "3req-1opt-enum",
   "attempts": 2}

Usage:
    export GEMINI_API_KEY=...  # or ~/.gem file
    python scripts/gen_tool_call_sft.py \\
        --seed-scenarios data/tool_calls/scenarios_seed.jsonl \\
        --out data/tool_calls/v1.jsonl \\
        --n-rows 40000 --concurrency 60
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Local imports (validator + vocab).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tool_call_vocab import (  # noqa: E402
    validate_call, tool_schema_to_json_schema,
)


# ────────────────────────────────────────────────────────────────────────────
# Gemini client (matches if_generate.py pattern)
# ────────────────────────────────────────────────────────────────────────────

MODEL_ID = os.environ.get("TOOL_MODEL_ID", "gemini-3.1-flash-lite")
_CLIENT = None
_CFG = None
_OR_SESSION = None

# Running counters for OpenRouter cost tracking. Populated by _call_openrouter
# when the model routes through OR (the response's `usage.cost` field is USD).
_OR_COST_USD = 0.0
_OR_CALLS = 0
_OR_INPUT_TOK = 0
_OR_OUTPUT_TOK = 0


def _read_key_file(names: list[str]) -> str | None:
    for name in names:
        for p in [Path.home() / name, Path.home() / f".{name}"]:
            if p.exists():
                return p.read_text().strip()
    return None


def _is_openrouter_model(mid: str) -> bool:
    return "/" in mid


async def _call_gemini(prompt: str) -> str | None:
    global _CLIENT, _CFG
    if _CLIENT is None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise SystemExit("pip install google-genai")
        key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
               or _read_key_file(["gem", "gemini_key"]))
        if not key:
            raise SystemExit("No GOOGLE_API_KEY / GEMINI_API_KEY and no ~/gem file.")
        _CLIENT = genai.Client(api_key=key)
        _CFG = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
    try:
        resp = await _CLIENT.aio.models.generate_content(
            model=MODEL_ID, contents=prompt, config=_CFG)
        return (resp.text or "").strip() or None
    except Exception as e:
        print(f"  gemini error: {type(e).__name__}: {str(e)[:120]}",
              file=sys.stderr)
        return None


async def _call_openrouter(prompt: str) -> str | None:
    """OpenRouter chat-completions call. Model IDs with '/' route here."""
    global _OR_SESSION
    import aiohttp
    if _OR_SESSION is None:
        key = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY")
               or _read_key_file(["or", "openrouter"]))
        if not key:
            raise SystemExit("No OPENROUTER_API_KEY set and no ~/or file.")
        _OR_SESSION = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://claude-code-tool-call",
                     "X-Title": "danish-tool-call-generation"},
            connector=aiohttp.TCPConnector(limit=1000, limit_per_host=0),
            timeout=aiohttp.ClientTimeout(total=120))
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2000,
        # OR extension: response.usage will include prompt/completion tokens
        # AND a `cost` field (USD) for the exact price of this generation.
        "usage": {"include": True},
        # Route to the fastest provider for this model — otherwise OR
        # sometimes lands us on a slow replica and caps throughput.
        "provider": {"sort": "throughput"},
    }
    try:
        async with _OR_SESSION.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"  or {resp.status}: {text[:200]}", file=sys.stderr)
                return None
            data = await resp.json()
            # Accumulate cost + token stats globally so the run summary
            # can report actual $ spent, not just estimates.
            global _OR_COST_USD, _OR_CALLS, _OR_INPUT_TOK, _OR_OUTPUT_TOK
            usage = data.get("usage") or {}
            _OR_COST_USD += float(usage.get("cost", 0.0) or 0.0)
            _OR_INPUT_TOK += int(usage.get("prompt_tokens", 0) or 0)
            _OR_OUTPUT_TOK += int(usage.get("completion_tokens", 0) or 0)
            _OR_CALLS += 1
            return (data["choices"][0]["message"]["content"] or "").strip() or None
    except Exception as e:
        print(f"  or error: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return None


async def call_gemini(prompt: str) -> str | None:
    """Route by MODEL_ID: '/' → OpenRouter, else → Google GenAI."""
    if _is_openrouter_model(MODEL_ID):
        return await _call_openrouter(prompt)
    return await _call_gemini(prompt)


# ────────────────────────────────────────────────────────────────────────────
# Structural template grammar (Axis 1)
# ────────────────────────────────────────────────────────────────────────────

TYPES = ["tekst", "heltal", "tal", "boolsk",
         "liste_af_tekst", "liste_af_heltal", "objekt", "enum_valg"]

# Marginals — tune to keep low-frequency shapes present but not dominant.
TYPE_WEIGHTS = {
    "tekst":            0.35,
    "heltal":           0.18,
    "tal":              0.10,
    "boolsk":           0.10,
    "liste_af_tekst":   0.09,
    "liste_af_heltal":  0.04,
    "objekt":           0.09,   # nested object
    "enum_valg":        0.05,   # string with enum
}

N_PARAM_WEIGHTS = {
    0: 0.02, 1: 0.10, 2: 0.18, 3: 0.22, 4: 0.18,
    5: 0.14, 6: 0.08, 7: 0.05, 8: 0.03,
}


def _weighted_choice(rng: random.Random, weights: dict) -> Any:
    items = list(weights.items())
    keys = [k for k, _ in items]
    ws = [w for _, w in items]
    return rng.choices(keys, weights=ws, k=1)[0]


def sample_template(rng: random.Random) -> dict:
    """Return a structural template spec.

    Shape:
        {"n_params": 4,
         "params": [
             {"slot": 0, "type": "tekst", "required": True,
              "decorators": ["længde", "mønster"]},
             {"slot": 1, "type": "heltal", "required": True,
              "decorators": ["min_maks"]},
             {"slot": 2, "type": "enum_valg", "required": False,
              "decorators": []},
             {"slot": 3, "type": "objekt", "required": False,
              "decorators": [],
              "sub_params": [
                  {"type": "tekst", "required": True, "decorators": []},
                  {"type": "heltal", "required": False, "decorators": []},
              ]},
         ]}
    """
    n = _weighted_choice(rng, N_PARAM_WEIGHTS)
    params = []
    for i in range(n):
        t = _weighted_choice(rng, TYPE_WEIGHTS)
        req = rng.random() < 0.60
        decorators = []
        if t in ("heltal", "tal") and rng.random() < 0.45:
            decorators.append("min_maks")
        if t == "tekst" and rng.random() < 0.20:
            decorators.append("længde")
        if t == "tekst" and rng.random() < 0.10:
            decorators.append("mønster")
        if t == "tekst" and rng.random() < 0.05:
            decorators.append("format")
        p = {"slot": i, "type": t, "required": req, "decorators": decorators}
        if t == "objekt":
            sub_n = rng.randint(2, 3)
            p["sub_params"] = []
            for _ in range(sub_n):
                st = _weighted_choice(rng, {
                    "tekst": 0.5, "heltal": 0.25, "tal": 0.1,
                    "boolsk": 0.10, "enum_valg": 0.05,
                })
                p["sub_params"].append(
                    {"type": st, "required": rng.random() < 0.7,
                     "decorators": []})
        params.append(p)
    return {"n_params": n, "params": params}


def template_hash(tpl: dict) -> str:
    """Short structural fingerprint for stats/logging."""
    parts = []
    for p in tpl["params"]:
        req = "R" if p["required"] else "O"
        deco = "".join(sorted(d[0] for d in p.get("decorators", [])))
        sub = ""
        if p.get("sub_params"):
            sub = "[" + ",".join(sp["type"][0] for sp in p["sub_params"]) + "]"
        parts.append(f"{p['type'][:3]}-{req}{deco}{sub}")
    return "|".join(parts) if parts else "empty"


def render_template_for_llm(tpl: dict) -> str:
    """Compact human-readable rendering of the structural template.

    Used inside the tool-invention prompt so Gemini knows what shape to
    produce. We avoid over-prescribing param semantics — Gemini picks names
    matching the scenario.
    """
    if not tpl["params"]:
        return "(ingen parametre — funktionen tager ingen argumenter)"
    lines = []
    for p in tpl["params"]:
        req = "påkrævet" if p["required"] else "valgfri"
        deco_txt = ""
        if p["decorators"]:
            deco_map = {"min_maks": "min/maks-grænser",
                        "længde": "længdebegrænsning",
                        "mønster": "regex-mønster",
                        "format": "format-hint (fx dato, email)"}
            deco_txt = " + " + ", ".join(deco_map.get(d, d)
                                         for d in p["decorators"])
        line = f'  {p["slot"]+1}. {p["type"]} ({req}){deco_txt}'
        if p.get("sub_params"):
            for sp in p["sub_params"]:
                sr = "påkrævet" if sp["required"] else "valgfri"
                line += f'\n     └── {sp["type"]} ({sr})'
        lines.append(line)
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Tool invention prompt (Axis 2 × Axis 1)
# ────────────────────────────────────────────────────────────────────────────

TOOL_INVENT_PROMPT = """Du er en API-designer, der opfinder værktøjer til at kalde fra en LLM.

SCENARIO: {scenario}

Opfind ét realistisk værktøj til dette scenario. Værktøjet skal have præcis følgende struktur:

Antal parametre: {n_params}
Parametre:
{template}

VIGTIGE REGLER:
- Alt SKAL være på DANSK: værktøjsnavn (snake_case, må gerne bruge æøå), parameternavne, beskrivelser.
- Værktøjsnavnet skal starte med et verbum (fx book_, hent_, opret_, opdater_, slet_, søg_, beregn_, send_).
- Parameternavne skal være beskrivende og konkrete for scenariet.
- Beskrivelser skal være korte (1 linje pr. parameter).
- Hvis en parameter er af typen "enum_valg", angiv 3-6 realistiske DANSKE valgmuligheder.
- Hvis en parameter har min_maks-grænser, vælg realistiske tal (fx alder 0-120).
- Hvis en parameter har regex-mønster, angiv et realistisk mønster (fx dato ^\\d{{4}}-\\d{{2}}-\\d{{2}}$).
- Hvis en parameter har format, brug: dato, tid, dato_tid, email, telefon, url.

Svar KUN med et JSON-objekt i dette format (ingen forklaring, ingen markdown-fence):

{{
  "navn": "værktøjsnavn",
  "beskrivelse": "Kort beskrivelse af hvad værktøjet gør.",
  "parametre": {{
    "parameter_1": {{"type": "tekst", "påkrævet": true, "beskrivelse": "..."}},
    ...
  }}
}}

For typer:
- "tekst" → JSON-string
- "heltal" → JSON-integer
- "tal" → JSON-number (decimal)
- "boolsk" → JSON-boolean
- "liste_af_tekst" → JSON-array af strings
- "liste_af_heltal" → JSON-array af integers
- "objekt" → nested objekt, tilføj felt "egenskaber": {{...}}
- "enum_valg" → tekst med felt "valg": ["mulighed1", ...]

For dekoratorer:
- min_maks → felt "min" og/eller "maks"
- længde → felt "længde_min" og/eller "længde_maks"
- mønster → felt "mønster" (regex-streng)
- format → felt "format" (en af ovennævnte hint-strings)

Producer nu værktøjet:"""


# ────────────────────────────────────────────────────────────────────────────
# Call generation prompt
# ────────────────────────────────────────────────────────────────────────────

DIFFICULTY_INSTRUCTIONS = {
    "verbatim": (
        "Brugerens tekst indeholder alle nødvendige værdier ORDRET (fx tider som "
        "'19:00', tal som skrevne cifre, navne i citationstegn). Model'en behøver "
        "ikke at gætte eller konvertere noget."),
    "inference": (
        "Brugerens tekst udtrykker værdierne på naturligt dansk der KRÆVER "
        "omformning: 'syv i aften' → '19:00', 'fire' → 4, 'om en uge' → en ISO-dato, "
        "'nogle stykker' → 3-4. Model'en skal fortolke intentionen."),
    "defaults": (
        "Brugeren udelader en eller flere VALGFRIE parametre. Kun de påkrævede skal "
        "være med i kaldet. Ingen forklaring i argumenterne."),
    "clarify": (
        "Brugeren mangler information til en PÅKRÆVET parameter. Model'en skal "
        "IKKE producere et JSON-kald; i stedet skal assistant stille et præcist "
        "uddybende spørgsmål på dansk om den manglende oplysning."),
    "refuse": (
        "Brugerens forespørgsel er UDEN FOR alle værktøjernes formål (fx spørger "
        "om noget helt andet). Model'en skal IKKE producere et kald; assistant skal "
        "høfligt forklare at værktøjerne ikke matcher forespørgslen."),
    "multi_chain": (
        "Brugeren beskriver en opgave der kræver TO kald i rækkefølge til samme "
        "eller forskellige værktøjer. Assistant emitterer to JSON-objekter, ét pr. "
        "linje, i den rigtige rækkefølge."),
}

CALL_GEN_PROMPT = """Du er en dansk sprogmodel, der bruger værktøjer efter brugerens ønske.

VÆRKTØJSKATALOG:
{catalog_json}

MÅLVÆRKTØJ: "{target}"
SVÆRHEDSGRAD: {difficulty}
{difficulty_instructions}

Producer ét træningseksempel bestående af:
  1. En dansk brugerhenvendelse (1-3 sætninger).
  2. En dansk begrundelse fra assistant (1-3 korte sætninger om hvilket
     værktøj der vælges og hvorfor, samt hvordan værdierne udledes).
  3. Selve værktøjskaldet som ét JSON-objekt (eller INTET kald for
     sværhedsgraderne "clarify" og "refuse").

SVAR-FORMAT: én linje JSON med tre felter, INGEN markdown, INGEN forklaring:

{{"user": "<brugerens tekst>",
  "reasoning": "<assistant's begrundelse på dansk>",
  "call": {{"navn": "{target}", "argumenter": {{...}}}}
}}

VIGTIGE REGLER:
- Alt tekst på DANSK.
- Argumenterne skal PRÆCIS matche målværktøjets skema (påkrævede felter er obligatoriske; typer skal matche).
- For "clarify" og "refuse": sæt "call" til null.
- For "multi_chain": sæt "call" til en LISTE af to JSON-kald.
- BRUG IKKE parametre der ikke er i skemaet.
- Vær realistisk — brugerens tekst skal lyde naturlig.

Producer nu træningseksemplet:"""


FOLLOWUP_GEN_PROMPT = """Du er en dansk sprogmodel i midten af en agent-loop.

VÆRKTØJSKATALOG:
{catalog_json}

BRUGERHENVENDELSE:
{user_utterance}

DIT KALD:
{call_json}

Producer nu:
  1. Et realistisk værktøjsresultat (som selve backend'en ville returnere) — JSON.
  2. En dansk naturligsproglig opfølgning (1-3 sætninger) hvor du bruger resultatet til at svare brugeren.

REGLER:
- Resultatet skal give mening for det pågældende værktøj (fx booking → bekræftelses-ID + tidspunkt; opslag → data-objektet; oprettelse → nyt ID + status).
- Ved fejl-scenarier er det OK at returnere et fejl-objekt ("fejl": "..."), men vær sparsom med det (fx <15% af tilfælde).
- Opfølgningen skal SPECIFIKT nævne værdier fra resultatet (bekræftelses-ID, tal, navne, tidspunkter).
- Alt tekst på DANSK.
- Ved flere kald (LISTE): returner en LISTE af resultater i samme rækkefølge, og lad opfølgningen dække dem alle.

SVAR-FORMAT: én linje JSON med to felter, INGEN markdown:

{{"result": <JSON-værdi ELLER liste ved flere kald>,
  "followup": "<dansk svar til brugeren>"
}}

Producer nu resultat og opfølgning:"""


# ────────────────────────────────────────────────────────────────────────────
# JSON parsing helpers
# ────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Any | None:
    """Extract the first top-level JSON object/array from text.

    Handles code fences, leading prose, trailing commas.
    """
    if not text:
        return None
    # Strip ```json ... ``` fences.
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Find first {...} or [...] block by bracket balance.
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    blob = text[start:i+1]
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError:
                        return None
    return None


# ────────────────────────────────────────────────────────────────────────────
# Row generation
# ────────────────────────────────────────────────────────────────────────────

DIFFICULTY_WEIGHTS = {
    "verbatim": 0.40, "inference": 0.25, "defaults": 0.15,
    "clarify": 0.10, "refuse": 0.05, "multi_chain": 0.05,
}


def format_catalog(tools: list[dict]) -> str:
    """Kept for prompt-composition use (call generation) only; NOT used for
    output rendering. Rendering lives entirely in rerender_tool_calls.py.
    """
    return json.dumps(tools, ensure_ascii=False, indent=2)


async def invent_tool(scenario: dict, tpl: dict) -> dict | None:
    """Ask Gemini to invent a tool matching scenario + template."""
    prompt = TOOL_INVENT_PROMPT.format(
        scenario=scenario["beskrivelse"],
        n_params=tpl["n_params"],
        template=render_template_for_llm(tpl),
    )
    resp = await call_gemini(prompt)
    if not resp:
        return None
    tool = _extract_json(resp)
    if not isinstance(tool, dict):
        return None
    if "navn" not in tool or "parametre" not in tool:
        return None
    # Coerce non-dict parametre (Gemini sometimes emits a list).
    if not isinstance(tool["parametre"], dict):
        return None
    return tool


def check_tool_schema_shape(tool: dict, tpl: dict) -> bool:
    """Sanity check that the invented tool has ~the right shape.

    We're lenient — Gemini sometimes shifts one param around. Require:
    - correct param count (±0, exact)
    - each param has 'type' field
    - types are a permitted vocabulary.
    """
    params = tool.get("parametre", {})
    if len(params) != tpl["n_params"]:
        return False
    allowed = set(TYPES) | {"tekst", "heltal", "tal", "boolsk", "objekt"}
    for _, spec in params.items():
        if not isinstance(spec, dict):
            return False
        t = spec.get("type", "tekst")
        if t not in allowed:
            return False
    # Round-trip through translator to catch bad constraint values.
    try:
        tool_schema_to_json_schema(tool)
    except Exception:
        return False
    return True


async def generate_call_example(
    target_tool: dict,
    catalog: list[dict],
    difficulty: str,
) -> dict | None:
    """Ask Gemini for user utterance + reasoning + JSON call."""
    prompt = CALL_GEN_PROMPT.format(
        catalog_json=format_catalog(catalog),
        target=target_tool["navn"],
        difficulty=difficulty,
        difficulty_instructions=DIFFICULTY_INSTRUCTIONS[difficulty],
    )
    resp = await call_gemini(prompt)
    if not resp:
        return None
    ex = _extract_json(resp)
    if not isinstance(ex, dict):
        return None
    if not all(k in ex for k in ("user", "reasoning", "call")):
        return None
    return ex


async def generate_followup(
    catalog: list[dict],
    user_utterance: str,
    call: Any,
) -> dict | None:
    """Ask Gemini for a plausible tool_result + Danish assistant followup.

    Only called for rows that actually have a call (skip clarify/refuse).
    Returns {"result": ..., "followup": "..."} or None on failure.
    """
    prompt = FOLLOWUP_GEN_PROMPT.format(
        catalog_json=format_catalog(catalog),
        user_utterance=user_utterance,
        call_json=json.dumps(call, ensure_ascii=False, indent=2),
    )
    resp = await call_gemini(prompt)
    if not resp:
        return None
    ex = _extract_json(resp)
    if not isinstance(ex, dict):
        return None
    if "result" not in ex or "followup" not in ex:
        return None
    if not isinstance(ex["followup"], str) or not ex["followup"].strip():
        return None
    return ex


def validate_example(ex: dict, target: dict, catalog: list[dict],
                     difficulty: str) -> tuple[bool, str]:
    """Validate one generated example against difficulty + schema rules."""
    call = ex.get("call")
    user = ex.get("user", "")
    reasoning = ex.get("reasoning", "")
    if not user or not reasoning:
        return False, "empty user/reasoning"

    if difficulty in ("clarify", "refuse"):
        if call is not None:
            return False, f"{difficulty} must have call=null"
        return True, ""

    if difficulty == "multi_chain":
        if not isinstance(call, list) or len(call) < 2:
            return False, "multi_chain must have call as list of >=2"
        for c in call:
            ok, err = validate_call(c, catalog)
            if not ok:
                return False, f"multi_chain sub-call: {err}"
        return True, ""

    # verbatim / inference / defaults — single call to target.
    if not isinstance(call, dict):
        return False, "call is not a dict"
    if call.get("navn") != target["navn"]:
        return False, f"call target mismatch: {call.get('navn')!r}"
    ok, err = validate_call(call, catalog)
    return ok, err


def build_sft_row(scenario: dict, tpl: dict, target_tool: dict,
                  catalog: list[dict], difficulty: str,
                  ex: dict, followup: dict | None,
                  attempts: int) -> dict:
    """Assemble final JSONL row.

    Emits RAW fields only. Any downstream SFT training format is produced
    by scripts/rerender_tool_calls.py — that's the single source of truth
    for the `messages` layout.

    `followup` may be None for clarify/refuse rows (no call → no result).
    """
    return {
        # --- raw pieces (source of truth) ---
        "tool_catalog": catalog,
        "target_tool": target_tool["navn"],
        "user_utterance": ex["user"],
        "assistant_reasoning": ex["reasoning"],
        "assistant_call": ex["call"],
        # None for clarify/refuse; dict/list for verbatim/inference/defaults/multi_chain.
        "tool_result": followup["result"] if followup else None,
        "assistant_followup": followup["followup"] if followup else None,
        # --- metadata ---
        "difficulty": difficulty,
        "scenario": scenario["id"],
        "structural_hash": template_hash(tpl),
        "n_tools_in_catalog": len(catalog),
        "attempts": attempts,
    }


# ────────────────────────────────────────────────────────────────────────────
# Tool cache — reuse invented tools as distractors
# ────────────────────────────────────────────────────────────────────────────

class ToolCache:
    """Growing pool of invented tools keyed by scenario for distractor sampling."""

    def __init__(self) -> None:
        self.by_scenario: dict[str, list[dict]] = {}
        self.all: list[dict] = []

    def add(self, scenario_id: str, tool: dict) -> None:
        self.by_scenario.setdefault(scenario_id, []).append(tool)
        self.all.append(tool)

    def sample_distractors(self, rng: random.Random, scenario_id: str,
                           n: int, exclude_names: set[str]) -> list[dict]:
        if n <= 0 or not self.all:
            return []
        # Prefer distractors from other scenarios (harder selection task).
        cross = [t for t in self.all
                 if t["navn"] not in exclude_names]
        rng.shuffle(cross)
        picked, seen = [], set(exclude_names)
        for t in cross:
            if t["navn"] in seen:
                continue
            picked.append(t)
            seen.add(t["navn"])
            if len(picked) >= n:
                break
        return picked


# ────────────────────────────────────────────────────────────────────────────
# Main async pipeline
# ────────────────────────────────────────────────────────────────────────────

async def worker(
    rng: random.Random,
    scenarios: list[dict],
    cache: ToolCache,
    difficulty_bucket: dict,   # from CLI: fixed weights
    max_tools_per_row: tuple[int, int],  # (min, max)
    max_attempts: int,
) -> dict | None:
    """Produce ONE validated SFT row (or None if all attempts failed)."""
    attempts = 0
    scenario = rng.choice(scenarios)
    difficulty = _weighted_choice(rng, difficulty_bucket)
    while attempts < max_attempts:
        attempts += 1
        tpl = sample_template(rng)
        target_tool = await invent_tool(scenario, tpl)
        if target_tool is None or not check_tool_schema_shape(target_tool, tpl):
            continue

        # Build catalog: target + 0..N distractors.
        n_extra = rng.randint(*max_tools_per_row) - 1
        distractors = cache.sample_distractors(
            rng, scenario["id"], n_extra, {target_tool["navn"]})
        catalog = [target_tool] + distractors
        rng.shuffle(catalog)

        # Generate the call example.
        ex = await generate_call_example(target_tool, catalog, difficulty)
        if ex is None:
            continue
        ok, err = validate_example(ex, target_tool, catalog, difficulty)
        if not ok:
            # Retry with a fresh template/scenario roll for this row.
            continue

        # For rows with an actual call, add a synthesized tool_result +
        # Danish assistant followup so agent-loop training is possible.
        # Skip clarify/refuse — no call means no result.
        followup = None
        if ex["call"] is not None:
            followup = await generate_followup(catalog, ex["user"], ex["call"])
            if followup is None:
                # Followup failed — retry the whole row.
                continue

        cache.add(scenario["id"], target_tool)
        return build_sft_row(scenario, tpl, target_tool, catalog,
                             difficulty, ex, followup, attempts)
    return None


async def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    scenarios = [json.loads(line) for line in
                 Path(args.seed_scenarios).read_text().splitlines()
                 if line.strip()]
    print(f"loaded {len(scenarios)} scenarios", flush=True)

    cache = ToolCache()
    tools_range = tuple(int(x) for x in args.tools_per_row.split(","))
    if len(tools_range) != 2:
        raise SystemExit("--tools-per-row must be 'min,max'")

    difficulty_bucket = dict(DIFFICULTY_WEIGHTS)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: if the output file already has rows, count them, rebuild the
    # distractor cache from their tool_catalog fields, and open in append
    # mode. Each row is independent — the "next row" only needs the growing
    # cache pool for distractor sampling, so we can resume without loss.
    resumed = 0
    if out_path.exists() and out_path.stat().st_size > 0:
        seen_names: set = set()
        with out_path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                resumed += 1
                for t in r.get("tool_catalog", []):
                    name = t.get("navn")
                    if name and name not in seen_names:
                        cache.add(r.get("scenario", "_resumed_"), t)
                        seen_names.add(name)
        print(f"resuming: {resumed} rows already in {out_path}, "
              f"cache pre-populated with {len(cache.all)} unique tools",
              flush=True)
        out_fh = out_path.open("a")
    else:
        out_fh = out_path.open("w")

    sem = asyncio.Semaphore(args.concurrency)
    stats = Counter()
    written = 0
    remaining = max(0, args.n_rows - resumed)
    if remaining == 0:
        print(f"target ({args.n_rows}) already met by resumed rows; nothing to do.",
              flush=True)
        out_fh.close()
        return

    async def produce_one():
        nonlocal written
        async with sem:
            row = await worker(rng, scenarios, cache, difficulty_bucket,
                               tools_range, args.max_attempts)
            if row is None:
                stats["failed"] += 1
                return
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_fh.flush()
            stats["ok"] += 1
            stats[f"diff:{row['difficulty']}"] += 1
            written += 1
            if written % 25 == 0:
                cost_str = ""
                if _OR_CALLS > 0:
                    cost_str = (f"  cost=${_OR_COST_USD:.4f}  "
                                f"per_row=${_OR_COST_USD/max(written,1):.5f}")
                print(f"  written={written}  failed={stats['failed']}  "
                      f"cache={len(cache.all)}{cost_str}  "
                      + " ".join(f"{k.split(':',1)[1]}={v}"
                                 for k, v in stats.items()
                                 if k.startswith("diff:")),
                      flush=True)

    tasks = [asyncio.create_task(produce_one()) for _ in range(remaining)]
    await asyncio.gather(*tasks)
    out_fh.close()

    # Clean up OpenRouter session if we opened one.
    if _OR_SESSION is not None:
        await _OR_SESSION.close()

    total = resumed + written
    print(f"\ndone. resumed={resumed}  new={written}  total={total}  "
          f"failed={stats['failed']}", flush=True)
    if _OR_CALLS > 0:
        print(f"OpenRouter cost (this session): ${_OR_COST_USD:.4f} over "
              f"{_OR_CALLS} calls ({_OR_INPUT_TOK} in / {_OR_OUTPUT_TOK} out "
              f"tokens)  per-new-row: ${_OR_COST_USD/max(written,1):.4f}",
              flush=True)
    print(f"per-difficulty: " + " ".join(f"{k.split(':',1)[1]}={v}"
                                          for k, v in stats.items()
                                          if k.startswith("diff:")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-scenarios",
                    default="data/tool_calls/scenarios_expanded.jsonl",
                    help="JSONL of {id, beskrivelse} rows. Default = "
                         "the 3786-scenario expanded pool.")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--n-rows", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--tools-per-row", default="1,5",
                    help="min,max total tools in catalog (target + distractors).")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="Per-row invention+generation retries on failure.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
