"""Build a GRPO JSON-schema training set for Danish, across 5 task types.

Task types:
    generate      — "give me a JSON for X with fields Y" (values open-ended)
    extract       — passage (Danish prose) + schema → JSON with grounded values
    rewrite       — bullet list / free text → JSON with same info restructured
    tool_call     — "call function_name with these args as JSON: user wants X"
    fill_template — "here is a JSON template with null values, fill from this info"

Each row's ground truth is the seed's `fields` list; the verifier only checks
JSON parseability + key-set match. For task_types with a `passage`, an optional
grounding check adds signal by requiring string values to appear as substrings
of the passage.

Row schema:
    task_type     str
    prompt        str        — user-facing Danish prompt (may include passage/template)
    fields        list[str]  — required top-level keys (ground truth)
    types         list[str]  — per-field type hints (informational)
    domain        str
    strict        bool       — if True, exact key-set match required
    passage       str|None   — source prose (extract/rewrite/fill_template)
    gold_values   dict|None  — {field: value} for extract; used for eval
    seed_idx      int
    variant       int

Usage:
    uv run python scripts/build_grpo_json_da.py \\
        --out data/grpo_json_da_smoke --n-variants 3 --concurrency 8 --limit-seeds 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path


MODEL = "google/gemini-2.5-flash-lite"


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
                     "HTTP-Referer": "https://claude-code-json",
                     "X-Title": "grpo-json-da"},
            timeout=aiohttp.ClientTimeout(total=90))
    return _SESSION


_TYPE_TO_JSON = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "bool": {"type": "boolean"},
    "list[str]": {"type": "array", "items": {"type": "string"}},
    "dict": {"type": "object"},
}


def _gold_schema(fields_types):
    """Force gemini to fill each required field with the right type."""
    props = {f: _TYPE_TO_JSON.get(t, {"type": "string"}) for f, t in fields_types}
    return {"type": "object", "properties": props,
            "required": [f for f, _ in fields_types],
            "additionalProperties": False}


def build_schema(task_type: str, fields_types):
    if task_type == "generate":
        return {"type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"], "additionalProperties": False}
    # extract / rewrite / fill_template — need passage + gold_values
    return {"type": "object",
            "properties": {"prompt": {"type": "string"},
                           "passage": {"type": "string"},
                           "gold_values": _gold_schema(fields_types)},
            "required": ["prompt", "passage", "gold_values"],
            "additionalProperties": False}


async def call_llm(sys_msg: str, user_msg: str, schema: dict, max_retries: int = 4):
    session = await _get_session()
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": sys_msg},
                     {"role": "user", "content": user_msg}],
        "temperature": 0.9,
        "max_tokens": 2000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "row", "strict": True, "schema": schema},
        },
    }
    backoff = 2.0
    for _ in range(max_retries):
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions", json=body
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(backoff); backoff *= 2; continue
                if resp.status != 200:
                    txt = await resp.text()
                    print(f"  http {resp.status}: {txt[:150]}", file=sys.stderr)
                    return None
                data = await resp.json()
                raw = data["choices"][0]["message"]["content"]
                return json.loads(raw)
        except Exception as e:
            print(f"  err {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            await asyncio.sleep(backoff); backoff *= 2
    return None


# ── seeds (domain + description + (field, type) tuples) ─────────────────────

SEEDS = [
    # People / social
    ("person",         "en dansk person",                 [("navn","str"), ("alder","int")]),
    ("person_full",    "en person med kontaktinfo",       [("fornavn","str"), ("efternavn","str"), ("email","str"), ("telefon","str")]),
    ("family_member",  "et familiemedlem",                [("navn","str"), ("relation","str")]),
    ("employee",       "en medarbejder",                  [("navn","str"), ("stilling","str"), ("løn","int"), ("ansat_dato","str")]),
    ("student",        "en studerende",                   [("navn","str"), ("studie","str"), ("semester","int"), ("gennemsnit","float")]),
    ("author",         "en bogforfatter",                 [("navn","str"), ("fødselsår","int"), ("nationalitet","str"), ("kendte_værker","list[str]")]),

    # Places
    ("city",           "en by",                           [("navn","str"), ("land","str")]),
    ("country_basic",  "et land",                         [("navn","str"), ("hovedstad","str"), ("befolkning","int")]),
    ("country_full",   "et land med detaljer",            [("navn","str"), ("hovedstad","str"), ("valuta","str"), ("sprog","list[str]"), ("areal_km2","int")]),
    ("address_da",     "en dansk adresse",                [("gade","str"), ("nummer","int"), ("postnummer","int"), ("by","str")]),
    ("landmark",       "et vartegn",                      [("navn","str"), ("by","str"), ("bygget_år","int")]),

    # Media
    ("book",           "en bog",                          [("titel","str"), ("forfatter","str"), ("år","int")]),
    ("book_full",      "en bog i biblioteket",            [("titel","str"), ("forfatter","str"), ("år","int"), ("genre","str"), ("sider","int"), ("isbn","str")]),
    ("movie",          "en film",                         [("titel","str"), ("år","int"), ("instruktør","str")]),
    ("song",           "en sang",                         [("titel","str"), ("kunstner","str"), ("album","str"), ("varighed_sek","int")]),
    ("podcast_ep",     "en podcast-episode",              [("titel","str"), ("show","str"), ("varighed_min","int"), ("udgivet","str")]),

    # Food
    ("recipe_min",     "en madopskrift",                  [("navn","str"), ("ingredienser","list[str]"), ("tid_min","int")]),
    ("recipe_full",    "en detaljeret madopskrift",       [("navn","str"), ("portioner","int"), ("ingredienser","list[str]"), ("trin","list[str]"), ("tid_min","int"), ("sværhedsgrad","str")]),
    ("drink",          "en drik",                         [("navn","str"), ("type","str"), ("alkohol_procent","float")]),
    ("fruit",          "en frugt",                        [("navn","str"), ("farve","str"), ("smag","str")]),

    # Nature
    ("animal",         "et dyr",                          [("art","str"), ("levested","str"), ("kost","str")]),
    ("bird_species",   "en fugleart",                     [("navn","str"), ("latinsk_navn","str"), ("vingefang_cm","int"), ("træktype","str")]),
    ("plant",          "en plante",                       [("navn","str"), ("familie","str"), ("højde_cm","int")]),

    # Weather
    ("weather_now",    "aktuelt vejr et sted",            [("by","str"), ("temperatur","float"), ("enhed","str"), ("vindhastighed","float"), ("beskrivelse","str")]),
    ("weather_forecast", "en vejrudsigt",                 [("dato","str"), ("min_temp","float"), ("max_temp","float"), ("regn_mm","float")]),
    ("coordinate",     "et geografisk koordinat",         [("breddegrad","float"), ("længdegrad","float")]),

    # Products / commerce
    ("product",        "et produkt i en webshop",         [("navn","str"), ("pris","float"), ("valuta","str"), ("på_lager","bool")]),
    ("product_full",   "et detaljeret produkt",           [("navn","str"), ("beskrivelse","str"), ("pris","float"), ("valuta","str"), ("kategori","str"), ("på_lager","bool"), ("antal","int")]),
    ("order",          "en ordre",                        [("ordre_id","str"), ("kunde","str"), ("total","float"), ("status","str")]),
    ("shopping_item",  "en vare på en indkøbsliste",      [("vare","str"), ("antal","int"), ("enhed","str")]),
    ("restaurant",     "en restaurant",                   [("navn","str"), ("by","str"), ("køkken","str"), ("prisniveau","int")]),

    # API / tech
    ("api_success",    "et vellykket API-svar",           [("status","str"), ("data","dict")]),
    ("api_error",      "et API-fejlsvar",                 [("fejl","bool"), ("besked","str"), ("kode","int")]),
    ("http_response",  "et HTTP-svar",                    [("kode","int"), ("headers","dict"), ("body","str")]),
    ("user_session",   "en brugersession",                [("session_id","str"), ("bruger","str"), ("udløber_kl","str")]),

    # Calendar
    ("calendar_event", "en kalenderbegivenhed",           [("titel","str"), ("start","str"), ("slut","str"), ("sted","str")]),
    ("meeting",        "et arbejdsmøde",                  [("emne","str"), ("deltagere","list[str]"), ("start","str"), ("varighed_min","int")]),
    ("birthday",       "en fødselsdag",                   [("person","str"), ("dato","str"), ("alder","int")]),

    # Q/A
    ("qa_pair",        "et spørgsmål og svar",            [("spørgsmål","str"), ("svar","str")]),
    ("mcq",            "et multiple-choice spørgsmål",    [("spørgsmål","str"), ("valg","list[str]"), ("korrekt","str")]),
    ("flashcard",      "et hjælpekort",                   [("forside","str"), ("bagside","str"), ("emne","str")]),

    # Status / config
    ("status_ok",      "en simpel status",                [("status","str")]),
    ("bool_flag",      "et boolsk flag",                  [("aktiv","bool")]),
    ("config_pair",    "en konfigurationsværdi",          [("nøgle","str"), ("værdi","str")]),

    # Numbers
    ("counter",        "en tæller",                       [("antal","int")]),
    ("measurement",    "en måling",                       [("værdi","float"), ("enhed","str"), ("tidspunkt","str")]),
    ("stats_summary",  "en statistisk opsummering",       [("min","float"), ("max","float"), ("middel","float"), ("std","float")]),

    # Lists / collections
    ("word_list",      "en liste af ord",                 [("ord","list[str]")]),
    ("number_list",    "en liste af tal",                 [("tal","list[str]")]),
    ("tagged_note",    "en tagget notat",                 [("titel","str"), ("indhold","str"), ("tags","list[str]")]),

    # Rating
    ("rating",         "en anmeldelse",                   [("titel","str"), ("bedømmelse","int"), ("kommentar","str")]),
    ("scoreboard",     "en resultatliste for et hold",    [("hold","str"), ("point","int"), ("kampe","int"), ("sejre","int"), ("nederlag","int")]),

    # Danish civics
    ("danish_king",    "en dansk konge",                  [("navn","str"), ("regerede_fra","int"), ("regerede_til","int"), ("hus","str")]),
    ("da_political_party", "et dansk politisk parti",     [("navn","str"), ("forkortelse","str"), ("grundlagt","int")]),
    ("subject_grade",  "en karakter i et skolefag",       [("fag","str"), ("karakter","int"), ("skala","str")]),

    # Health
    ("vital_signs",    "vitale målinger",                 [("puls","int"), ("blodtryk_sys","int"), ("blodtryk_dia","int"), ("temp_c","float")]),
    ("prescription",   "en recept",                       [("medicin","str"), ("dosis","str"), ("frekvens","str"), ("varighed_dage","int")]),

    # Travel
    ("flight",         "en flyafgang",                    [("selskab","str"), ("flynummer","str"), ("afgang","str"), ("ankomst","str"), ("fra","str"), ("til","str")]),
    ("train_departure", "en toafgang",                    [("linje","str"), ("fra","str"), ("til","str"), ("afgang_kl","str"), ("spor","int")]),
    ("hotel_booking",  "en hotelreservation",             [("hotel","str"), ("by","str"), ("check_ind","str"), ("check_ud","str"), ("gæster","int")]),

    # Games
    ("board_game",     "et brætspil",                     [("navn","str"), ("spillere_min","int"), ("spillere_max","int"), ("varighed_min","int")]),
    ("chess_game",     "et skakparti",                    [("hvid","str"), ("sort","str"), ("resultat","str"), ("åbning","str")]),

    # Money
    ("transaction",    "en banktransaktion",              [("dato","str"), ("beløb","float"), ("valuta","str"), ("modtager","str")]),
    ("invoice",        "en faktura",                      [("faktura_nr","str"), ("beløb","float"), ("moms","float"), ("forfaldsdato","str")]),

    # Support
    ("support_ticket", "en supportbillet",                [("id","str"), ("emne","str"), ("prioritet","str"), ("status","str"), ("opretter","str")]),
    ("bug_report",     "en fejlrapport",                  [("titel","str"), ("beskrivelse","str"), ("sværhedsgrad","str"), ("modul","str")]),

    # Programming / tech
    ("git_commit",     "en Git-commit",                   [("hash","str"), ("forfatter","str"), ("besked","str"), ("dato","str")]),
    ("git_pr",         "en pull request",                 [("titel","str"), ("forfatter","str"), ("branch","str"), ("kommentarer","int"), ("status","str")]),
    ("github_issue",   "et GitHub-issue",                 [("titel","str"), ("nummer","int"), ("labels","list[str]"), ("åben","bool")]),
    ("npm_package",    "et npm-pakke",                    [("navn","str"), ("version","str"), ("licens","str"), ("beskrivelse","str")]),
    ("docker_container", "en Docker-container",           [("navn","str"), ("image","str"), ("status","str"), ("port","int")]),
    ("env_var",        "en miljøvariabel",                [("navn","str"), ("værdi","str"), ("beskyttet","bool")]),
    ("code_snippet",   "en kodesnippet",                  [("sprog","str"), ("beskrivelse","str"), ("linjer","int")]),

    # Sports
    ("match_result",   "et kampresultat",                 [("hjemmehold","str"), ("udehold","str"), ("hjemme_score","int"), ("ude_score","int"), ("dato","str")]),
    ("football_player", "en fodboldspiller",              [("navn","str"), ("klub","str"), ("position","str"), ("nationalitet","str"), ("mål","int")]),
    ("tournament",     "en turnering",                    [("navn","str"), ("sport","str"), ("år","int"), ("vinder","str")]),
    ("player_stats",   "en spillers sæsonstatistik",      [("navn","str"), ("sæson","str"), ("kampe","int"), ("mål","int"), ("assists","int")]),

    # Music / entertainment
    ("album",          "et musikalbum",                   [("titel","str"), ("kunstner","str"), ("år","int"), ("genre","str"), ("antal_spor","int")]),
    ("concert",        "en koncert",                      [("kunstner","str"), ("sted","str"), ("dato","str"), ("billetpris","float")]),
    ("playlist",       "en spilleliste",                  [("navn","str"), ("ejer","str"), ("antal_sange","int"), ("varighed_min","int")]),
    ("tv_show",        "en tv-serie",                     [("titel","str"), ("kanal","str"), ("startår","int"), ("antal_sæsoner","int"), ("genre","str")]),
    ("video_game",     "et videospil",                    [("titel","str"), ("udvikler","str"), ("år","int"), ("platform","str"), ("genre","str")]),
    ("character",      "en fiktiv karakter",              [("navn","str"), ("værk","str"), ("skaber","str"), ("art","str")]),

    # Nature (extended)
    ("mineral",        "et mineral",                      [("navn","str"), ("formel","str"), ("hårdhed","float"), ("farve","str")]),
    ("mountain",       "et bjerg",                        [("navn","str"), ("højde_m","int"), ("bjergkæde","str"), ("land","str")]),
    ("river",          "en flod",                         [("navn","str"), ("længde_km","int"), ("kilde","str"), ("udmunding","str")]),
    ("lake",           "en sø",                           [("navn","str"), ("areal_km2","float"), ("dybde_m","int"), ("land","str")]),
    ("biome",          "et biom",                         [("navn","str"), ("gennemsnits_temp_c","float"), ("nedbør_mm","int"), ("typiske_arter","list[str]")]),

    # Medical
    ("symptom",        "et sygdomssymptom",               [("navn","str"), ("varighed_dage","int"), ("sværhedsgrad","str"), ("relateret_til","str")]),
    ("diagnosis",      "en diagnose",                     [("kode","str"), ("beskrivelse","str"), ("system","str"), ("behandling","str")]),
    ("appointment",    "en lægeaftale",                   [("læge","str"), ("dato","str"), ("tid","str"), ("årsag","str")]),
    ("allergy",        "en allergi",                      [("stof","str"), ("reaktion","str"), ("sværhedsgrad","str")]),

    # Legal
    ("court_case",     "en retssag",                      [("sagsnr","str"), ("retten","str"), ("sagsøger","str"), ("sagsøgte","str"), ("emne","str")]),
    ("contract",       "en kontrakt",                     [("parter","list[str]"), ("dato","str"), ("beløb","float"), ("valuta","str"), ("emne","str")]),

    # Finance
    ("stock",          "en aktie",                        [("ticker","str"), ("selskab","str"), ("kurs","float"), ("valuta","str")]),
    ("portfolio",      "en investeringsportefølje",       [("ejer","str"), ("total_værdi","float"), ("valuta","str"), ("antal_aktier","int")]),
    ("expense",        "en udgift",                       [("kategori","str"), ("beløb","float"), ("dato","str"), ("beskrivelse","str")]),
    ("budget_line",    "en budgetlinje",                  [("kategori","str"), ("budget","float"), ("brugt","float"), ("resterende","float")]),

    # Fitness / health
    ("workout",        "en træningssession",              [("aktivitet","str"), ("varighed_min","int"), ("kalorier","int"), ("puls_gns","int")]),
    ("exercise_set",   "et træningssæt",                  [("øvelse","str"), ("antal_sæt","int"), ("antal_reps","int"), ("vægt_kg","float")]),
    ("macro_target",   "et makro-mål",                    [("protein_g","int"), ("kulhydrat_g","int"), ("fedt_g","int"), ("kalorier","int")]),

    # Real estate
    ("property",       "en ejendom",                      [("adresse","str"), ("boligtype","str"), ("kvm","int"), ("værelser","int"), ("pris","float")]),
    ("rental_listing", "en lejebolig-annonce",            [("adresse","str"), ("husleje","float"), ("kvm","int"), ("værelser","int"), ("ledig_fra","str")]),

    # Fashion / retail
    ("garment",        "et stykke tøj",                   [("type","str"), ("mærke","str"), ("størrelse","str"), ("farve","str"), ("pris","float")]),
    ("outfit",         "et outfit",                       [("lejlighed","str"), ("stykker","list[str]"), ("stil","str")]),

    # Vehicle
    ("car",            "en bil",                          [("mærke","str"), ("model","str"), ("årgang","int"), ("brændstof","str"), ("hestekræfter","int")]),
    ("bike",           "en cykel",                        [("mærke","str"), ("type","str"), ("gear","int"), ("farve","str")]),
    ("boat",           "en båd",                          [("navn","str"), ("type","str"), ("længde_m","float"), ("hjemmehavn","str")]),

    # Education
    ("course",         "et kursus",                       [("titel","str"), ("underviser","str"), ("ects","int"), ("periode","str")]),
    ("exam",           "en eksamen",                      [("fag","str"), ("form","str"), ("dato","str"), ("varighed_min","int")]),
    ("essay",          "en essayopgave",                  [("titel","str"), ("emne","str"), ("ord_antal","int"), ("afleveringsdato","str")]),

    # Communication
    ("email_summary",  "et email-resumé",                 [("afsender","str"), ("modtager","str"), ("emne","str"), ("dato","str")]),
    ("meeting_note",   "et møde-notat",                   [("emne","str"), ("dato","str"), ("beslutninger","list[str]"), ("næste_skridt","list[str]")]),
    ("sms",            "en SMS-besked",                   [("afsender","str"), ("modtager","str"), ("besked","str"), ("tidspunkt","str")]),

    # History
    ("battle",         "et historisk slag",               [("navn","str"), ("år","int"), ("parter","list[str]"), ("vinder","str")]),
    ("dynasty",        "en dynasti",                      [("navn","str"), ("startår","int"), ("slutår","int"), ("land","str")]),
    ("treaty",         "en traktat",                      [("navn","str"), ("år","int"), ("parter","list[str]"), ("emne","str")]),

    # Science
    ("chemical_element", "et grundstof",                  [("navn","str"), ("symbol","str"), ("atomnummer","int"), ("atommasse","float")]),
    ("physical_constant", "en fysisk konstant",           [("navn","str"), ("symbol","str"), ("værdi","float"), ("enhed","str")]),
    ("scientific_paper", "en videnskabelig artikel",      [("titel","str"), ("forfatter","str"), ("tidsskrift","str"), ("år","int"), ("doi","str")]),

    # Danish civics (extended)
    ("da_kommune",     "en dansk kommune",                [("navn","str"), ("region","str"), ("borgmester","str"), ("indbyggertal","int")]),
    ("da_region",      "en dansk region",                 [("navn","str"), ("hovedby","str"), ("kommuner","int"), ("indbyggertal","int")]),
    ("folketing_seat", "en folketingsplads",              [("parti","str"), ("navn","str"), ("kreds","str")]),
    ("da_holiday",     "en dansk helligdag",              [("navn","str"), ("dato","str"), ("religiøs","bool")]),

    # Deep nested
    ("company_deep",   "et firma med afdelinger",         [("navn","str"), ("stiftet","int"), ("afdelinger","dict"), ("hovedkontor","dict")]),
    ("family_tree",    "en familiestamtavle",             [("root_navn","str"), ("børn","list[str]"), ("stamfader","dict")]),
    ("recipe_deep",    "opskrift med sektioner",          [("navn","str"), ("sektioner","dict"), ("total_tid_min","int")]),
    ("json_config",    "en indlejret konfiguration",      [("app","str"), ("indstillinger","dict"), ("debug","bool")]),

    # Simple 1-field variants (calibration)
    ("just_name",      "en person med kun et navn",       [("navn","str")]),
    ("just_flag",      "en boolsk værdi",                 [("aktiv","bool")]),
    ("just_price",     "en pris",                         [("pris","float")]),
    ("just_id",        "en identifikator",                [("id","str")]),

    # Longer / high-arity (7-9 fields, stress cases)
    ("company_full",   "et fuldt firmaprofil",            [("navn","str"), ("cvr","str"), ("branche","str"), ("stiftet","int"), ("antal_ansatte","int"), ("omsætning","float"), ("valuta","str"), ("hjemmeside","str")]),
    ("scientific_paper_full", "en videnskabelig artikel (fuld)", [("titel","str"), ("forfattere","list[str]"), ("tidsskrift","str"), ("år","int"), ("bind","int"), ("side_fra","int"), ("side_til","int"), ("doi","str")]),
    ("job_posting",    "en jobopslag",                    [("titel","str"), ("virksomhed","str"), ("by","str"), ("kontrakttype","str"), ("løn_fra","int"), ("løn_til","int"), ("ansøgningsfrist","str")]),
]


# Task-type mix targets (rows will roughly follow this ratio; task_type is
# sampled per variant, not per seed, so each seed sees multiple types)
# tool_call removed — needs its own verb-seed catalogue + intent-not-args
# framing to be useful; current generic-seed reuse produced too many inlined-
# value or invent-from-nothing rows. Revisit as a separate pipeline.
TASK_MIX = {
    "extract": 0.45,
    "generate": 0.25,
    "rewrite": 0.20,
    "fill_template": 0.10,
}
TASK_TYPES = list(TASK_MIX.keys())
TASK_WEIGHTS = list(TASK_MIX.values())


# ── system prompts per task type ────────────────────────────────────────────

SYS_GENERATE = """Du er en assistent, der laver træningsprompter til en dansk sprogmodel.
Din opgave: Modtag en JSON-schema-specifikation (domæne, beskrivelse, feltnavne+typer)
og skriv EN dansk brugerprompt, der beder om et JSON-objekt med præcis de felter.

KRAV:
- Prompten skal være på DANSK.
- Feltnavnene skal nævnes ordret (samme casing, samme accenter).
- Sproget skal variere: nogle er formelle, andre uformelle, tekniske eller kortfattede.
- Nogle gange (~50%) skal du tilføje "kun disse felter" eller "og ingen andre felter".
- Nogle gange (~30%) skal du tilføje "svar med kun JSON" eller "returnér udelukkende JSON".
- Prompten må IKKE indeholde et eksempelsvar eller det færdige JSON.
- Prompten SKAL indeholde alle feltnavnene præcis én gang.
- Længden: 1-4 sætninger. Ikke over 350 tegn.

Svar UDELUKKENDE med: {"prompt": "..."}."""


SYS_EXTRACT = """Du er en assistent, der laver træningsdata til en dansk sprogmodel.
Din opgave: Givet en JSON-schema-specifikation (domæne, beskrivelse, feltnavne+typer),
skab (a) en dansk PASSAGE (2-5 sætninger prosa) der indeholder alle information nødvendig
for at udfylde skemaet, og (b) en dansk BRUGERPROMPT der beder modellen om at udtrække
et JSON-objekt med præcis de felter fra passagen.

KRAV:
- Passage OG prompt skal være på DANSK.
- Passagen skal være naturlig prosa (ikke en liste), 2-5 sætninger.
- Passagen skal indeholde ALLE de informationer, som skemaet forlanger — plausible værdier.
- Prompten skal nævne alle feltnavne ordret én gang, og bede om JSON-udtrækning.
- Prompten skal referere til passagen eller inkludere den (f.eks. "Baseret på følgende tekst:").
- Nogle gange (~40%) bed om "kun JSON" eller "svar udelukkende med JSON".
- Ingen eksempelsvar eller færdigt JSON i prompten.

Svar UDELUKKENDE med: {"prompt": "...", "passage": "...", "gold_values": {...}}
hvor gold_values er et objekt med præcis de påkrævede felter og deres værdier fra passagen.
For string-felter skal værdien være en DIREKTE ordret substring af passagen."""


SYS_REWRITE = """Du er en assistent, der laver træningsdata til en dansk sprogmodel.
Din opgave: Givet en JSON-schema-specifikation (domæne, beskrivelse, feltnavne+typer),
skab (a) en dansk KILDETEKST i punktform eller stikord (IKKE prosa) med alle nødvendige
informationer, og (b) en dansk BRUGERPROMPT der beder modellen om at OMFORME kilden til
et JSON-objekt med præcis de påkrævede felter.

KRAV:
- Kilde OG prompt skal være på DANSK.
- Kilden er punktopstilling ("- ..."), semikolon-liste, eller markdown-list — IKKE prosa.
- Kilden skal indeholde alle informationer skemaet forlanger.
- Prompten skal nævne alle feltnavne ordret én gang, og bede om omformning til JSON.
- Prompten skal referere til eller inkludere kildeteksten.
- Nogle gange (~40%) bed om "kun JSON".

Svar UDELUKKENDE med: {"prompt": "...", "passage": "...", "gold_values": {...}}
hvor passage er kildeteksten og gold_values er de rigtige feltværdier."""


SYS_FILL_TEMPLATE = """Du er en assistent, der laver træningsdata til en dansk sprogmodel.
Din opgave: Givet en JSON-schema-specifikation (domæne, beskrivelse, feltnavne+typer),
skab (a) en dansk PASSAGE (2-4 sætninger med information), og (b) en dansk BRUGERPROMPT
der giver et JSON-TEMPLATE med null-værdier og beder modellen om at udfylde det ved
hjælp af informationen i passagen.

KRAV:
- Passage og prompt skal være på DANSK.
- Passagen skal indeholde alle nødvendige oplysninger.
- Prompten skal indeholde et JSON-template som fx: {"navn": null, "alder": null}
- Prompten skal bede modellen udfylde template med værdier fra passagen.
- Alle feltnavne nævnes én gang (i template).

Svar UDELUKKENDE med: {"prompt": "...", "passage": "...", "gold_values": {...}}
hvor gold_values er de rigtige udfyldte værdier."""


SYS_BY_TYPE = {
    "generate": SYS_GENERATE,
    "extract": SYS_EXTRACT,
    "rewrite": SYS_REWRITE,
    "fill_template": SYS_FILL_TEMPLATE,
}


def build_user_msg(seed, task_type: str) -> str:
    domain, desc, fields_types = seed
    fields_list = "\n".join(f"- {n} ({t})" for n, t in fields_types)
    return (
        f"Domæne: {domain}\n"
        f"Beskrivelse: {desc}\n"
        f"Task-type: {task_type}\n"
        f"Felter og typer:\n{fields_list}"
    )


# ── main loop ────────────────────────────────────────────────────────────────

_REJECT_COUNTS = {"llm_err": 0, "empty_prompt": 0, "missing_field_in_prompt": 0,
                  "no_gold_dict": 0, "gold_missing_field": 0, "grounding_fail": 0}


def _grounds(gold_val, passage: str) -> bool:
    """Loose grounding: substring match after lowercasing and collapsing spaces."""
    if not isinstance(gold_val, str) or not passage:
        return True  # non-str values not checked
    def norm(s):
        return " ".join(s.lower().replace("\n", " ").split())
    return norm(gold_val) in norm(passage)


async def process_seed(idx: int, seed, rng: random.Random, n_variants: int, sem, debug: bool = False):
    domain, desc, fields_types = seed
    fields = [f for f, _ in fields_types]
    types = [t for _, t in fields_types]
    rows = []

    for v in range(n_variants):
        task_type = rng.choices(TASK_TYPES, weights=TASK_WEIGHTS, k=1)[0]
        async with sem:
            resp = await call_llm(
                SYS_BY_TYPE[task_type],
                build_user_msg(seed, task_type),
                build_schema(task_type, fields_types),
            )
        if resp is None or "prompt" not in resp:
            _REJECT_COUNTS["llm_err"] += 1
            if debug: print(f"    [drop:{task_type}] llm_err", file=sys.stderr)
            continue
        prompt = (resp.get("prompt") or "").strip()
        if not prompt:
            _REJECT_COUNTS["empty_prompt"] += 1
            if debug: print(f"    [drop:{task_type}] empty_prompt", file=sys.stderr)
            continue
        passage = resp.get("passage")
        gold = resp.get("gold_values")

        missing = [f for f in fields if f not in prompt]
        if missing:
            _REJECT_COUNTS["missing_field_in_prompt"] += 1
            if debug: print(f"    [drop:{task_type}] prompt missing fields {missing}", file=sys.stderr)
            continue

        if task_type in {"extract", "rewrite", "fill_template"}:
            if not isinstance(gold, dict):
                _REJECT_COUNTS["no_gold_dict"] += 1
                if debug:
                    print(f"    [drop:{task_type}] no gold dict; resp keys={list(resp.keys())} "
                          f"gold_type={type(resp.get('gold_values')).__name__}", file=sys.stderr)
                continue
            gold_missing = [f for f in fields if f not in gold]
            if gold_missing:
                _REJECT_COUNTS["gold_missing_field"] += 1
                if debug: print(f"    [drop:{task_type}] gold missing {gold_missing}", file=sys.stderr)
                continue
            if task_type in {"extract", "rewrite"} and passage:
                bad = [f for (f, t) in fields_types
                       if t == "str" and not _grounds(gold.get(f), passage)]
                if bad:
                    _REJECT_COUNTS["grounding_fail"] += 1
                    if debug: print(f"    [drop:{task_type}] grounding_fail {bad} "
                                    f"gold={gold} pass={passage[:100]!r}", file=sys.stderr)
                    continue

        rows.append({
            "task_type": task_type,
            "prompt": prompt,
            "fields": fields,
            "types": types,
            "domain": domain,
            "strict": rng.random() < 0.4,
            "passage": passage,
            "gold_values": gold,
            "seed_idx": idx,
            "variant": v,
        })
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-variants", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-seeds", type=int, default=None)
    ap.add_argument("--print-samples", type=int, default=8)
    ap.add_argument("--debug-drops", action="store_true")
    args = ap.parse_args()

    seeds = SEEDS
    if args.limit_seeds:
        seeds = seeds[: args.limit_seeds]

    print(f"[cfg] seeds={len(seeds)}  variants/seed={args.n_variants}  "
          f"concurrency={args.concurrency}  target rows={len(seeds)*args.n_variants}\n"
          f"      task mix: {TASK_MIX}", flush=True)

    rng = random.Random(args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    tasks = [process_seed(i, s, rng, args.n_variants, sem, debug=args.debug_drops)
             for i, s in enumerate(seeds)]
    all_rows = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        rows = await coro
        all_rows.extend(rows)
        done += 1
        if done % 10 == 0 or done == len(tasks):
            print(f"  [{done}/{len(tasks)} seeds] rows so far: {len(all_rows)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    global _SESSION
    if _SESSION:
        await _SESSION.close()
        _SESSION = None

    Path(args.out).mkdir(parents=True, exist_ok=True)
    # Write jsonl FIRST so a schema conflict downstream doesn't nuke the whole run.
    with open(Path(args.out) / "rows.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # For Arrow storage, serialize gold_values to a JSON string — the dict
    # shape varies per row (different key sets) so a struct schema won't fit.
    from datasets import Dataset
    arrow_rows = [
        {**r, "gold_values": json.dumps(r["gold_values"], ensure_ascii=False) if r["gold_values"] is not None else None}
        for r in all_rows
    ]
    ds = Dataset.from_list(arrow_rows)
    ds.save_to_disk(args.out)

    # per-task-type count
    from collections import Counter
    counts = Counter(r["task_type"] for r in all_rows)
    print(f"\n[done] {len(all_rows)} rows in {time.time()-t0:.0f}s → {args.out}", flush=True)
    print(f"       counts: {dict(counts)}", flush=True)
    print(f"       drops:  {_REJECT_COUNTS}", flush=True)

    if args.print_samples:
        print(f"\n--- {args.print_samples} sample rows ---")
        for r in rng.sample(all_rows, min(args.print_samples, len(all_rows))):
            print(f"\n[{r['task_type']}  domain={r['domain']}  strict={r['strict']}  "
                  f"fields={r['fields']}]")
            print(f"  prompt: {r['prompt']}")
            if r["passage"]:
                print(f"  passage: {r['passage']}")
            if r["gold_values"]:
                print(f"  gold:    {r['gold_values']}")


if __name__ == "__main__":
    asyncio.run(main())
