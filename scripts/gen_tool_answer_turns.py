"""Answer the dangling terminal call in every tool dialogue.

The source corpus always cuts at the last call, so 17,123 of 25,411 calls
(67.4%) have no result behind them and no answer after them. The consequence is
a lopsided corpus: reasoning-before-a-call is 78.7% of the trained tokens and
grounded answers are 3.5%. Both live in the `<|assistant|>` slot, so after a
`<|tool_result|>` the model picks between two learned registers at ~22:1 odds
and reliably picks the wrong one -- on four unseen tools it fabricated
`4 * 12 = 48` while the tool had just returned `dog_years: 28`.

This fills in the missing half-turn: for each dangling call, invent a result
that conforms to the tool's `returns` schema, then write the Danish answer that
reads it. Existing turns are never touched.

Cache is keyed on (tool, arguments, question, fingerprint of the tool's
`returns` block) and appended as it goes, so a rerun costs only what is new.
The fingerprint is in the key because the schema is an INPUT to the payload:
edit `returns` -- nest it, dedupe its synonym fields -- and every cached result
was built to a contract that no longer holds.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"

# Meta-talk. The failure this pass exists to fix is the model narrating the
# call instead of answering ("Så det korrekte værktøjskald er..."), so an
# answer that does the same thing is worse than no answer at all.
META = re.compile(r"\b(værktøjskald|værktøjet|funktionskald|tool_call|"
                  r"json|parameter|parametre|argument(er)?|api|"
                  r"jeg (skal|vil) (nu |bare )?(kalde|formatere|generere))\b",
                  re.I)
# No sign. `2024-03-15` is a date, not two negative numbers, and reading it as
# one made every reformatted date ("15. marts 2024") look invented.
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in (Path.home() / "or", Path.home() / ".or"):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("no OpenRouter key (~/or)")


ANSWER_SYS = """Du laver træningsdata til en dansk assistent, der bruger værktøjer.

Du får et værktøj (navn, beskrivelse, parametre og hvilke felter det
returnerer), brugerens spørgsmål, og det kald assistenten har lavet.

Lav to ting:

1. "resultat": et realistisk svar fra værktøjet, som JSON-objekt. Brug PRÆCIS
   de felter værktøjet siger det returnerer -- ingen ekstra felter, ingen
   udeladte. Værdierne skal passe til kaldets argumenter og være konkrete
   (rigtige tal, rigtige navne), ikke pladsholdere som "abc" eller 0.

2. "relevante_felter": navnene på de felter i "resultat", som brugerens
   spørgsmål faktisk beder om. Som regel ét felt, sjældent mere end to. De
   øvrige felter er baggrund og hører IKKE i svaret.

3. "svar": assistentens svar til brugeren, på dansk, 1-2 sætninger.

Krav til "svar":
- Det skal indeholde værdierne fra "relevante_felter" -- og KUN dem.
- Nævn ikke de andre felters værdier. Et svar der remser hele resultatet op,
  er et dårligt svar, også selvom alt i det er sandt.
- Det skal svare på brugerens spørgsmål, ikke beskrive hvad du har gjort.
- Skriv ALDRIG om værktøjer, kald, funktioner, parametre eller JSON.
- Ingen engelske ord eller sætninger.
- Regn ikke videre på tallene og find ikke på tal, der ikke står i "resultat".

Spørgsmål: "Er der kaffe tilbage på 4. etage?"
resultat  {"cups_left": 8, "days_since_service": 12, "working": true}
  relevante_felter  ["cups_left"]
  godt svar   "Der er 8 kopper kaffe tilbage på 4. etage."
  dårligt svar "Der er 8 kopper tilbage, maskinen virker, og den blev
                serviceret for 12 dage siden."   <- remser resultatet op
"""


def _nums(text):
    """Numbers appearing in a piece of text, as floats."""
    out = set()
    for m in NUM.finditer(str(text)):
        try:
            out.add(float(m.group().replace(",", ".")))
        except ValueError:
            continue
    return out


DA_THOUSANDS = re.compile(r"^\d{1,3}(?:\.\d{3})+$")


def _readings(tok):
    """Every number a Danish token could mean, with its decimal precision.

    Danish writes thousands with a period and decimals with a comma, so
    `76.500` is seventy-six thousand five hundred -- reading it as 76.5 marked
    correct answers as fabrications.
    """
    out = []
    if DA_THOUSANDS.match(tok):
        out.append((float(tok.replace(".", "")), 0))
    t = tok.replace(",", ".")
    try:
        out.append((float(t), len(t.split(".")[1]) if "." in t else 0))
    except ValueError:
        pass
    return out


def _traces_to(tok, pool):
    """Is `tok` a faithful rendering of some number in `pool`?

    Answers restate values rather than copy them: 153.9380400259 is written
    "153.94", 0.98 is written "98%", 76500 is written "76.500". All three are
    correct readings of the payload; a literal comparison calls them
    fabrications, which is what rejected 45 clean rows.
    """
    for val, d in _readings(tok):
        for p in pool:
            if val == p or round(p, d) == val:
                return True
            if d == 0 and int(val) == int(p):
                return True
            if round(p * 100, d) == val:      # 0.98 -> 98%
                return True
    return False


def _coerce(obj):
    """Numeric strings back to numbers.

    Leaves in a v4 `returns` block carry a Danish description and no type, so
    the schema cannot state one and the provider returns every value as a
    string: `{"area": "50.26548245743669"}`. That is both unnatural as a tool
    result and invisible to the grounding check, which reads numbers.
    """
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce(v) for v in obj]
    if isinstance(obj, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", obj.strip()):
        f = float(obj)
        return int(f) if f.is_integer() and "." not in obj else f
    return obj


def _opaque(s: str) -> bool:
    """A machine token with no quotable content: base64, id, hash, URL.

    Whitespace is the discriminator -- real prose has some. A long unbroken
    run of letters+digits, or anything URL-shaped, carries no fact a Danish
    answer could cite.
    """
    s = s.strip()
    if not s or " " in s:
        return False
    if s.lower().startswith(("http://", "https://", "www.", "data:")):
        return True
    if len(s) > 40:
        return True
    return (len(s) >= 8 and any(c.isdigit() for c in s)
            and any(c.isalpha() for c in s))


DA_MONTHS = ["januar", "februar", "marts", "april", "maj", "juni", "juli",
             "august", "september", "oktober", "november", "december"]
ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
# Danish spells a leading 1 as an article: "1 moden banan" -> "en moden banan".
DA_ONE = re.compile(r"^1(?=\s)")


def _date_forms(s):
    """Danish renderings of an ISO date.

    A tool returns `2024-03-15`; a Danish answer writes "den 15. marts". The
    ISO string is never a substring of a correctly worded answer, so the
    relevance check read every date field as uncited.
    """
    m = ISO_DATE.match(str(s).strip())
    if not m:
        return []
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not 1 <= mo <= 12:
        return []
    mon = DA_MONTHS[mo - 1]
    return [f"{d}. {mon} {y}", f"{d}. {mon}", f"{d}/{mo} {y}", f"{d}/{mo}-{y}",
            f"{d}/{mo}", f"{d:02d}-{mo:02d}-{y}", f"{d:02d}/{mo:02d}/{y}"]


def _cited(answer, v):
    """Is this payload value present in the answer, Danish formatting allowed?

    Whole-value substring is the honest test for a scalar, but three payload
    shapes defeat it even when the answer is perfect: ISO dates get written
    out in Danish, delimited fields get read with a real connective ("A, B og
    C"), and long prose gets quoted by its opening. The grounding check above
    already tolerates the last two via `_cands`; the relevance check called
    this and so rejected 22 of 26 ToolACE answers that cite everything asked.
    """
    low = answer.lower()
    if isinstance(v, (int, float)):
        return any(_traces_to(m.group(), {float(v)}) for m in NUM.finditer(low))
    s = str(v).strip()
    if len(s) <= 2:
        return False
    if s.lower() in low:
        return True
    if any(f.lower() in low for f in _date_forms(s)):
        return True
    if DA_ONE.match(s):
        rest = s[1:].strip().lower()
        if rest and any(f"{a} {rest}" in low for a in ("en", "et", "én")):
            return True
    # A delimited field is cited when every part of it is -- the answer is free
    # to rejoin them with "og" instead of a comma.
    parts = [p.strip() for p in re.split(r"[,;|\n]+", s)]
    parts = [p for p in parts if len(p) > 2]
    if len(parts) > 1 and all(_cited(answer, p) for p in parts):
        return True
    words = s.split()
    return len(words) > 6 and " ".join(words[:6]).lower() in low


def _json_in_string(v):
    """A payload field holding serialised JSON, parsed.

    ToolACE returns `{"jobs": "[{\\"by\\": \\"mats\\", ...}]"` -- a list of
    records delivered as one string. Left as a string it is a single leaf that
    no answer can quote whole.
    """
    if isinstance(v, str) and v.strip()[:1] in "[{":
        try:
            p = json.loads(v)
        except (ValueError, TypeError):
            return None
        if isinstance(p, (dict, list)):
            return p
    return None


def _citable(obj):
    """Leaves of `obj` an answer could quote."""
    return [x for _, x in _leaves(obj)
            if not isinstance(x, bool) and x not in (None, "")
            and not (isinstance(x, str) and _opaque(x))]


def _quorum(n):
    """How many of `n` parts an answer must quote: a strict majority, cap 3.

    A plain majority is wrong at both ends. At n=2 it rounds to 1, which let
    "Ja, jeg fandt to opskrifter" through on a two-record payload -- the thin
    answer this check exists to reject. At n=50 it would demand 26 records be
    named, when quoting three plainly demonstrates the field was read.
    """
    return min(3, n // 2 + 1)


def _cited_beyond(answer, x, ctx):
    """Cited, and not merely an echo of the question.

    Records match on ANY of their fields, which is weak when the field is a
    tag the user already said: "to italienske opskrifter" matched two recipes
    on `italiensk` alone and passed an answer that names neither. A value the
    question already contains is no evidence the payload was read. Applied to
    containers only -- a scalar that happens to appear in the question is
    still the thing being asked for.
    """
    if not _cited(answer, x):
        return False
    s = str(x).strip().lower()
    return bool(s) and s not in ctx.lower()


def _element_cited(answer, el, ctx=""):
    """One element of a list field: a record counts if any of its fields is quoted."""
    if isinstance(el, (dict, list)):
        return any(_cited_beyond(answer, x, ctx) for x in _citable(el))
    return _cited_beyond(answer, el, ctx)


def _relevant_cited(answer, v, ctx=""):
    """Did the answer use this relevant field?

    A scalar must survive intact; a CONTAINER may not. A tool returning 50
    jobs is answered by naming a few, and a schedule of three dates by listing
    the dates and not the wrapper -- demanding every leaf marks those
    summaries as ungrounded. So a list needs a majority of its ELEMENTS
    represented (one quoted field per record is how records get cited) and an
    object a majority of its leaves. Scalars keep the old all-or-nothing rule,
    which is what the planted `answer-misses-relevant` control tests.
    """
    inner = _json_in_string(v)
    if inner is not None:
        v = inner
    if isinstance(v, list):
        els = [e for e in v if e not in (None, "")]
        if not els:
            return True
        hit = sum(1 for e in els if _element_cited(answer, e, ctx))
        return hit >= _quorum(len(els))
    if isinstance(v, dict):
        leaves = _citable(v)
        if not leaves:
            return True
        return sum(1 for x in leaves
                   if _cited_beyond(answer, x, ctx)) >= _quorum(len(leaves))
    return _cited(answer, v)


def _leaves(obj, prefix=""):
    """Scalar leaves of a result payload, as (path, value).

    A field holding serialised JSON is expanded rather than yielded whole:
    ToolACE delivers lists of records as one string, and treating that as a
    single leaf hid every value inside it from the grounding check as well as
    the relevance check.
    """
    inner = _json_in_string(obj)
    if inner is not None:
        obj = inner
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def _arrays(obj):
    """Every list inside a payload, so its length can be cited."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _arrays(v)
    elif isinstance(obj, list):
        yield obj
        for v in obj:
            yield from _arrays(v)


def _spec_fields(returns):
    """Top-level field names a `returns` block declares.

    Normalised to the first path segment because v4 stores the tree flattened
    (`restaurants[].address`) while v5 nests it. Comparing a real payload's
    `restaurants` key against the flat form marked every array-valued result
    as carrying extra fields -- 29 of 51 rejects in the first smoke.
    """
    if not isinstance(returns, dict):
        return set()
    props = returns.get("properties")
    if not isinstance(props, dict):
        return set()
    return {k.split(".")[0].removesuffix("[]") for k in props}


def _catalogue(msgs):
    for m in msgs:
        c = m.get("content") or ""
        if c.startswith("Værktøjer:"):
            try:
                return json.loads(c.split("Værktøjer:", 1)[1].strip()
                                  .split("\n\n")[0])
            except Exception:
                return []
    return []


def _last_user(msgs, upto):
    for m in reversed(msgs[:upto]):
        if m["role"] == "user":
            c = m.get("content") or ""
            return c.split("\n\n", 1)[-1] if c.startswith("Værktøjer:") else c
    return ""


def dangling(row):
    """(index, call, spec, question) for the row's unanswered terminal call.

    A call is dangling when no tool_result follows it. Only the terminal one
    can be -- mid-dialogue calls always got their result from the source.
    """
    msgs = row["messages"]
    call_at = None
    for i, m in enumerate(msgs):
        if m["role"] == "tool_call":
            nxt = msgs[i + 1]["role"] if i + 1 < len(msgs) else None
            if nxt != "tool_result":
                call_at = i
    if call_at is None:
        return None
    try:
        call = json.loads(msgs[call_at]["content"])
    except Exception:
        return None
    spec = next((t for t in _catalogue(msgs)
                 if t.get("name") == call.get("name")), None)
    if not spec:
        return None
    return call_at, call, spec, _last_user(msgs, call_at)


def spec_fingerprint(spec):
    """Hash of the contract the payload was generated against.

    The `returns` block is an INPUT to the result -- it becomes the response
    schema -- so a payload built for one version of it is not a valid answer
    for another. v4 stores the tree flattened and v5 nests it, and deduping
    the synonym fields (`final_price`/`new_price`/`discounted_price` all
    holding the same number) changes it again. Without this in the key those
    edits are invisible and the cache serves payloads shaped by a schema that
    no longer exists.

    Hashed over the CANONICAL form -- the JSON Schema actually sent to the
    provider -- not the raw block. v4 stores the tree flat and v5 nests it, but
    `_schema_from_returns` nests v4 before generating, so both produce the same
    contract and the same payload. Hashing the raw block would treat that pure
    representation change as a semantic one and discard 15,141 valid payloads;
    hashing the canonical form invalidates only tools whose field set really
    moved, such as those whose synonym fields get deduped.
    """
    canon = _schema_from_returns(spec.get("returns") or {})
    return hashlib.sha1(json.dumps(canon, sort_keys=True,
                                   ensure_ascii=False).encode()
                        ).hexdigest()[:12]


def cache_key(call, question, spec):
    return json.dumps([call.get("name"), call.get("arguments") or {},
                       question[:200], spec_fingerprint(spec)],
                      sort_keys=True, ensure_ascii=False)


# ── gate ────────────────────────────────────────────────────────────────────

LIST_ENUM = re.compile(r"(?m)^\s*\d+[.)](?=\s)")


def gate(result, answer, spec, context="", relevant=None):
    """Why each check exists is in the reason string; returns None if clean.

    `context` is the question plus the call -- numbers the user supplied are
    fair to repeat ("på 4. etage"), so only numbers appearing in NEITHER the
    payload nor the context count as invented.
    """
    if not isinstance(result, dict) or not result:
        return "result-not-an-object"
    if not isinstance(answer, str) or not answer.strip():
        return "answer-empty"
    answer = answer.strip()
    # The cap exists to stop the model padding, but some tools legitimately
    # return long text: a lyrics or article tool cannot answer in 70 words
    # without truncating the thing it was asked for. Raise the cap when the
    # payload itself is long -- 137 v7 rows were rejected for reciting the
    # lyrics they were asked to fetch.
    longest = max((len(str(v).split()) for _p, v in _leaves(result)), default=0)
    cap = 70 if longest <= 40 else min(400, 70 + longest)
    if len(answer.split()) > cap:
        return "answer-too-long"
    if META.search(answer):
        return "answer-is-meta"

    declared = _spec_fields(spec.get("returns") or {})
    if declared:
        got = set(result)
        if got - declared:
            return f"result-extra-fields:{sorted(got - declared)[:3]}"
        if declared - got:
            return f"result-missing-fields:{sorted(declared - got)[:3]}"

    # Grounding. The whole point of the turn is that the answer reads the
    # result, so an answer that shares no value with it is the failure mode
    # being trained away, not a mild stylistic miss.
    vals = [v for _, v in _leaves(result)
            if not isinstance(v, bool) and v not in (None, "")]
    low = answer.lower()
    # Numbers the payload states, including those inside strings ("10 dollars")
    # and list lengths ("der er 3 actionfilm").
    pool = {float(v) for v in vals if isinstance(v, (int, float))}
    # OPAQUE strings contribute no citable numbers. A base64 image, an event id
    # (`evt_12345`), an IMDb id (`tt0111161`) or a URL all contain digits, and
    # scraping them into the pool means a payload with nothing quotable still
    # looks like it states facts -- so the "payload states no fact" exemption
    # below never fires and a correct answer is rejected. 176 rows in v7.
    pool |= {float(m.group().replace(",", "."))
             for v in vals if isinstance(v, str) and not _opaque(v)
             for m in NUM.finditer(v)}
    pool |= {float(len(v)) for v in _arrays(result)}

    # Substring is the honest test and covers 84.4% on its own. Numeric
    # tolerance adds 7.5pp and every point of it is real: Danish writes 898.09
    # as "898,09" and 12000.0 as "12.000", and answers round 50.26548245743669
    # to "50.27". None of those are substrings of the payload.
    # DELIMITED FIELDS AND LONG STRINGS. Whole-value substring is too literal
    # for two payload shapes the corpus actually contains:
    #   "The Shawshank Redemption,The Godfather,The Dark Knight" -- an answer
    #   that reads it naturally ("…, The Godfather og The Dark Knight") shares
    #   no substring with the comma-joined original;
    #   "Begivenheden 'Project Meeting' er oprettet succesfuldt." -- quoted
    #   almost verbatim, but the trailing period breaks the match.
    # Both were scored ungrounded. ~420 of v7's 514 answer-not-grounded rows
    # are this, not a missing citation.
    def _cands(v):
        yield str(v)
        if isinstance(v, str):
            for part in re.split(r"[,;|\n]+", v):        # delimited fields
                part = part.strip()
                if len(part) > 2:
                    yield part
            words = v.split()
            if len(words) > 6:                           # long prose: prefix
                yield " ".join(words[:6])

    cites = any(c.lower() in low
                for v in vals for c in _cands(v) if len(c.strip()) > 2) or \
        any(_traces_to(m.group(), pool) for m in NUM.finditer(low))
    # A payload that states no fact -- {"message": "E-mail sendt", "status":
    # "succes"} -- has nothing to cite; the title and recipient in the answer
    # came from the question. 6.9% of pairs. Requiring a citation there
    # rejects correct answers for having nothing to quote.
    if not cites and pool:
        return "answer-not-grounded"

    # PRECISION. Grounding alone made reciting the payload optimal -- 43% of
    # v5's answers cite every field, and the eval scored that as success. The
    # generator now declares which fields the question asks for, so an answer
    # that drags in the others is rejected even though every value in it is
    # genuinely from the payload.
    if relevant:
        rel = [k for k in relevant if k in result]
        if not rel:
            return "relevant-fields-not-in-payload"
        if len(rel) > 3:
            return "too-many-relevant-fields"
        extra = {k: v for k, v in result.items() if k not in rel}
        # every relevant FIELD must survive into the answer -- per field, not
        # per flattened leaf, so `_relevant_cited` can tell a scalar (all of it
        # must appear) from a container (a summary of it may).
        for k in rel:
            v = result[k]
            if isinstance(v, bool) or v in (None, ""):
                continue
            if not _relevant_cited(answer, v, str(context)):
                return f"answer-misses-relevant:{str(v)[:24]}"
        # and no irrelevant one may
        for k, v in extra.items():
            for x in [y for _, y in _leaves({k: v})]:
                if isinstance(x, bool) or x in (None, ""):
                    continue
                if str(x).strip() and str(x) in str(context):
                    continue            # the user said it; repeating is fine
                if _cited(answer, x):
                    return f"answer-cites-irrelevant:{k}"

    # Invented numbers. `4 * 12 = 48` on a payload holding 28 is exactly what
    # the probe caught the model doing; an answer that does it here would
    # teach it.
    # Token sets, not substring search: `4` inside "på 4. etage" is followed by
    # a period, so a `(?![\d.,])` guard rejects the very number it should
    # allow -- which is what the clean-pair control caught.
    allowed = _nums(json.dumps(result, ensure_ascii=False) + " " + context)
    allowed |= pool
    # A digit opening a line and followed by "." or ")" is a LIST ENUMERATOR,
    # not a quantity. The gate returns the first unexplained number it finds,
    # so every answer that formats a payload list as "1. … 2. …" tripped on
    # the 1 -- 654 rows in v7, 23% of all invents-number flags and the single
    # largest reason in the corpus. Blanked before scanning rather than
    # skipped in the loop, so a real number later in the line is still caught.
    scan = LIST_ENUM.sub(lambda m: " " * len(m.group()), answer)
    for m in NUM.finditer(scan):
        if not _traces_to(m.group(), allowed):
            return f"answer-invents-number:{m.group()}"
    return None


# Planted defects, one per check that can fire on generated content. A gate
# that never fires is indistinguishable from clean data, so the run asserts it
# catches all of these before it trusts a single pass verdict.
_CUPS = {"returns": {"properties": {"cups_left": {}}}}
_CTX = "Er der kaffe tilbage på 4. etage?"
CONTROLS = [
    ({"cups_left": 8}, "Jeg har kaldt værktøjet og formateret parametrene.",
     _CUPS, _CTX),
    ({"cups_left": 8}, "Der er kaffe tilbage på etagen.", _CUPS, _CTX),
    ({"dog_years": 28}, "Din hund er 4 * 12 = 48 hundeår gammel.",
     {"returns": {"properties": {"dog_years": {}}}},
     "Min hund er 4 menneskeår gammel."),
    ({"cups_left": 8, "extra": 1}, "Der er 8 kopper tilbage.", _CUPS, _CTX),
    ({}, "Der er 8 kopper tilbage.", {"returns": {"properties": {}}}, _CTX),
    ({"cups_left": 8}, "   ", _CUPS, _CTX),
]


# Clean pairs that MUST pass. Each is a shape the first smoke wrongly rejected:
# a value followed by a period, a rounded float, a reformatted date, and a
# probability written as a percentage. Counts alone read all four as the model
# generating badly; the rejects file showed the gate was the problem.
CLEAN = [
    ({"cups_left": 8}, "Der er 8 kopper kaffe tilbage på 4. etage.",
     _CUPS, _CTX),
    ({"number": 42}, "Jeg har genereret et tilfældigt tal, som er 42.",
     {"returns": {"properties": {"number": {}}}}, "Giv mig et tilfældigt tal"),
    ({"area": 153.9380400259}, "Arealet af cirklen er 153.94 kvadratenheder.",
     {"returns": {"properties": {"area": {}}}}, "radius på 7"),
    ({"estimated_delivery": "2024-03-15", "status": "Leveret"},
     "Din pakke blev leveret den 15. marts 2024.",
     {"returns": {"properties": {"estimated_delivery": {}, "status": {}}}},
     "Hvor er min pakke?"),
    ({"confidence": 0.98, "sentiment": "negativ"},
     "Teksten udtrykker en negativ følelse med 98% sikkerhed.",
     {"returns": {"properties": {"confidence": {}, "sentiment": {}}}},
     "Analyser denne tekst"),
    ({"restaurants": [{"name": "Carbone", "rating": 4.7}]},
     "Jeg fandt Carbone med en bedømmelse på 4.7.",
     {"returns": {"properties": {"restaurants[]": {},
                                 "restaurants[].name": {},
                                 "restaurants[].rating": {}}}},
     "italienske restauranter"),
]


# The precision check needs its own controls: a gate whose new term never
# fires is indistinguishable from one that was never added.
_COFFEE = {"returns": {"properties": {"cups_left": {}, "days_since_service": {},
                                      "working": {}}}}
_PAY = {"cups_left": 8, "days_since_service": 12, "working": True}
_SCHED = {"schedule": [{"date": "2024-03-15"}, {"date": "2024-03-16"},
                       {"date": "2024-03-17"}], "sport": "NFL"}
_SCHED_SPEC = {"returns": {"properties": {"schedule[]": {},
                                          "schedule[].date": {},
                                          "sport": {}}}}
_RECIPES = {"recipes": [{"navn": "Kylling i tomatsauce", "køkken": "italiensk"},
                        {"navn": "Kylling cacciatore", "køkken": "italiensk"}]}
_RECIPES_SPEC = {"returns": {"properties": {"recipes[]": {},
                                            "recipes[].navn": {},
                                            "recipes[].køkken": {}}}}
PRECISION_CONTROLS = [
    (_PAY, "Der er 8 kopper tilbage, og den blev serviceret for 12 dage siden.",
     _COFFEE, "Er der kaffe tilbage?", ["cups_left"],
     "answer-cites-irrelevant"),
    # cites a payload value, but not the one asked for -- has to reach the
    # relevance check rather than being caught by the grounding check
    (_PAY, "Maskinen blev serviceret for 12 dage siden.", _COFFEE,
     "Er der kaffe tilbage?", ["cups_left"], "answer-misses-relevant"),
    (_PAY, "Der er 8 kopper tilbage.", _COFFEE, "Er der kaffe tilbage?",
     ["temperature"], "relevant-fields-not-in-payload"),
    # CONTAINER COVERAGE. A list field is satisfied by a summary, so the rule
    # is a majority of elements -- these two prove that still refuses an
    # answer that names none of them, and one that names a third.
    (_SCHED, "Programmet gælder NFL.", _SCHED_SPEC,
     "Hvornår spilles kampene?", ["schedule"], "answer-misses-relevant"),
    (_SCHED, "Første kamp er den 15. marts.", _SCHED_SPEC,
     "Hvornår spilles kampene?", ["schedule"], "answer-misses-relevant"),
    # names neither recipe -- every term it uses ("italienske", "kylling")
    # came from the question, so nothing shows the payload was read
    (_RECIPES, "Ja, jeg har fundet to italienske opskrifter med kylling.",
     _RECIPES_SPEC, "Find italienske opskrifter med kylling", ["recipes"],
     "answer-misses-relevant"),
]
PRECISION_CLEAN = [
    (_PAY, "Der er 8 kopper kaffe tilbage.", _COFFEE,
     "Er der kaffe tilbage?", ["cups_left"]),
    # ISO dates written out in Danish -- no substring survives the rendering
    (_SCHED, "Kampene spilles den 15. marts, 16. marts og 17. marts.",
     _SCHED_SPEC, "Hvornår spilles kampene?", ["schedule"]),
    # a comma-delimited field rejoined with a real connective, and the Danish
    # article for a leading 1
    ({"ingredienser": "1 moden banan, 1 spsk peanutbutter, 1/4 tsk kanel"},
     "Brug en moden banan, 1 spsk peanutbutter og 1/4 tsk kanel.",
     {"returns": {"properties": {"ingredienser": {}}}},
     "Hvad skal jeg bruge?", ["ingredienser"]),
    # a list of records delivered as a serialised JSON string, answered by
    # naming each record once
    ({"jobs": '[{"by": "mats", "id": 39423844, "text": "Senior Backend Engineer"},'
              ' {"by": "ana", "id": 39423845, "text": "Frontend Developer"}]'},
     "De seneste opslag er Senior Backend Engineer og Frontend Developer.",
     {"returns": {"properties": {"jobs": {}}}},
     "Vis mig de seneste jobopslag", ["jobs"]),
]


def check_controls():
    for res, ans, sp, ctx, rel, want in PRECISION_CONTROLS:
        got = gate(res, ans, sp, ctx, relevant=rel)
        if got is None or not got.startswith(want):
            raise SystemExit(f"precision control {want!r} -> {got!r}")
    for res, ans, sp, ctx, rel in PRECISION_CLEAN:
        got = gate(res, ans, sp, ctx, relevant=rel)
        if got is not None:
            raise SystemExit(f"precision gate rejects a clean pair: {got}")
    bad = [i for i, c in enumerate(CONTROLS)
           if gate(*c) is None]
    if bad:
        raise SystemExit(f"gate is blind: controls {bad} passed")
    for i, c in enumerate(CLEAN):
        why = gate(*c)
        if why is not None:
            raise SystemExit(f"gate rejects clean pair {i}: {why}")
    print(f"gate: {len(CONTROLS) + len(PRECISION_CONTROLS)} planted defects "
          f"caught, {len(CLEAN) + len(PRECISION_CLEAN)} clean pairs pass",
          flush=True)


# ── generation ──────────────────────────────────────────────────────────────

def _schema_from_returns(returns):
    """The tool's `returns` block as a JSON Schema for the result payload.

    Gating field membership after generation detects the wrong shape; handing
    the provider the schema prevents it. v4 stores the tree flattened
    (`restaurants[].address`), so it is nested first with the same helper the
    generator uses. Leaves carry only a Danish description and no type, so
    they stay type-free -- the point is the key set, which is what the model
    got wrong.
    """
    if not isinstance(returns, dict):
        return None
    props = returns.get("properties")
    if not isinstance(props, dict) or not props:
        return None
    if any("." in k or k.endswith("[]") for k in props):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from translate_toolmind_da import _nest_paths
        props = _nest_paths({k: (v if isinstance(v, dict) else {})
                             for k, v in props.items()})

    def conv(node):
        if not isinstance(node, dict):
            return {}
        t = node.get("type")
        if t == "array":
            return {"type": "array", "items": conv(node.get("items") or {})}
        sub = node.get("properties")
        if isinstance(sub, dict) and sub:
            return {"type": "object",
                    "properties": {k: conv(v) for k, v in sub.items()},
                    "required": sorted(sub),
                    "additionalProperties": False}
        return {"type": t} if t else {}

    return {"type": "object",
            "properties": {k: conv(v) for k, v in props.items()},
            "required": sorted(props), "additionalProperties": False}


async def one_answer(session, call, spec, question, tries=3):
    shown = {"navn": spec.get("name"),
             "beskrivelse": spec.get("description"),
             "parametre": spec.get("parameters"),
             "returnerer": spec.get("returns")}
    user = (f"VÆRKTØJ:\n{json.dumps(shown, ensure_ascii=False)}\n\n"
            f"BRUGERENS SPØRGSMÅL:\n{question}\n\n"
            f"KALDET:\n{json.dumps(call, ensure_ascii=False)}")
    rs = _schema_from_returns(spec.get("returns") or {})
    # With a schema the payload is constrained to the declared keys; without
    # one (1,532 tools carry no `returns`) the model returns it as a JSON
    # string and the gate is the only check. Those rows also invent redundant
    # synonym fields -- three keys all holding 160 -- so they are skipped
    # unless explicitly asked for.
    if rs is None:
        if not ALLOW_UNSPECED[0]:
            return None, "no-returns-spec", []
        payload = {"type": "string"}
    else:
        payload = rs
    body = {"model": MODEL, "temperature": 0.4, "max_tokens": 1600,
            "messages": [{"role": "system", "content": ANSWER_SYS},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "svar", "strict": True, "schema": {
                    "type": "object",
                    "properties": {"resultat": payload,
                                   "relevante_felter": {
                                       "type": "array",
                                       "items": {"type": "string"}},
                                   "svar": {"type": "string"}},
                    "required": ["resultat", "relevante_felter", "svar"],
                    "additionalProperties": False}}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    if a == tries - 1:
                        return None, f"http-{r.status}", []
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                out = json.loads(d["choices"][0]["message"]["content"])
                res = out["resultat"]
                if isinstance(res, str):
                    res = json.loads(res)
                return _coerce(res), out["svar"], out.get("relevante_felter") or []
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None, "exhausted", []


ALLOW_UNSPECED = [False]


async def main_async(args):
    import aiohttp
    from datasets import load_dataset

    check_controls()
    if args.rows:
        rows = [json.loads(l) for l in args.rows.open() if l.strip()]
    else:
        ds = load_dataset(args.repo, "sft", split=args.split)
        rows = [dict(r) for r in ds]
    if args.n:
        rows = rows[:args.n]
    print(f"{len(rows):,} rows from "
          f"{args.rows or args.repo + ':' + args.split}", flush=True)

    # DISTRACTOR FIELDS ARE INVISIBLE TO THIS STAGE, ON PURPOSE.
    #
    # `gen_return_distractors` extends the `returns` contract upstream, so by
    # the time a row arrives here its catalogue declares fields that exist to
    # be NOT cited. Two things would go wrong if this stage saw them.
    #
    # The fingerprint is in the cache key because the schema is an input to
    # the payload -- but that rule is about a field set that MOVES (v4 flat vs
    # v5 nested, synonyms deduped), where a cached payload is no longer valid.
    # A pure ADDITION is not that: the cached result still satisfies every
    # field it was built for. Letting the addition change the fingerprint would
    # re-buy ~15k answers to no purpose.
    #
    # And a payload generated here WITH the distractor declared would be a
    # payload the answer was written against -- so the answer might cite it,
    # which is precisely the property being bought. The value belongs in the
    # payload only after the answer is fixed; `inject_distractors.py` puts it
    # there, and covers source-supplied results too, which never reach this
    # splice at all.
    strip = {}
    if args.distractors:
        for line in args.distractors.open():
            if line.strip():
                rec = json.loads(line)
                strip[rec["sig"]] = {f["felt"] for f in rec["fields"]}
        print(f"stripping distractor fields for {len(strip):,} signatures "
              f"(cache keys stay on the pre-distractor contract)", flush=True)

    def _sig(fn):
        props = ((fn.get("parameters") or {}).get("properties") or {})
        return f"{fn.get('name')}({','.join(sorted(props))})"

    def _strip(spec):
        names = strip.get(_sig(spec))
        if not names:
            return spec
        rets = (spec.get("returns") or {}).get("properties") or {}
        if not (names & set(rets)):
            return spec
        out = json.loads(json.dumps(spec))
        props = out["returns"]["properties"]
        for n in names:
            props.pop(n, None)
        return out

    jobs = []
    for i, r in enumerate(rows):
        d = dangling(r)
        if d:
            call_at, call, spec, question = d
            jobs.append((i, (call_at, call, _strip(spec), question)))
    print(f"{len(jobs):,} dangling terminal calls "
          f"({100*len(jobs)/max(1,len(rows)):.1f}% of rows)", flush=True)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    if args.cache.exists():
        for line in args.cache.open():
            try:
                rec = json.loads(line)
                have[rec["k"]] = (rec["resultat"], rec["svar"],
                                  rec.get("relevante_felter") or [])
            except Exception:
                continue
    todo = [(i, d) for i, d in jobs
            if cache_key(d[1], d[3], d[2]) not in have]
    print(f"{len(have):,} cached, {len(todo):,} to generate", flush=True)
    if args.dry_run:
        # What a run would COST, before it costs it. The fingerprint means a
        # spec edit shows up here as a jump in `to generate`, which is the
        # signal worth seeing before launching.
        free = sum(1 for _, d in todo
                   if _schema_from_returns(d[2].get("returns") or {}) is None)
        print(f"dry-run: {len(todo)-free:,} would hit the API, "
              f"{free:,} skipped as no-returns-spec", flush=True)
        return

    # Generation and gating are separate passes: caching happens BEFORE the
    # verdict, so re-gating after a gate change costs nothing and a rejected
    # generation is never re-bought. Every gate fix in this file was found by
    # reading rejects, so that property paid for itself several times.
    done = [0]
    if todo:
        sem = asyncio.Semaphore(args.concurrency)
        lock = asyncio.Lock()
        failed = Counter()
        async with aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {_key()}",
                         "Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=300)) as s:
            with args.cache.open("a", buffering=1) as fh:
                async def run(i, d):
                    _, call, spec, question = d
                    async with sem:
                        res, ans, rel = await one_answer(
                            s, call, spec, question)
                    done[0] += 1
                    if done[0] % 100 == 0:
                        print(f"  generated {done[0]:,}/{len(todo):,}"
                              f"  ({sum(failed.values()):,} failed)",
                              flush=True)
                    if res is None:
                        failed[ans or "api-failed"] += 1
                        return
                    k = cache_key(call, question, spec)
                    async with lock:
                        have[k] = (res, ans, rel)
                        fh.write(json.dumps(
                            {"k": k, "tool": spec.get("name"),
                             "resultat": res, "svar": ans,
                             "relevante_felter": rel},
                            ensure_ascii=False) + "\n")
                await asyncio.gather(*[run(i, d) for i, d in todo])
        if failed:
            print("generation failures: "
                  + ", ".join(f"{w} {c:,}" for w, c in failed.most_common()),
                  flush=True)

    reasons = Counter()
    accepted = {}
    with args.rejects.open("w", buffering=1) as rej:
        for i, d in jobs:
            _, call, spec, question = d
            k = cache_key(call, question, spec)
            got = have.get(k)
            if not got:
                continue
            res, ans, rel = got
            why = gate(res, ans, spec,
                       question + " " + json.dumps(call, ensure_ascii=False),
                       relevant=rel)
            if why:
                reasons[why.split(":")[0]] += 1
                # Rejects are written out, not just counted. A gate tuned from
                # counts alone cannot tell over-strictness from dirty
                # generations, and the two want opposite fixes.
                rej.write(json.dumps(
                    {"why": why, "tool": spec.get("name"),
                     "declared": sorted(_spec_fields(spec.get("returns") or {})),
                     "resultat": res, "svar": ans, "q": question[:200]},
                    ensure_ascii=False) + "\n")
                continue
            accepted[k] = (res, ans, rel)

    kept = sum(1 for i, d in jobs
               if cache_key(d[1], d[3], d[2]) in accepted)
    print(f"accepted {kept:,}/{len(jobs):,} calls "
          f"({100*kept/max(1,len(jobs)):.1f}%)", flush=True)
    if reasons:
        print("rejected:")
        for w, c in reasons.most_common():
            print(f"  {w:<28} {c:,}")
    have = accepted

    # splice
    attached = 0
    for i, d in jobs:
        call_at, call, spec, question = d
        got = have.get(cache_key(call, question, spec))
        if not got:
            continue
        res, ans = got[0], got[1]
        rel = got[2] if len(got) > 2 else []
        msgs = rows[i]["messages"]
        msgs[call_at + 1:call_at + 1] = [
            {"role": "tool_result",
             "content": json.dumps(res, ensure_ascii=False)},
            {"role": "assistant", "content": ans}]
        # Carry the relevance label onto the ROW. Without it the eval must
        # recover "which fields did the question ask for" by string-matching
        # the reference reply against payload values -- a proxy that misses a
        # paraphrased field and over-matches a value two fields share. Free
        # here and exact; deriving it later is neither.
        rows[i].setdefault("answer_relevance", []).append(
            {"at": call_at + 2, "fields": rel})
        attached += 1
    print(f"attached to {attached:,} rows", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}", flush=True)

    if args.show:
        print("\n" + "=" * 74)
        for i, d in random.Random(0).sample(jobs, min(args.show, len(jobs))):
            for m in rows[i]["messages"][d[0] - 1:d[0] + 3]:
                c = m["content"]
                print(f"  [{m['role']:<11}] {c[:400]}")
            print("-" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="jensjepsen/danish-tool-dialogues-v4")
    ap.add_argument("--split", default="train")
    ap.add_argument("--rows", type=Path, default=None,
                    help="local jsonl instead of the hub")
    ap.add_argument("--distractors", type=Path, default=None,
                    help="distractor map; their fields are stripped from the "
                         "spec so cache keys stay on the pre-distractor "
                         "contract and payloads stay free of them")
    ap.add_argument("--out", type=Path,
                    default=Path("scratch/tool_answers/answered.jsonl"))
    ap.add_argument("--cache", type=Path,
                    default=Path("scratch/tool_answers/answers.jsonl"))
    ap.add_argument("--rejects", type=Path,
                    default=Path("scratch/tool_answers/rejects.jsonl"))
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="report how many calls would hit the API, then exit")
    ap.add_argument("--allow-unspeced", action="store_true",
                    help="also answer calls whose tool declares no returns; "
                         "their payloads are unconstrained and pad with "
                         "synonym fields")
    args = ap.parse_args()
    ALLOW_UNSPECED[0] = args.allow_unspeced
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
