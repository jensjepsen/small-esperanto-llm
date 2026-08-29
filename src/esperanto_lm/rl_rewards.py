"""Verifier-based reward functions for GRPO on Danish IF + GSM8K.

Two rewards:
- reward_gsm8k(completions, gold, **kw) -> [0/1]  numeric answer match
- reward_ifeval(completions, constraints, params, **kw) -> [0..1]  mean pass rate

Both callables are TRL GRPOTrainer-compatible: they receive `completions`
(list[str]) plus any dataset columns as keyword args, and return a list of
floats (one per completion). TRL applies its own advantage math on top.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Set GRPO_LEGACY_REWARDS=1 to disable all three post-hoc reward-tweaks
# (empty-output gate, duplicate-JSON-key penalty, arithmetic-error penalty)
# and get behavior matching the pre-2026-08-18 reward layer. Useful for
# A/B against historical H100 baselines that predate these gates.
_LEGACY = os.environ.get("GRPO_LEGACY_REWARDS") == "1"

# if_constraints.py lives in scripts/; add to path once
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from if_constraints import ALL as _IF_ALL  # noqa: E402

_IF_BY_NAME = {c.name: c for c in _IF_ALL}

# Google IFEval verifier registry (lazy-imported to keep base reward light)
_GOOGLE_REG = None


def _google_reg():
    global _GOOGLE_REG
    if _GOOGLE_REG is None:
        from ifeval_google import instructions_registry as reg  # noqa: E402
        _GOOGLE_REG = reg.INSTRUCTION_DICT
    return _GOOGLE_REG


# ── GSM8K ───────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_num(text: str | None) -> str | None:
    """Last number in the text, comma-stripped, or None."""
    if not text:
        return None
    matches = list(_NUM_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group().replace(",", "")


def _norm_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    try:
        f = float(s)
    except ValueError:
        return s
    import math
    if math.isnan(f) or math.isinf(f):
        return None  # can't score; treat as no-answer
    return str(int(f)) if f == int(f) else f"{f:g}"


# Match a whole arithmetic CHAIN `A op B op C ... = R` outside a bigger
# expression. Evaluates the full LHS with standard precedence so chained
# sums like `27 + 22 + 11 = 60` and mixed `2 + 3 * 4 = 14` are checked
# correctly instead of being false-positive-flagged from a sub-slice.
# Lookahead excludes `xX` so algebra like `= 2x + 5` doesn't get flagged
# as result `2` (the `x` immediately after signals variable, not multiply).
_ARITH_CHAIN = re.compile(
    r"(?<![\d./+\-*x×÷])"                                              # not mid-expr
    r"(-?\d+(?:[.,]\d+)?(?:\s*[+\-*×xX/÷]\s*-?\d+(?:[.,]\d+)?)+)"      # LHS chain
    r"\s*=\s*"
    r"(-?\d+(?:[.,]\d+)?)"                                             # result
    r"(?![\d./xX])"                                                     # not mid-expr / algebra var
)

_SAFE_EVAL_CHARS = re.compile(r"[\d.+\-*/ ]+")
_THOUSANDS_NUM = re.compile(r"^-?[1-9]\d{0,2}(?:\.\d{3})+$")


def _parse_num(s: str) -> float | None:
    """Parse a number token, respecting Danish/international conventions:
      '130.000' (period as thousands sep, groups of 3) → 130000
      '1.234.567'                                       → 1234567
      '1,5' (Danish decimal comma)                      → 1.5
      '1.5' (decimal point)                             → 1.5
    Leading-zero guarded so '0.500' stays 0.5, not 500."""
    s = s.strip().replace(" ", "")
    if _THOUSANDS_NUM.match(s):
        return float(s.replace(".", ""))
    try:
        return float(s.replace(",", "."))
    except (TypeError, ValueError):
        return None


_NUM_TOKEN = re.compile(r"-?\d+(?:[.,]\d+)*")


def _eval_chain(lhs: str) -> float | None:
    """Evaluate a validated arithmetic chain (digits, ops, whitespace only).
    Normalizes each numeric token via `_parse_num` (thousands-sep aware,
    Danish decimal aware), then ×/÷/x/X → */, then hands to eval()."""
    s = _NUM_TOKEN.sub(
        lambda m: (str(_parse_num(m.group()))
                   if _parse_num(m.group()) is not None else m.group()),
        lhs,
    )
    s = (s.replace("×", "*").replace("÷", "/")
             .replace("x", "*").replace("X", "*"))
    if not _SAFE_EVAL_CHARS.fullmatch(s):
        return None
    try:
        return float(eval(s, {"__builtins__": {}}, {}))
    except (SyntaxError, ZeroDivisionError, ValueError, TypeError):
        return None


def _wrong_equations(text: str) -> int:
    """How many equations have a wrong result. Matches full arithmetic chains
    with correct operator precedence — no false positives from sub-slicing
    multi-term sums or fractions."""
    if not text:
        return 0
    n = 0
    for m in _ARITH_CHAIN.finditer(text):
        lhs, c = m.groups()
        expected = _eval_chain(lhs)
        if expected is None:
            continue
        actual = _parse_num(c)
        if actual is None:
            continue
        if abs(expected - actual) < 1e-6:
            continue
        # Tolerate small float rounding (0.5%)
        if abs(expected - actual) / max(abs(expected), 1e-9) < 5e-3:
            continue
        n += 1
    return n


# Default "1" → penalty OFF unless a launcher explicitly opts in with
# GRPO_DISABLE_GSM_ARITH_PENALTY=0.
_NO_ARITH = os.environ.get("GRPO_DISABLE_GSM_ARITH_PENALTY", "1") == "1"
ARITH_PENALTY_PER_EQ = 0.0 if (_LEGACY or _NO_ARITH) else 0.05
ARITH_PENALTY_CAP = 6
"""Per-equation arithmetic-execution penalty for reward_gsm8k. Discovered
via eval_gsm8k_da_freshopt_dump.jsonl: ~13% of wrong-answer rows have a
detectable execution error (55/2=27, 7*49=333, chained-sum drop, etc.).
Penalty is intentionally small (final-answer reward remains dominant) but
gives GRPO a smooth signal to clean up mid-chain arithmetic — including
on correct-final rows where the model happens to hit the answer despite
a broken step.

OFF by default: measured on two matched mixed3 runs (LR=5e-6, beta=0.02,
combined-v4, 1:1:1 IF/gsm/json), penalty ON vs OFF over steps 125-1000:

  metric        OFF            ON
  ifeval PS     28.0-33.9      25.4-30.1   (OFF ahead ~3-4pp from step 375)
  json reward   59.5-85.6      55.3-83.8   (OFF ahead at 7/8 evals)
  gsm8k pass@1  19.1-26.3      21.0-26.0   (wash)

The penalty did not help the task it targets and cost IF and json. Likely
mechanism: unclamped rewards revive all-wrong groups that previously had
std=0 and contributed no gradient, raising gsm's effective share of each
mixed update and crowding out the other two tasks."""


def reward_gsm8k(completions: list[str], gold: list[str], **_):
    """1.0 if last number equals gold, minus 0.05 per detected wrong equation
    (capped at 6 → max −0.3 penalty). Range is [−0.3, 1.0] — deliberately
    NOT clamped at 0.

    The old `max(0.0, r)` floor made the penalty inert on wrong-answer rows:
    every wrong completion scored exactly 0.0 regardless of how many broken
    equations it contained. For an all-wrong rollout group that means
    std=0 → zero advantage → no gradient at all. Letting the reward go
    negative spreads those groups over [−0.3, 0], so GRPO can still learn
    "less wrong" on prompts the model cannot yet solve. GRPO normalises by
    group std, so the absolute sign of the reward is irrelevant to the
    update — only the within-group ordering matters."""
    out = []
    for c, g in zip(completions, gold):
        pred = _norm_num(_extract_num(c))
        target = _norm_num(g if _NUM_RE.fullmatch((g or "").strip())
                           else _extract_num(g))
        r = 1.0 if (pred is not None and pred == target) else 0.0
        r -= ARITH_PENALTY_PER_EQ * min(_wrong_equations(c), ARITH_PENALTY_CAP)
        out.append(r)
    return out


# ── IFEval (danish-IF v4 verifier set: 46 constraints) ─────────────────────

def _check_google(name: str, params: dict, text: str) -> bool:
    """Verify a `google:...` constraint. `name` includes the `google:` prefix.
    Returns False on any exception (missing param, unknown name, etc.)."""
    reg = _google_reg()
    key = name[len("google:"):]
    cls = reg.get(key)
    if cls is None:
        return False
    # Strip None values — ifeval-da's param dicts carry every possible key
    # (with None for unused ones), and Google's build_description rejects
    # None where it expects a real value.
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    try:
        inst = cls(instruction_id=key)
        inst.build_description(**clean)
        return bool(inst.check_following(text))
    except Exception:
        return False


MIN_COMPLETION_CHARS = 0 if _LEGACY else 10
"""Vacuous-output gate. Constraints like `no_lists`, `single_paragraph`,
`no_commas`, `keywords:forbidden_words`, `punctuation:no_comma`, and several
others vacuously pass on `""` or 1-char outputs — the model can then win
free reward by emitting nothing. Below this char threshold, treat the row
as reward 0 regardless of what the verifier says. Set to 0 via
GRPO_LEGACY_REWARDS=1 to disable."""


def reward_ifeval_combined(completions: list[str],
                           constraints: list[list[str]],
                           params: list[list[dict]],
                           **_):
    """Mixed reward for our-46 + google-schema constraints.

    Schema (matches data/grpo_if_rewrite_v1):
      constraints: list[list[str]]      — names, parallel per row
      params:      list[list[dict]]     — one param dict per constraint, parallel to constraints
    Names starting with 'google:' dispatch to the Google IFEval verifier;
    all others go to our 46-set. Reward per row = mean pass over listed constraints.
    """
    out = []
    for text, cons, plist in zip(completions, constraints, params):
        if not cons:
            out.append(0.0); continue
        if len((text or "").strip()) < MIN_COMPLETION_CHARS:
            out.append(0.0); continue
        # `plist` might be a JSON string on some HF versions — normalize
        if isinstance(plist, str):
            try:
                plist = json.loads(plist)
            except (TypeError, ValueError):
                plist = [{}] * len(cons)
        if plist is None or len(plist) != len(cons):
            plist = [{}] * len(cons)
        n_ok = 0
        for name, p in zip(cons, plist):
            if isinstance(p, str):
                try:
                    p = json.loads(p)
                except (TypeError, ValueError):
                    p = {}
            p = p or {}
            if name.startswith("google:"):
                if _check_google(name, p, text):
                    n_ok += 1
            else:
                c = _IF_BY_NAME.get(name)
                if c is None:
                    continue
                try:
                    if c.check(text, p):
                        n_ok += 1
                except Exception:
                    pass
        out.append(n_ok / len(cons))
    return out


def _json_candidates(text: str):
    """Yield JSON-string candidates from a completion in decreasing preference:
    raw, stripped, ```json fence contents, first {...} block. Shared by both
    the parse path (_try_parse_json) and the dupe-key audit (_dupe_key_extras)
    so they see the SAME payload — otherwise fenced outputs bypass the dupe
    penalty entirely."""
    yield text
    yield text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if m:
        yield m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        yield m.group(0)


def _try_parse_json(text: str):
    """Best-effort JSON extraction: raw, ```json fence, first {...}, first [...]."""
    for cand in _json_candidates(text):
        try:
            return json.loads(cand)
        except Exception:
            pass
    return None


def _norm_pass(s: str) -> str:
    return " ".join(s.lower().replace("\n", " ").split())


def _value_matches(pred, gold, t: str) -> bool:
    """Best-effort per-type equality against gold."""
    if pred is None:
        return False
    if t == "str":
        if not isinstance(pred, str):
            pred = str(pred)
        g = gold if isinstance(gold, str) else str(gold)
        return pred.strip().lower() == g.strip().lower()
    if t == "int":
        try:
            return int(str(pred).replace(",", "").split(".")[0].strip()) == int(gold)
        except (TypeError, ValueError):
            return False
    if t == "float":
        try:
            return abs(float(str(pred).replace(",", ".")) - float(gold)) < 1e-3
        except (TypeError, ValueError):
            return False
    if t == "bool":
        try:
            return bool(pred) == bool(gold)
        except Exception:
            return False
    if t == "list[str]":
        if not isinstance(pred, list) or not isinstance(gold, list):
            return False
        return sorted(str(x).strip().lower() for x in pred) == sorted(str(x).strip().lower() for x in gold)
    # dict / unknown — exact equal, best effort
    return pred == gold


def _is_nullish(v) -> bool:
    """None, empty string, or whitespace-only string."""
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _type_shape_ok(v, t: str) -> bool:
    """True if v matches the declared JSON type in the seed schema.
    Strict on output shape — string-in-int-slot / string-in-list-slot etc. fail.
    Gold-value match remains permissive (int-shaped string parses fine); this
    penalty is purely about the raw type."""
    if t == "str":
        return isinstance(v, str)
    if t == "int":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "float":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "bool":
        return isinstance(v, bool)
    if t == "list[str]":
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    if t == "dict":
        return isinstance(v, dict)
    return True  # unknown type — don't penalize


DUPE_KEY_PENALTY_PER_EXTRA = 0.0 if _LEGACY else 0.15
DUPE_KEY_PENALTY_CAP = 5


# Diagnostic: log resolved reward-tweak state at import so it's visible in
# every training log (no more guessing whether GRPO_LEGACY_REWARDS reached
# the process).
print(f"[rewards] GRPO_LEGACY_REWARDS={_LEGACY}  "
      f"GRPO_DISABLE_GSM_ARITH_PENALTY={_NO_ARITH}  "
      f"MIN_COMPLETION_CHARS={MIN_COMPLETION_CHARS}  "
      f"ARITH_PENALTY_PER_EQ={ARITH_PENALTY_PER_EQ}  "
      f"DUPE_KEY_PENALTY_PER_EXTRA={DUPE_KEY_PENALTY_PER_EXTRA}",
      flush=True)


def _dupe_key_extras(text: str) -> int:
    """How many DUPLICATE key emissions (beyond the first) at the top level.
    `json.loads` collapses duplicates last-wins, which lets the model earn
    full reward on degenerate `{"k": "v", "k": "v", "k": "v", ...}` output.
    We re-parse with object_pairs_hook to count extras.

    Runs the SAME fence-stripping / block-extraction as `_try_parse_json`
    (via `_json_candidates`) — otherwise ```json ... ``` fenced outputs
    (which the model produces by default) bypass the penalty entirely,
    leaving the dupe-spam hack unpunished."""
    pairs = None
    for cand in _json_candidates(text):
        try:
            pairs = json.loads(cand, object_pairs_hook=list)
            break
        except Exception:
            continue
    if pairs is None or not isinstance(pairs, list):
        return 0
    seen = set(); extras = 0
    for k, _ in pairs:
        if k in seen:
            extras += 1
        else:
            seen.add(k)
    return extras


def reward_json_schema(completion: str, fields: list[str], strict: bool,
                       passage: str | None = None, types: list[str] | None = None,
                       gold_values: dict | None = None) -> float:
    """Graded reward for schema-directed JSON gen.

    Score ladder (max 1.2 for rows with `gold_values`, 1.0 otherwise):
      0.0             — unparseable JSON
      0.3             — parses to a dict, no required fields matched
      +up to 0.4      — linear on fraction of required fields present (superset frac)
      +0.3            — all required present. Under `strict`, extra keys forfeit this.
      +up to 0.2      — value-match fraction vs `gold_values` (only when gold_values given).
                        Per-type: str case-ins-strip-equal, int/float exact-ish, bool
                        exact, list[str] case-ins set-equal.
      -0.15 per null  — for extract/rewrite/fill_template rows (i.e. `passage` given),
                        each required field whose output value is null / empty / whitespace-only.
                        Kills the fill_template null-template-parrot shortcut where
                        model scored ~1.0 by returning the empty scaffold verbatim.
      -0.15 per dupe  — duplicate top-level JSON keys (capped at 5). json.loads
                        collapses dupes silently → free reward on degenerate
                        `{"k":"v","k":"v",...}` output.

    Grounding penalty from earlier versions is dropped — gold value match is a
    stronger signal (a value matching gold IS grounded by construction).
    """
    obj = _try_parse_json(completion)
    if obj is None or not isinstance(obj, dict):
        return 0.0
    keys = set(obj.keys())
    required = set(fields)
    present = required & keys
    frac = len(present) / len(required)
    r = 0.3 + 0.4 * frac
    if present == required:
        if strict:
            if keys == required:
                r += 0.3
        else:
            r += 0.3
    if gold_values and isinstance(gold_values, dict):
        types = types or [""] * len(fields)
        n_gold = 0
        n_match = 0
        for f, t in zip(fields, types):
            if f not in gold_values:
                continue
            n_gold += 1
            if _value_matches(obj.get(f), gold_values[f], t):
                n_match += 1
        if n_gold > 0:
            r += 0.2 * (n_match / n_gold)
    if passage:
        # Null penalty: punishes template-parrot / lazy-fill hack.
        n_null = sum(1 for f in fields if f in obj and _is_nullish(obj[f]))
        r -= 0.15 * n_null
    if types:
        # Type-shape penalty (universal): punishes wrong raw JSON type per
        # field. E.g. `"pris": "175.00"` (str for float), `"labels": "a, b"`
        # (str for list[str]). Fires only for fields present in output;
        # missing-field is already handled by the superset-frac term.
        # Applies to all task_types incl. generate (which has no gold).
        n_type_mis = sum(
            1 for f, t in zip(fields, types)
            if f in obj and not _type_shape_ok(obj[f], t)
        )
        r -= 0.05 * n_type_mis
    # Duplicate-key penalty (universal): kills the dupe-spam degeneracy.
    r -= DUPE_KEY_PENALTY_PER_EXTRA * min(_dupe_key_extras(completion), DUPE_KEY_PENALTY_CAP)
    return round(max(0.0, r), 4)


# ── NER ─────────────────────────────────────────────────────────────────────

# Canonical entity buckets. Keys the model may use for each — it routinely
# answers with its own key names (`navn`, `placering`, `årstal`) instead of the
# requested ones, so a strict key match scores correct extractions as zero.
_NER_TYPES = ("person", "org", "sted", "dato")
# The JSON keys the PROMPT asks for. Deliberately distinct from _NER_TYPES:
# the internal bucket is "org" (gold labels, KEYMAP targets) but the prompt
# now requests "organisation", because the model populated "org" in 0/93 gold
# org entities and "organisation" in 21.7% of them. Schema conformance must be
# scored against what was ASKED FOR, or the term penalises the model for
# obeying the prompt.
_NER_REQUIRED_KEYS = ("person", "organisation", "sted", "dato")

# internal bucket -> the JSON key the prompt asks for
_NER_KEY_FOR_BUCKET = {"person": "person", "org": "organisation",
                       "sted": "sted", "dato": "dato"}
_NER_BUCKET_FOR_KEY = {v: k for k, v in _NER_KEY_FOR_BUCKET.items()}
_NER_GLOSS = {
    "person": 'personer under "person"',
    "org": 'organisationer under "organisation"',
    "sted": 'steder og lande under "sted"',
    "dato": 'datoer og årstal under "dato"',
}
_NER_BUCKET_ORDER = ("person", "org", "sted", "dato")


# ── NER prompt slots ────────────────────────────────────────────────────────
# The prompt is split into a FREE part and a CONTRACT part. Openings, layout
# and the wording of the conditions vary; the schema spec and key names do not,
# because the verifier keys off them.
#
# Why vary at all: the policy has twice keyed off surface tokens rather than
# reading the instruction — it converged on a memorised {person, places,
# dates, numbers} object, and renaming one key "org"->"organisation" moved org
# recall 0%->21.7% while *explaining* what an organisation is moved it 0%->0%.
# A constant opening is a constant prefix that can trigger the whole memorised
# continuation.
#
# CAUTION on the condition slots: their semantics are load-bearing. A
# paraphrase that quietly drops "præcis som de står i teksten" makes the model
# normalise entities and the verifier then marks correct answers wrong. Every
# variant here must preserve meaning — validated behaviourally, not by taste
# (see scripts/gen_ner_openings.py).

# Openings template over the REQUESTED TYPES rather than saying "navngivne
# enheder" generically. Two reasons: it is what a Danish speaker would
# actually write ("Find alle personer og steder i teksten"), and it makes the
# opening vary with the subset instead of being a constant prefix that can
# trigger a memorised continuation. The type names in the opening are plain
# Danish plurals — the JSON key names remain the contract and appear only in
# the schema spec.
_NER_PLURAL = {"person": "personer", "org": "organisationer",
               "sted": "steder", "dato": "datoer"}


def _da_list(items) -> str:
    """Danish enumeration: a / a og b / a, b og c."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " og " + items[-1]


# {types} is filled with the Danish plural list. LAYOUT: "any" works with the
# passage before or after; "after" claims the text is below and must only be
# used when the passage actually follows.
NER_OPENINGS = [
    ("Find alle {types} i denne tekst", "any"),
    ("Find alle {types} i teksten nedenunder", "after"),
    ("Udtræk {types} fra teksten nedenfor", "after"),
    ("Hvilke {types} optræder i denne tekst?", "any"),
    ("Gennemgå teksten og find alle {types}", "any"),
    ("Identificér samtlige {types} i teksten", "any"),
    ("Angiv de {types} der nævnes i teksten", "any"),
    ("Opgave: udtræk {types} fra den givne tekst", "any"),
    ("Kan du finde alle {types} i teksten?", "any"),
    ("List de {types} som forekommer i teksten", "any"),
]
# Used only when all four types are requested — a speaker would say "alle
# navngivne enheder" rather than enumerate every type.
NER_OPENINGS_ALLTYPES = [
    ("Find alle navngivne enheder i denne tekst", "any"),
    ("Udtræk alle navngivne enheder fra teksten nedenfor", "after"),
    ("Hvilke navngivne enheder optræder i denne tekst?", "any"),
    ("Gennemgå teksten og find alle navngivne enheder", "any"),
    ("Identificér samtlige navngivne enheder i teksten", "any"),
]
NER_COND_ONLYKEYS = [
    "Medtag kun de nævnte nøgler og ingen andre.",
    "Brug udelukkende de nøgler der er nævnt ovenfor.",
    "Svaret må ikke indeholde andre nøgler end de angivne.",
]
NER_COND_EMPTY = [
    "Er der ingen af en slags, så lad listen være tom.",
    "Hvis en type ikke forekommer, skal dens liste være tom.",
    "Findes der ingen af en given type, efterlades listen tom.",
]
NER_COND_VERBATIM = [
    "Skriv enhederne præcis som de står i teksten.",
    "Gengiv enhederne ordret som de fremgår af teksten.",
    "Enhederne skal skrives nøjagtigt som i teksten, uden ændringer.",
]


def ner_prompt(buckets, rng=None) -> str:
    """Prompt template for a SUBSET of entity types, with optional variation.

    rng=None returns the CANONICAL variant (first compatible opening, first of
    every condition, passage after the instruction) so held-out eval stays
    byte-stable and comparable across runs. Training passes an rng.

    Openings are chosen to match the drawn layout: one that says the text is
    "nedenfor" must not be paired with the passage placed first.

    Returns a template with a literal `{t}` slot for the sentence.
    """
    bs = [b for b in _NER_BUCKET_ORDER if b in set(buckets)]
    if not bs:
        raise ValueError("ner_prompt needs at least one bucket")
    keys = [_NER_KEY_FOR_BUCKET[b] for b in bs]
    shape = "{{" + ", ".join(f'"{k}": []' for k in keys) + "}}"
    gloss = ", ".join(_NER_GLOSS[b] for b in bs)
    types = _da_list(_NER_PLURAL[b] for b in bs)

    if rng is None:
        passage_first, pick = False, (lambda xs: xs[0])
    else:
        passage_first = rng.random() < 0.35
        pick = rng.choice

    # For the full request the "alle navngivne enheder" phrasings go FIRST, so
    # the canonical variant stays byte-identical to the prompt every measured
    # run used — held-out numbers remain comparable.
    bank = (NER_OPENINGS_ALLTYPES + NER_OPENINGS if len(bs) == 4
            else list(NER_OPENINGS))
    ok = [o for o, lay in bank if lay == "any" or not passage_first]
    opening = pick(ok).format(types=types)

    conds = " ".join([pick(NER_COND_ONLYKEYS), pick(NER_COND_EMPTY),
                      pick(NER_COND_VERBATIM)])
    spec = f"Svar kun med JSON på formen {shape} — {gloss}. {conds}"

    if passage_first:
        return f'Tekst:\n\n"{{t}}"\n\n{opening}\n\n{spec}'
    sep = "" if opening.rstrip().endswith("?") else ":"
    return f'{opening}{sep}\n\n"{{t}}"\n\n{spec}'


_NER_KEYMAP = {}
for _k in ("person", "personer", "people", "navn", "navne", "name", "names"):
    _NER_KEYMAP[_k] = "person"
for _k in ("org", "organisation", "organisationer", "organization",
           "organizations", "virksomhed", "virksomheder"):
    _NER_KEYMAP[_k] = "org"
for _k in ("sted", "steder", "places", "place", "placering", "lokation",
           "location", "locations", "land", "lande", "by", "byer"):
    _NER_KEYMAP[_k] = "sted"
for _k in ("dato", "datoer", "dates", "date", "aar", "år", "årstal",
           "aarstal", "year", "tid"):
    _NER_KEYMAP[_k] = "dato"

_NER_KV_RE = re.compile(r'"([^"]+)"\s*:\s*(\[[^\]]*\]|"[^"]*"|[-\d.]+)')
_NER_STR_RE = re.compile(r'"([^"]*)"')


def parse_ner(text: str) -> list[tuple[str, str]] | None:
    """Entities from a model completion. None = no JSON object at all.

    Tolerant on purpose: accepts the model's own key synonyms, accepts a bare
    scalar where a list was asked for, and preserves duplicate keys (json.loads
    silently keeps only the last).
    """
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    out = []
    for key_raw, val_raw in _NER_KV_RE.findall(m.group(0)):
        key = _NER_KEYMAP.get(key_raw.strip().lower())
        if key is None:
            continue
        v = val_raw.strip()
        if v.startswith("["):
            vals = _NER_STR_RE.findall(v)
        elif v.startswith('"'):
            vals = [v.strip('"')]
        else:
            vals = [v]
        for x in vals:
            x = x.strip()
            if x and x not in ("[]", "[],"):
                out.append((x.lower(), key))
    return out


def ner_emitted_keys(text: str) -> set[str]:
    """Top-level JSON keys the completion actually emitted, lower-cased."""
    if not text:
        return set()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return set()
    return {k.strip().lower() for k, _ in _NER_KV_RE.findall(m.group(0))}


def ner_schema_score(text: str, required_keys=None) -> float:
    """How well the completion follows the REQUESTED schema, in [0, 1].

    Fraction of the requested keys present, minus a penalty per extraneous
    top-level key. Scored on literal key names, not the synonym map, because
    that map is exactly what is blind to this axis.

    `required_keys` is the per-example subset; defaults to all four.
    """
    req = tuple(required_keys) if required_keys else _NER_REQUIRED_KEYS
    ks = ner_emitted_keys(text)
    if not ks:
        return 0.0
    present = len(ks & set(req)) / len(req)
    extra = len(ks - set(req))
    return max(0.0, min(1.0, present - 0.1 * min(extra, 6)))


def _ner_surface_match(a: str, b: str) -> bool:
    """Surface equality tolerant of Danish inflection.

    Danish inflects the entity itself: gold "Ruslands" (genitive) vs predicted
    "Rusland" is the same entity, but exact matching charges it as BOTH a false
    positive and a false negative — two penalties for a suffix. Accept a
    prefix relation with a short delta; require >=4 chars on the shorter side
    so "Dan"/"Danmark" doesn't slip through.
    """
    if a == b:
        return True
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    return len(lo) >= 4 and hi.startswith(lo) and (len(hi) - len(lo)) <= 3


def ner_match_counts(pred: list[tuple[str, str]],
                     gold: set[tuple[str, str]]) -> tuple[int, int, int]:
    """(tp, fp, fn) with exact matches resolved before inflectional ones.

    Two passes matter: a fuzzy match made first could consume the gold entry
    that an exact prediction needed, understating tp.
    """
    gold_left = list(gold)
    pred_left = []
    tp = 0
    for p in pred:                       # pass 1 — exact
        if p in gold_left:
            gold_left.remove(p)
            tp += 1
        else:
            pred_left.append(p)
    still = []
    for p in pred_left:                  # pass 2 — inflectional
        hit = None
        for i, g in enumerate(gold_left):
            if p[1] == g[1] and _ner_surface_match(p[0], g[0]):
                hit = i
                break
        if hit is None:
            still.append(p)
        else:
            gold_left.pop(hit)
            tp += 1
    return tp, len(still), len(gold_left)


def ner_emitted_keys(text: str) -> set[str]:
    """Top-level JSON keys the completion actually emitted, lower-cased."""
    if not text:
        return set()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return set()
    return {k.strip().lower() for k, _ in _NER_KV_RE.findall(m.group(0))}


def ner_schema_score(text: str, required_keys=None) -> float:
    """How well the completion follows the REQUESTED schema, in [0, 1].

    Fraction of the requested keys present, minus a penalty per extraneous
    top-level key. Scored on literal key names, not the synonym map, because
    that map is exactly what is blind to this axis.

    `required_keys` is the per-example subset; defaults to all four.
    """
    req = tuple(required_keys) if required_keys else _NER_REQUIRED_KEYS
    ks = ner_emitted_keys(text)
    if not ks:
        return 0.0
    present = len(ks & set(req)) / len(req)
    extra = len(ks - set(req))
    return max(0.0, min(1.0, present - 0.1 * min(extra, 6)))


def _ner_surface_match(a: str, b: str) -> bool:
    """Surface equality tolerant of Danish inflection.

    Danish inflects the entity itself: gold "Ruslands" (genitive) vs predicted
    "Rusland" is the same entity, but exact matching charges it as BOTH a false
    positive and a false negative — two penalties for a suffix. Accept a
    prefix relation with a short delta; require >=4 chars on the shorter side
    so "Dan"/"Danmark" doesn't slip through.
    """
    if a == b:
        return True
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    return len(lo) >= 4 and hi.startswith(lo) and (len(hi) - len(lo)) <= 3


def ner_match_counts(pred: list[tuple[str, str]],
                     gold: set[tuple[str, str]]) -> tuple[int, int, int]:
    """(tp, fp, fn) with exact matches resolved before inflectional ones.

    Two passes matter: a fuzzy match made first could consume the gold entry
    that an exact prediction needed, understating tp.
    """
    gold_left = list(gold)
    pred_left = []
    tp = 0
    for p in pred:                       # pass 1 — exact
        if p in gold_left:
            gold_left.remove(p)
            tp += 1
        else:
            pred_left.append(p)
    still = []
    for p in pred_left:                  # pass 2 — inflectional
        hit = None
        for i, g in enumerate(gold_left):
            if p[1] == g[1] and _ner_surface_match(p[0], g[0]):
                hit = i
                break
        if hit is None:
            still.append(p)
        else:
            gold_left.pop(hit)
            tp += 1
    return tp, len(still), len(gold_left)


def _parse_ner_gold(gv):
    """(gold_entities, requested_buckets) from the serialized `gold_values`.

    Two shapes are accepted. Legacy is a bare list [[surface, type], ...] and
    implies all four types were requested. Current is
    {"ents": [...], "buckets": ["person", "org"]} so each example can request a
    SUBSET — the model must read the schema rather than emit a memorised one.

    Serialized because Arrow needs a single column schema across all
    interleaved tasks.
    """
    if not gv:
        return set(), list(_NER_BUCKET_ORDER)
    if isinstance(gv, str):
        try:
            gv = json.loads(gv)
        except (TypeError, ValueError):
            return set(), list(_NER_BUCKET_ORDER)
    buckets = list(_NER_BUCKET_ORDER)
    items = gv
    if isinstance(gv, dict):
        items = gv.get("ents") or []
        b = gv.get("buckets")
        if b:
            buckets = [x for x in _NER_BUCKET_ORDER if x in set(b)]
    out = set()
    for item in items or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            s, t = item
            if isinstance(s, str) and t in buckets:   # gold filtered to request
                out.add((s.strip().lower(), t))
    return out, buckets


# Weight of the schema-conformance term in reward_ner. The extraction F1 keeps
# (1 - weight) so a model that extracts well but uses the wrong keys still
# earns most of the reward — that tolerance is what gave 79% gradient coverage
# at k=32 and must not be thrown away — while correct keys become strictly
# better. 0 disables the term entirely (pre-2026-08-28 behaviour).
NER_SCHEMA_WEIGHT = float(os.environ.get("GRPO_NER_SCHEMA_WEIGHT", "0.15"))
# Per-duplicate-entity penalty, capped at 4 duplicates (max -0.20).
NER_DUPE_PENALTY = float(os.environ.get("GRPO_NER_DUPE_PENALTY", "0.05"))


def reward_ner(completion: str, gold_values) -> float:
    """Entity-level F1 in [0, 1], plus a schema-conformance term.

    F1 rather than exact-set-match deliberately: measured on the v31 base at
    the run's real config (k=32, temp 1.0), exact match yields gradient on only
    26% of entity-bearing prompts vs 58% under F1. Exact match zeroes an entire
    group as soon as one span is wrong, so most groups have no spread and
    contribute nothing.

    Empty-and-correct scores 1.0 — abstention on an entity-free sentence is the
    right answer and must be rewarded, since a JSON-shape reward (which cannot
    tell a real entity from an invented one) previously drove the policy to
    invent entities on 301/301 entity-free sentences.

    Precision is half of F1, so hallucination is penalised directly. That is
    the property the schema reward lacked.

    The schema term exists because the tolerant parser above accepts the
    model's own key names, which means a model can score full marks while
    ignoring the requested format. Measured after 13,375 steps without it: the
    policy converged on `person/places/dates/numbers` (its SFT
    textman_extraction template) in 565/565 held-out rows, so `org` had no slot
    in the output space at all and all 151 gold org entities were unreachable —
    not hard, structurally absent. The base model did emit org keys, so this
    was a capability RL removed because nothing scored it.
    """
    pred_list = parse_ner(completion)
    gold, buckets = _parse_ner_gold(gold_values)
    if pred_list is None:
        return 0.0                      # no JSON emitted at all
    # Predictions of types we did NOT ask for are spurious, not ignored — the
    # instruction named which keys to produce.
    pred_list = [(x, t) for x, t in pred_list if t in buckets]
    req_keys = [_NER_KEY_FOR_BUCKET[b] for b in buckets]

    # Duplicates are counted BEFORE dedup. set() made repeated entities free,
    # so the policy emitted them in 97/565 held-out rows at zero cost —
    # ["Birte Weiss", "Birte Weiss"]. Same hole as the schema one: a tolerance
    # in the scorer removing a gradient. reward_json_schema already penalises
    # duplicate KEYS; this is the value-level equivalent.
    n_dupes = len(pred_list) - len(set(pred_list))
    pred_u = list(dict.fromkeys(pred_list))     # dedup, order-stable
    pred = set(pred_u)

    if not pred and not gold:
        f1 = 1.0                        # correctly abstained
    elif not pred or not gold:
        f1 = 0.0                        # emitted on empty, or empty on non-empty
    else:
        tp, fp, fn = ner_match_counts(pred_u, gold)
        if tp == 0:
            f1 = 0.0
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec)

    if NER_SCHEMA_WEIGHT <= 0:
        return round(max(0.0, f1 - NER_DUPE_PENALTY * min(n_dupes, 4)), 4)
    # MULTIPLICATIVE, not additive. An additive term pays a floor for merely
    # emitting the right key names even when the answer is wrong (empty output
    # on an entity row scored 0.15), which lifts the always-abstain plateau
    # from 0.28 to ~0.39 on the shipped 28%-empty mix — undoing the point of
    # --ner-empty-frac, since the base is already an 82-93% abstainer. Scaling
    # F1 instead keeps every wrong answer at 0 while preserving the same
    # correct-key advantage at equal extraction quality.
    w = NER_SCHEMA_WEIGHT
    r = f1 * ((1.0 - w) + w * ner_schema_score(completion, req_keys))
    return round(max(0.0, r - NER_DUPE_PENALTY * min(n_dupes, 4)), 4)


ICL_SCHEMA_WEIGHT = float(os.environ.get("GRPO_ICL_SCHEMA_WEIGHT", "0.25"))


def _icl_canon():
    """gen_icl_schema_format lives in scripts/ (already on sys.path above).
    Imported lazily so the base reward layer does not pay for it."""
    from gen_icl_schema_format import NULL, canon
    return canon, NULL


def reward_icl(completion: str, gold: str, fields, fmt: str) -> float:
    """(key, value)-pair F1 for schema/format induction, times a key-set term.

    F1 rather than exact match for the same reason as reward_ner: exact match
    zeroes a whole group the moment one value is wrong, so most groups have no
    spread and give no gradient. Measured on this task's own eval splits, exact
    match runs ~11pp below key-set match (85.3 vs 96.4 on eval_format) --- that
    gap is precisely the rows where partial credit exists and binary scoring
    throws it away.

    Unparseable output scores 0.0: the format IS the task here, so an answer
    that cannot be parsed in the requested format is not partially right.

    The key-set term is MULTIPLICATIVE, following reward_ner. Additive paid a
    floor for merely naming the right keys, which is the failure mode this task
    is most prone to --- the model already reaches 93-96% key-set match while
    exact sits at 47-85%, so an additive term would hand out most of its mass
    for the part that is already solved.
    """
    canon, NULL = _icl_canon()
    keys = list(fields or [])
    if not keys or not fmt:
        return 0.0
    try:
        pred_d = canon(completion, fmt, keys)
        gold_d = canon(gold, fmt, keys)
    except Exception:
        return 0.0
    if pred_d is None or gold_d is None:
        return 0.0

    def _pairs(d):
        return {(k, v) for k, vs in d.items() for v in vs if v != NULL}

    pred, gld = _pairs(pred_d), _pairs(gold_d)
    if not pred and not gld:
        f1 = 1.0                       # both all-empty: correct abstention
    elif not pred or not gld:
        f1 = 0.0
    else:
        tp = len(pred & gld)
        if tp == 0:
            f1 = 0.0
        else:
            prec, rec = tp / len(pred), tp / len(gld)
            f1 = 2 * prec * rec / (prec + rec)

    if ICL_SCHEMA_WEIGHT <= 0:
        return round(f1, 4)
    # Emitting keys that were never requested is spurious, not free --- the
    # demonstrations define the key set and inventing one is a schema error.
    pk, gk = set(pred_d), set(gold_d)
    if not pk and not gk:
        kf1 = 1.0
    elif not pk or not gk:
        kf1 = 0.0
    else:
        ktp = len(pk & gk)
        kf1 = 0.0 if ktp == 0 else (2 * (ktp / len(pk)) * (ktp / len(gk))
                                    / ((ktp / len(pk)) + (ktp / len(gk))))
    return round(f1 * ((1 - ICL_SCHEMA_WEIGHT) + ICL_SCHEMA_WEIGHT * kf1), 4)


def reward_mixed(completions: list[str],
                 task: list[str],
                 gold: list[str],
                 constraints: list[list[str]],
                 params: list[list[dict]],
                 fields: list[list[str]] | None = None,
                 types: list[list[str]] | None = None,
                 strict: list[bool] | None = None,
                 passage: list[str | None] | None = None,
                 gold_values: list[str | dict | None] | None = None,
                 **_):
    """Per-example dispatch reward for mixing gsm8k + combined-IF + json training.

    Each row carries a `task` marker; reward dispatches to the appropriate
    verifier. Unused columns can be empty defaults per row. TRL passes all
    dataset columns as kwargs; JSON-specific columns (fields/types/strict/
    passage) are optional so pre-JSON mixed datasets remain compatible.

    Returns per-row scalar in [0, 1]:
      gsm8k          → reward_gsm8k
      ifeval/combined → reward_ifeval_combined
      json           → reward_json_schema
      ner            → reward_ner (gold entities ride in `gold_values`)
      icl            → reward_icl (keys in `fields`, format in `types[0]`)
    """
    N = len(completions)
    fields = fields or [None] * N
    types = types or [None] * N
    strict = strict or [False] * N
    passage = passage or [None] * N
    gold_values = gold_values or [None] * N

    out = []
    for text, t, g, cons, p, f, ty, st, ps, gv in zip(
        completions, task, gold, constraints, params, fields, types, strict, passage, gold_values
    ):
        if t == "gsm8k":
            out.append(reward_gsm8k([text], gold=[g])[0])
        elif t == "ner":
            out.append(reward_ner(text, gv))
        elif t == "icl":
            # keys ride in `fields`, the format name in `types[0]` --- reusing
            # the union columns rather than widening the schema, which every
            # other builder would then have to default.
            out.append(reward_icl(text, g, f, (ty or [None])[0]))
        elif t == "json":
            if not f:
                out.append(0.0)
                continue
            # gold_values arrives from Arrow as a JSON string (dict schemas
            # vary per row, so the dataset stores it serialized).
            gv_dict = None
            if gv:
                if isinstance(gv, dict):
                    gv_dict = gv
                elif isinstance(gv, str) and gv.strip():
                    try:
                        gv_dict = json.loads(gv)
                    except (TypeError, ValueError):
                        gv_dict = None
            out.append(reward_json_schema(
                text, f, bool(st), passage=ps or None, types=ty,
                gold_values=gv_dict,
            ))
        else:  # ifeval / combined
            out.append(reward_ifeval_combined([text], [cons or []], [p or []])[0])
    # Stash per-example (task, reward) pairs so the trainer wrapper can bucket
    # by task and log per-task stats. Overwritten every call.
    global LAST_MIXED_TASKS, LAST_MIXED_REWARDS
    LAST_MIXED_TASKS = list(task)
    LAST_MIXED_REWARDS = list(out)
    return out


# Per-call diagnostic buffer populated by reward_mixed. `_gen_score_with_skip`
# in train_grpo_verifier.py reads these to log per-task reward / advantage
# distributions to wandb. Reset every call to reward_mixed.
LAST_MIXED_TASKS: list[str] | None = None
LAST_MIXED_REWARDS: list[float] | None = None


def reward_ifeval(completions: list[str],
                  constraints: list[list[str]],
                  params: list[str],
                  **_):
    """Mean fraction of listed constraints satisfied by each completion.

    `constraints`: list of constraint-name lists (one per row).
    `params`:      list of JSON-encoded {name: kwargs} dicts (one per row).
    Returns a float in [0, 1] per row.
    """
    out = []
    for text, cons, p_json in zip(completions, constraints, params):
        try:
            p = json.loads(p_json) if isinstance(p_json, str) else (p_json or {})
        except (TypeError, ValueError):
            p = {}
        if not cons:
            out.append(0.0)
            continue
        if len((text or "").strip()) < MIN_COMPLETION_CHARS:
            out.append(0.0)
            continue
        n_ok = 0
        for name in cons:
            c = _IF_BY_NAME.get(name)
            if c is None:
                continue
            try:
                if c.check(text, p.get(name, {})):
                    n_ok += 1
            except Exception:
                pass
        out.append(n_ok / len(cons))
    return out
