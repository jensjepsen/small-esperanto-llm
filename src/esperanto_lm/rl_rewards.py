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
import re
import sys
from pathlib import Path

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


# Match `A op B = C` outside a bigger expression (no fraction/chained-sum
# context on either side). Rejects things like `2/3 * 9 = 6` where the
# regex would otherwise pull `3 * 9 = 6` and call it wrong.
_ARITH_EQ = re.compile(
    r"(?<![\d./+\-*x×÷])"
    r"(-?\d+(?:[.,]\d+)?)\s*"
    r"([+\-*×xX/÷])\s*"
    r"(-?\d+(?:[.,]\d+)?)\s*=\s*"
    r"(-?\d+(?:[.,]\d+)?)"
    r"(?![\d./])"
)


def _eval_binop(a: str, op: str, b: str) -> float | None:
    try:
        x = float(a.replace(",", "."))
        y = float(b.replace(",", "."))
    except (TypeError, ValueError):
        return None
    if op == "+": return x + y
    if op == "-": return x - y
    if op in "*×xX": return x * y
    if op in "/÷":
        if y == 0: return None
        return x / y
    return None


def _wrong_equations(text: str) -> int:
    """How many `A op B = C` lines have the wrong C. Tight regex avoids
    fraction/multi-term false positives; a residual ~1-in-8 false-positive
    rate is acceptable at the per-equation reward magnitudes used below."""
    if not text:
        return 0
    n = 0
    for m in _ARITH_EQ.finditer(text):
        a, op, b, c = m.groups()
        expected = _eval_binop(a, op, b)
        if expected is None:
            continue
        try:
            actual = float(c.replace(",", "."))
        except (TypeError, ValueError):
            continue
        if abs(expected - actual) < 1e-6:
            continue
        # Tolerate small float rounding (0.5%)
        if abs(expected - actual) / max(abs(expected), 1e-9) < 5e-3:
            continue
        n += 1
    return n


ARITH_PENALTY_PER_EQ = 0.05
ARITH_PENALTY_CAP = 6
"""Per-equation arithmetic-execution penalty for reward_gsm8k. Discovered
via eval_gsm8k_da_freshopt_dump.jsonl: ~13% of wrong-answer rows have a
detectable execution error (55/2=27, 7*49=333, chained-sum drop, etc.).
Penalty is intentionally small (final-answer reward remains dominant) but
gives GRPO a smooth signal to clean up mid-chain arithmetic — including
on correct-final rows where the model happens to hit the answer despite
a broken step."""


def reward_gsm8k(completions: list[str], gold: list[str], **_):
    """1.0 if last number equals gold, minus 0.05 per detected wrong equation
    (capped at 6 → max −0.3 penalty)."""
    out = []
    for c, g in zip(completions, gold):
        pred = _norm_num(_extract_num(c))
        target = _norm_num(g if _NUM_RE.fullmatch((g or "").strip())
                           else _extract_num(g))
        r = 1.0 if (pred is not None and pred == target) else 0.0
        r -= ARITH_PENALTY_PER_EQ * min(_wrong_equations(c), ARITH_PENALTY_CAP)
        out.append(max(0.0, r))
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


MIN_COMPLETION_CHARS = 10
"""Vacuous-output gate. Constraints like `no_lists`, `single_paragraph`,
`no_commas`, `keywords:forbidden_words`, `punctuation:no_comma`, and several
others vacuously pass on `""` or 1-char outputs — the model can then win
free reward by emitting nothing. Below this char threshold, treat the row
as reward 0 regardless of what the verifier says."""


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


def _try_parse_json(text: str):
    """Best-effort JSON extraction: raw, ```json fence, first {...}, first [...]."""
    for cand in [text, text.strip()]:
        try:
            return json.loads(cand)
        except Exception:
            pass
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
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


def _dupe_key_extras(text: str) -> int:
    """How many DUPLICATE key emissions (beyond the first) at the top level.
    `json.loads` collapses duplicates last-wins, which lets the model earn
    full reward on degenerate `{"k": "v", "k": "v", "k": "v", ...}` output.
    We re-parse with object_pairs_hook to count extras."""
    try:
        pairs = json.loads(text, object_pairs_hook=list)
    except Exception:
        return 0
    if not isinstance(pairs, list):
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
    r -= 0.15 * min(_dupe_key_extras(completion), 5)
    return round(max(0.0, r), 4)


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
    return out


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
