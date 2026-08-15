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


def reward_gsm8k(completions: list[str], gold: list[str], **_):
    """1.0 if last number in completion equals gold (int/float normalized)."""
    out = []
    for c, g in zip(completions, gold):
        pred = _norm_num(_extract_num(c))
        target = _norm_num(g if _NUM_RE.fullmatch((g or "").strip())
                           else _extract_num(g))
        out.append(1.0 if (pred is not None and pred == target) else 0.0)
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


def reward_json_schema(completion: str, fields: list[str], strict: bool,
                       passage: str | None = None, types: list[str] | None = None) -> float:
    """Graded reward for schema-directed JSON gen. Returns in [0, 1].

    0.0             — unparseable JSON
    0.3             — parses, dict, no required fields matched
    +up to 0.4      — linear on fraction of required fields present (superset frac)
    +0.3            — all required present (bonus). Under `strict`, extra keys forfeit this.
    -0.1 per        — string-typed value NOT a substring of `passage` (grounding penalty)
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
    if passage:
        np_ = _norm_pass(passage)
        types = types or [""] * len(fields)
        for f, t in zip(fields, types):
            if t == "str":
                v = obj.get(f)
                if isinstance(v, str) and v and _norm_pass(v) not in np_:
                    r -= 0.1
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

    out = []
    for text, t, g, cons, p, f, ty, st, ps in zip(
        completions, task, gold, constraints, params, fields, types, strict, passage
    ):
        if t == "gsm8k":
            out.append(reward_gsm8k([text], gold=[g])[0])
        elif t == "json":
            if not f:
                out.append(0.0)
                continue
            out.append(reward_json_schema(text, f, bool(st), passage=ps or None, types=ty))
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
