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
        return str(int(f)) if f == int(f) else f"{f:g}"
    except ValueError:
        return s


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
