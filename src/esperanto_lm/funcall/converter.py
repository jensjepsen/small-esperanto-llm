"""Convert verbal-CoT math datasets into multi-turn funcall format.

Produces rows of the shape::

    [user]      Q
    [assistant] narration <|tool_call|>EXPR<|/tool_call|>
    [tool]      <|tool_result|>VAL<|/tool_result|>
    [assistant] more narration ... (optionally another tool_call/result)
    [assistant] #### N

Two extractors:

GSM8K — uses the explicit ``<<expr=result>>`` markers
    that come from the original openai/gsm8k annotations. Direct,
    near-lossless (~97% of rows convert).

Orca-math — plain prose. Heuristic regex matches
    multi-operand ``A op B (op C)* = D`` expressions and verifies each
    with sympy. Stricter filters drop rows with bad final-answer claims
    or algebraic ``x = N`` leftovers so training stays clean.
"""
from __future__ import annotations

import json
import re

from .tokens import (
    TOOL_CALL_OPEN as TC_OPEN,
    TOOL_CALL_CLOSE as TC_CLOSE,
    TOOL_RESULT_OPEN as TR_OPEN,
    TOOL_RESULT_CLOSE as TR_CLOSE,
)

__all__ = [
    "convert_gsm_answer",
    "convert_orca_answer",
    "verify_arith",
    "BRIDGE_ON_REUSE",
    "STRIP_VERBAL",
]

# Converter options.
#   STRIP_VERBAL: when True, strip "EXPR = " before the tool call. Default
#     False so the model sees bridging narration that references prior
#     results.
#   BRIDGE_ON_REUSE: when True, inject "nun ni havas R, do " before any
#     call whose first operand equals the previous tool result and isn't
#     already mentioned in the surrounding narrative.
STRIP_VERBAL = False
BRIDGE_ON_REUSE = True


def verbal_strip_pattern(expr: str) -> str:
    """Regex matching the verbal form of an expression at end of prefix.

    Used to remove the redundant "EXPR = " that often precedes a tool
    call in source text. Accepts common operator variants the source
    might use (``x``, ``×`` for multiplication; ``÷`` for division;
    comma decimal separator).
    """
    out = []
    for ch in expr:
        if ch == "*":
            out.append(r"\s*[*x×]\s*")
        elif ch == "/":
            out.append(r"\s*[/÷]\s*")
        elif ch == "+":
            out.append(r"\s*\+\s*")
        elif ch == "-":
            out.append(r"\s*-\s*")
        elif ch == ".":
            out.append(r"[.,]")
        else:
            out.append(re.escape(ch))
    return (r"\s*[$€£¥]?\s*" + "".join(out) + r"\s*=\s*[$€£¥]?\s*$")


def verify_arith(expr: str, expected: str) -> bool:
    """Check that sympy(expr) ≈ expected (within 1e-4 float tolerance).

    Sympify handles ``^`` as exponent, deep parentheses, and rational
    arithmetic — and crucially won't execute arbitrary code.
    """
    try:
        from sympy import sympify
        safe = re.sub(r"[^\d+\-*/.()^ ]", "", expr).replace(" ", "")
        got = sympify(safe, evaluate=True)
        try:
            return abs(float(got) - float(expected)) < 1e-4
        except Exception:
            return str(got).strip() == str(expected).strip()
    except Exception:
        return False


def _starts_with_value(expr: str, value: str) -> bool:
    """Does `expr` start with `value` as the first numeric operand?"""
    expr_s = expr.strip()
    val_s = str(value).strip()
    if not val_s or not expr_s:
        return False
    pat = r"^-?" + re.escape(val_s) + r"(?=$|[\s+\-*/().,])"
    return re.search(pat, expr_s) is not None


# ---------- GSM8K converter ----------

GSM_CALC = re.compile(r"<<([^=>]+)=([^>]+)>>([^\s]*)?")


def convert_gsm_answer(answer_text: str) -> list[dict] | None:
    """Convert a GSM-style answer with ``<<expr=result>>`` markers into a
    multi-turn message sequence to follow the user prompt.

    Returns a list of ``{role, content}`` dicts (assistant + tool turns,
    coalesced where consecutive) or ``None`` if no convertible
    arithmetic was found, or if any marked computation fails sympy
    verification (we drop the whole row rather than emit a bad call).
    """
    parts: list[tuple[str, str]] = []
    cursor = 0
    matched_any = False
    prev_result: str | None = None

    for m in GSM_CALC.finditer(answer_text):
        expr, result, _inline_num = m.group(1), m.group(2), m.group(3) or ""
        if not verify_arith(expr, result):
            return None  # bail on first bad calc — keep dataset clean
        matched_any = True
        prefix_text = answer_text[cursor : m.start()]
        if STRIP_VERBAL:
            prefix_text = re.sub(verbal_strip_pattern(expr), "", prefix_text)
        if prefix_text.strip():
            parts.append(("assistant", prefix_text.strip()))
        if (
            BRIDGE_ON_REUSE
            and prev_result is not None
            and _starts_with_value(expr, prev_result)
            and prev_result not in (prefix_text or "")
        ):
            parts.append(("assistant", f"nun ni havas {prev_result} , do"))
        parts.append(("assistant", f"{TC_OPEN}{expr.strip()}{TC_CLOSE}"))
        parts.append(("tool", f"{TR_OPEN}{result.strip()}{TR_CLOSE}"))
        prev_result = result.strip()
        cursor = m.end()

    if not matched_any:
        return None

    tail = answer_text[cursor:].strip()
    if tail:
        parts.append(("assistant", tail))

    # Coalesce consecutive same-role parts
    coalesced: list[list] = []
    for role, content in parts:
        if coalesced and coalesced[-1][0] == role:
            coalesced[-1][1] = coalesced[-1][1] + " " + content
        else:
            coalesced.append([role, content])
    return [{"role": r, "content": c} for r, c in coalesced]


# ---------- Orca-math converter (heuristic) ----------

# Multi-operand: A op B (op C)* = D. Operators: + - * / ×
ORCA_CALC = re.compile(
    r"(\d+(?:[.,]\d+)?"
    r"(?:\s*[+\-*/×]\s*\d+(?:[.,]\d+)?){1,})"
    r"\s*=\s*"
    r"(-?\d+(?:[.,]\d+)?)"
)

# Row-level final-answer consistency check (permissive: allows period or
# label words between expression-end and the `=`, to catch cross-sentence
# claims like "totala brikoj = 14 + ... + 12. totalaj brikoj = 170.").
ROW_FINAL_CHECK = re.compile(
    r"(?<![a-zA-Z])"
    r"(\d+(?:[.,]\d+)?(?:\s*[+\-*/×]\s*\d+(?:[.,]\d+)?){1,})"
    r"[\s.,]+"
    r"(?:[a-zA-Z][\w]*[\s.,]+){0,4}"
    r"=\s*"
    r"(-?\d+(?:[.,]\d+)?)"
)

# Algebraic-leftover guard: drop rows whose narrative still contains
# variable-assignment patterns like "x = 12" alongside tool calls. Goal:
# training data where every numerical step is a tool call and any
# post-tool text is interpretive prose, not symbolic algebra.
ALGEBRAIC_LEFTOVER = re.compile(r"(?<=[ \t])[a-z]\s*=\s*-?\d")


def norm_num(s: str) -> str:
    """Normalize operator/decimal variants so sympy can parse."""
    return s.replace(",", ".").replace("×", "*").replace(" ", "")


def row_has_bad_arithmetic(text: str) -> bool:
    """True if any 2+-operand expression has a stated result that fails
    sympy verification. Catches Libby-class errors where Gemini wrote
    correct intermediates but a wrong final aggregation.
    """
    for m in ROW_FINAL_CHECK.finditer(text):
        if not verify_arith(norm_num(m.group(1)), norm_num(m.group(2))):
            return True
    return False


def convert_orca_answer(text: str) -> list[dict] | None:
    """Convert orca-math prose into a multi-turn funcall sequence.

    Filters:
      - row is dropped if any multi-operand claim fails sympy verification
      - row is dropped if narrative contains algebraic ``x = N`` leftovers
      - per-match verification: bad individual expressions are skipped,
        not row-dropped

    Returns ``[{role, content}]`` or ``None`` if no usable arithmetic.
    """
    if row_has_bad_arithmetic(text):
        return None
    if ALGEBRAIC_LEFTOVER.search(text):
        return None

    parts: list[tuple[str, str]] = []
    cursor = 0
    found = 0
    prev_result: str | None = None

    for m in ORCA_CALC.finditer(text):
        raw_expr, c = m.group(1), m.group(2)
        expr = norm_num(raw_expr)
        result_norm = norm_num(c)
        if not verify_arith(expr, result_norm):
            continue
        prefix = text[cursor : m.start()]
        if prefix.strip():
            parts.append(("assistant", prefix.strip()))
        if (
            BRIDGE_ON_REUSE
            and prev_result is not None
            and _starts_with_value(expr, prev_result)
            and prev_result not in (prefix or "")
        ):
            parts.append(("assistant", f"nun ni havas {prev_result} , do"))
        parts.append(("assistant", f"{TC_OPEN}{expr}{TC_CLOSE}"))
        parts.append(("tool", f"{TR_OPEN}{result_norm}{TR_CLOSE}"))
        prev_result = result_norm
        cursor = m.end()
        found += 1

    if not found:
        return None

    tail = text[cursor:].strip()
    if tail:
        parts.append(("assistant", tail))

    coalesced: list[list] = []
    for role, content in parts:
        if coalesced and coalesced[-1][0] == role:
            coalesced[-1][1] = coalesced[-1][1] + " " + content
        else:
            coalesced.append([role, content])
    return [{"role": r, "content": c} for r, c in coalesced]
