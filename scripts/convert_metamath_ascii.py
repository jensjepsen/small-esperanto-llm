"""Produce an ASCII-math variant of a MetaMath-EO JSONL and (optionally) push
to HF as a new config on the existing repo.

Difference from the source: LaTeX math (``\\frac``, ``\\sqrt``, ``\\cdot``,
``\\boxed``, ``\\dbinom`` etc.) and Unicode math (``≠``, ``≥``, ``∞`` etc.)
are replaced with ASCII equivalents. Currency (``$5``, ``$1,234.56``,
``£20``, ``€500``) is preserved — the converter distinguishes bare
currency from LaTeX inline math by inspecting the ``$...$`` content.

Fields ``q_eo``, ``a_eo``, ``q_en``, ``a_en`` are all converted; other
fields (``orig_idx``, ``type``, tokens…) pass through unchanged.

Example:
  uv run python scripts/convert_metamath_ascii.py \\
    --input /mnt/data2/metamath_gsm_eo.jsonl \\
    --output /mnt/data2/metamath_gsm_eo_ascii.jsonl \\
    --push jensjepsen/esperanto-metamath-gsm
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ── LaTeX → ASCII math ──────────────────────────────────────────────────

# Recursive brace-matching helper: run repeatedly until no more replacements
# so nested `\boxed{\sqrt{5}}` unwraps in the correct order.
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_OVERLINE = re.compile(r"\\overline\{([^{}]*)\}")
_DBINOM = re.compile(r"\\d?binom\{([^{}]*)\}\{([^{}]*)\}")
_FRAC = re.compile(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}")
_SQRT = re.compile(r"\\sqrt\{([^{}]*)\}")
_MATHBB = re.compile(r"\\mathbb\{([^{}]*)\}")
_TEXT = re.compile(r"\\text\{([^{}]*)\}")

# Bare macros — replaced by simple token.
BARE_MACROS = {
    r"\cdot": " * ",
    r"\times": " * ",
    r"\div": " / ",
    r"\neq": " != ",
    r"\ne": " != ",
    r"\ge": " >= ",
    r"\le": " <= ",
    r"\geq": " >= ",
    r"\leq": " <= ",
    r"\infty": "inf",
    r"\pi": "pi",
    r"\pm": "+-",
    r"\to": " -> ",
    r"\in": " in ",
    r"\%": "%",
    r"\$": "$",
    r"\{": "{",
    r"\}": "}",
    # Kept-Latin nice-form Greek — turn into ASCII names to keep model-friendly
    r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma",
    r"\delta": "delta", r"\theta": "theta", r"\lambda": "lambda",
    r"\sigma": "sigma", r"\phi": "phi",
}

UNICODE_MATH = {
    "×": "*", "÷": "/", "·": "*",
    "≠": "!=", "≥": ">=", "≤": "<=",
    "∞": "inf", "π": "pi", "±": "+-", "→": "->",
    "√": "sqrt", "…": "...",
    "ℝ": "R", "ℕ": "N", "ℤ": "Z", "ℚ": "Q", "ℂ": "C",
    "∈": " in ", "∉": " notin ",
    "²": "^2", "³": "^3",
}

# Inline-math delimiter unwrap. Only fires when the content is
# unambiguously math: starts with a letter/backslash/paren, or contains a
# LaTeX macro or `^`/`_` subscript. `=` alone is NOT enough because
# MetaMath GSM writes currency arithmetic like `$6 = $102` — that pattern
# has to survive.
_LOOKS_LIKE_MATH = re.compile(
    r"^\s*[a-zA-Z(\\]"   # var / macro / paren opening
    r"|\\[a-zA-Z]+"       # LaTeX macro anywhere
    r"|[\^_]"             # sub/superscript
)
_INLINE_MATH = re.compile(r"\$([^$\n]{1,300}?)\$")


def _inline_math_sub(text: str) -> str:
    """Unwrap `$...$` only when content is unambiguously math."""
    def _sub(m: re.Match) -> str:
        content = m.group(1)
        if _LOOKS_LIKE_MATH.search(content):
            return content
        return m.group(0)  # currency
    return _INLINE_MATH.sub(_sub, text)


def latex_to_ascii(text: str) -> str:
    if not text:
        return text

    # 1) Unwrap inline `$...$` math FIRST — otherwise `$\boxed{40}$` becomes
    # `$40$` after step 2 and never loses the currency-like wrapper.
    text = _inline_math_sub(text)

    # 2) Named LaTeX macros. Loop for nested braces.
    prev = None
    while prev != text:
        prev = text
        text = _BOXED.sub(r"\1", text)
        text = _OVERLINE.sub(r"\1", text)  # drop the overline; the digits are already there
        text = _DBINOM.sub(r"C(\1,\2)", text)
        text = _FRAC.sub(r"(\1)/(\2)", text)
        text = _SQRT.sub(r"sqrt(\1)", text)
        text = _MATHBB.sub(r"\1", text)
        text = _TEXT.sub(r"\1", text)

    # 3) Bare macros
    for k, v in BARE_MACROS.items():
        text = text.replace(k, v)

    # 4) Unicode math
    for k, v in UNICODE_MATH.items():
        text = text.replace(k, v)

    # 5) LaTeX display math delimiters
    text = re.sub(r"\\\[|\\\]|\\\(|\\\)", " ", text)

    # 6) Collapse runs of whitespace (but keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


# ── Conversion driver ──────────────────────────────────────────────────


CONVERT_FIELDS = ("q_eo", "a_eo", "q_en", "a_en")


def convert_row(r: dict) -> dict:
    out = dict(r)
    for k in CONVERT_FIELDS:
        v = out.get(k)
        if isinstance(v, str):
            out[k] = latex_to_ascii(v)
    return out


def convert_file(src: Path, dst: Path) -> tuple[int, int]:
    n = n_changed = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            new = convert_row(r)
            fout.write(json.dumps(new, ensure_ascii=False) + "\n")
            if any(new.get(k) != r.get(k) for k in CONVERT_FIELDS):
                n_changed += 1
    return n, n_changed


# ── HF push ─────────────────────────────────────────────────────────────


def push_to_hf(jsonl_path: Path, repo: str, private: bool = False) -> None:
    from datasets import Dataset, DatasetDict

    token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        tp = Path.home() / ".cache/huggingface/token"
        if tp.exists():
            token = tp.read_text().strip()
    if not token:
        raise SystemExit("No HF token found.")

    rows: list[dict] = []
    sft_rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
            q = (r.get("q_eo") or "").strip()
            a = (r.get("a_eo") or "").strip()
            if q and a:
                sft_rows.append({
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    "orig_idx": r["orig_idx"],
                    "type": r.get("type", ""),
                })

    default_dd = DatasetDict({"train": Dataset.from_list(rows)})
    sft_dd = DatasetDict({"train": Dataset.from_list(sft_rows)})

    print(f"pushing default-ascii config ({len(rows):,}) → {repo}…", flush=True)
    default_dd.push_to_hub(repo, config_name="default-ascii",
                           token=token, private=private)
    print(f"pushing sft-ascii config ({len(sft_rows):,}) → {repo}…", flush=True)
    sft_dd.push_to_hub(repo, config_name="sft-ascii",
                       token=token, private=private)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/mnt/data2/metamath_gsm_eo.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("/mnt/data2/metamath_gsm_eo_ascii.jsonl"))
    ap.add_argument("--push", type=str, default=None,
                    help="HF repo id to push to (adds default-ascii + sft-ascii configs)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"converting {args.input} → {args.output}…", flush=True)
    n, n_changed = convert_file(args.input, args.output)
    print(f"  wrote {n:,} rows; {n_changed:,} changed by conversion "
          f"({100 * n_changed / max(n, 1):.1f}%)")

    if args.push:
        push_to_hf(args.output, args.push, private=args.private)
        print(f"done → https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
