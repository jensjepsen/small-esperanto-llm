"""EN→EO translation with LaTeX + currency protection.

Wraps the base MarianMT translation loop with a two-stage sentinel
protection pass:

    (1) Currency amounts (``$25``, ``$1,234.56``) → ``<extra_N>``
    (2) LaTeX inline/display math (via pylatexenc AST) → ``<extra_N>``

Sentinels use the ``<extra_N>`` form because our SPM tokenizer preserves
angle-bracket tokens round-trip while dropping most Unicode private-use
characters to ``<unk>``. After translation, sentinels are restored in
reverse order to preserve nested/adjacent match positions.

Bonus observed on our current v9-mt: removing LaTeX clutter from the
translated prose also improves math-domain vocabulary disambiguation
("complex plane" → ``ebeno`` (math plane) instead of ``aviadilo``
(airplane)) because the model gets cleaner prose context.

CLI usage
---------

Single-string:

    uv run python mt/scripts/translate_with_latex.py \\
        --checkpoint jensjepsen/eo-mt-v9 \\
        --text 'Let $f(x) = x^2$ be a polynomial.'

JSONL file (translates each row's ``en`` field, writes ``eo_v9`` field):

    uv run python mt/scripts/translate_with_latex.py \\
        --checkpoint jensjepsen/eo-mt-v9 \\
        --in /path/to/input.jsonl --out /path/to/output.jsonl

Programmatic use::

    from mt.scripts.translate_with_latex import LatexAwareTranslator
    tr = LatexAwareTranslator('jensjepsen/eo-mt-v9')
    eo = tr.translate('Let $f(x) = x^2$ be a polynomial.')
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from pylatexenc.latexwalker import LatexMathNode, LatexWalker
from transformers import MarianMTModel

sys.path.insert(0, str(Path(__file__).parent))
from sp_tokenizer import SPMTokenizer

# Currency amounts in USD ($), GBP (£), EUR (€): 25, 1,234.56.
# European conventions (1.234,56) rare in orca-math; extend if needed.
_CURRENCY = re.compile(r"[$£€]\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")

# Unicode math → ASCII normalization. Our SPM tokenizer OOVs these to `<unk>`
# during encoding, destroying content like `9 × 4` or `36 cm²`. Normalize
# to ASCII-safe equivalents that survive round-trip cleanly.
_UNICODE_MATH_NORMALIZE = {
    # Operators
    "×": "*",   "÷": "/",  "−": "-",  "·": "*",  "∗": "*",
    # Comparators
    "≤": "<=",  "≥": ">=", "≠": "!=", "≈": "~",  "≡": "==",
    # Superscript digits → ^N (e.g. cm² → cm^2)
    "⁰": "^0",  "¹": "^1",  "²": "^2",  "³": "^3",  "⁴": "^4",
    "⁵": "^5",  "⁶": "^6",  "⁷": "^7",  "⁸": "^8",  "⁹": "^9",
    "⁻": "^-",  "⁺": "^+",
    # Subscript digits → _N
    "₀": "_0",  "₁": "_1",  "₂": "_2",  "₃": "_3",  "₄": "_4",
    "₅": "_5",  "₆": "_6",  "₇": "_7",  "₈": "_8",  "₉": "_9",
    # Common math sets (kept as English words; MT model knows R, Z, N, Q, C)
    "ℝ": "R",   "ℤ": "Z",   "ℕ": "N",   "ℚ": "Q",   "ℂ": "C",
    # Special
    "∅": "{}",  "∞": "inf", "√": "sqrt", "∂": "d",
    "∈": " in ", "∉": " notin ", "⊂": " subset ", "⊆": " subseteq ",
    "∀": "forall ", "∃": "exists ",
    "→": "->", "←": "<-", "↔": "<->", "⇒": "=>", "⇐": "<=",
    # Greek letters — spell out for common usage
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "θ": "theta", "λ": "lambda", "μ": "mu",
    "π": "pi", "σ": "sigma", "τ": "tau", "φ": "phi", "ω": "omega",
    "Δ": "Delta", "Σ": "Sigma", "Π": "Pi", "Ω": "Omega",
    # Fractions
    "½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4",
    "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
    # Degree
    "°": " deg",
    # More math ops surfaced in orca-math scan.
    # Use " +- " (space-surrounded, no slash) for ± since "+/-" gets the "+"
    # dropped by the tokenizer and the "/" reinterpreted as division.
    "±": " +- ", "∛": "cbrt", "∪": " union ", "∩": " intersect ",
    # Unicode squared-unit CJK-compatibility block (frequent in orca-math)
    "㎖": " ml", "㎗": " dl", "㎘": " kL", "㎜": " mm",
    "㎝": " cm", "㎞": " km", "㎟": " mm^2", "㎠": " cm^2",
    "㎡": " m^2", "㎢": " km^2", "㎣": " mm^3", "㎤": " cm^3",
    "㎥": " m^3", "㎦": " km^3", "㎏": " kg", "㎎": " mg",
    "㎍": " ug", "㎕": " uL", "㎫": " MPa", "㎩": " Pa",
    "㎾": " kW", "㎿": " MW", "㎐": " Hz", "㎑": " kHz",
    "㎒": " MHz", "㎓": " GHz", "㎔": " THz",
    # Curly quotes + dashes → ASCII
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
}


def normalize_unicode_math(text: str) -> str:
    """Replace Unicode math characters with ASCII equivalents. Preserves
    prose Unicode (accented Latin, quotes, dashes) — only substitutes
    known math glyphs."""
    for uni, ascii_ in _UNICODE_MATH_NORMALIZE.items():
        text = text.replace(uni, ascii_)
    return text


class LatexAwareTranslator:
    """v9-mt EN→EO translator with LaTeX/currency preservation."""

    def __init__(
        self,
        checkpoint: str = "jensjepsen/eo-mt-v9",
        tokenizer_path: str = "mt/data/tokenizer/spm_eneo_32k.model",
        device: str = "cuda",
        max_input_tokens: int = 500,
        max_output_tokens: int = 256,
    ):
        self.tok = SPMTokenizer(tokenizer_path)
        self.model = MarianMTModel.from_pretrained(checkpoint).to(device).eval()
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

    # ── Protection ─────────────────────────────────────────────────────

    @staticmethod
    def _protect(text: str) -> tuple[str, list[tuple[str, str]]]:
        """Replace currency + LaTeX math with ``<extra_N>`` sentinels.

        Returns ``(protected_text, mapping)`` where ``mapping[i]`` is
        ``(kind, original)`` for sentinel ``<extra_i>``. ``kind`` is
        ``'curr'`` for currency, ``'math'`` for LaTeX math nodes.
        """
        mapping: list[tuple[str, str]] = []

        def _curr(m: re.Match) -> str:
            idx = len(mapping)
            mapping.append(("curr", m.group(0)))
            return f"<extra_{idx}>"

        # Stage 1: currency first (regex is unambiguous for well-formed amounts)
        text = _CURRENCY.sub(_curr, text)

        # Stage 2: LaTeX math via AST parse
        try:
            nodes, _, _ = LatexWalker(text).get_latex_nodes()
        except Exception:
            return text, mapping

        # Walk in reverse so we don't invalidate earlier positions
        math_nodes = [
            n for n in nodes
            if isinstance(n, LatexMathNode)
        ]
        for n in reversed(math_nodes):
            orig = text[n.pos:n.pos + n.len]
            idx = len(mapping)
            mapping.append(("math", orig))
            text = text[:n.pos] + f" <extra_{idx}> " + text[n.pos + n.len:]

        return text, mapping

    @staticmethod
    def _restore(text: str, mapping: list[tuple[str, str]]) -> str:
        """Replace ``<extra_N>`` sentinels with originals.

        Restores in reverse order so a longer sentinel like ``<extra_10>``
        doesn't get partially matched by ``<extra_1>``.
        """
        for i in range(len(mapping) - 1, -1, -1):
            _, orig = mapping[i]
            text = text.replace(f"<extra_{i}>", orig)
        return text

    # ── Model call ─────────────────────────────────────────────────────

    def _generate(self, src: str) -> str:
        ids = self.tok.encode(src, lang="eo")
        if len(ids) > self.max_input_tokens:
            ids = ids[: self.max_input_tokens]
        be = self.tok.pad_batch([ids])
        with torch.no_grad():
            out = self.model.generate(
                bad_words_ids=[[1]],  # suppress <unk> — verified to fix
                                      # ~40% of the orca-math UNK cases by
                                      # picking the 2nd-most-likely token
                                      # (usually the correct math operator)
                input_ids=be.input_ids.to(self.device),
                attention_mask=be.attention_mask.to(self.device),
                max_length=self.max_output_tokens,
                do_sample=False,
                num_beams=1,
            )
        return self.tok.decode(out[0])

    # ── Public API ─────────────────────────────────────────────────────

    def translate(self, src: str) -> str:
        """Translate one EN string. Normalizes Unicode math to ASCII,
        then protects LaTeX + currency with sentinels."""
        src = normalize_unicode_math(src)
        protected, mapping = self._protect(src)
        pred = self._generate(protected)
        return self._restore(pred, mapping)

    def translate_batch(self, srcs: list[str]) -> list[str]:
        """Translate a batch by protecting each, running one batched
        generate, then restoring per-item."""
        srcs = [normalize_unicode_math(s) for s in srcs]
        prot_and_map = [self._protect(s) for s in srcs]
        prots = [p for p, _ in prot_and_map]
        ids_list = [self.tok.encode(p, lang="eo")[: self.max_input_tokens]
                    for p in prots]
        be = self.tok.pad_batch(ids_list)
        with torch.no_grad():
            out = self.model.generate(
                bad_words_ids=[[1]],  # suppress <unk>
                input_ids=be.input_ids.to(self.device),
                attention_mask=be.attention_mask.to(self.device),
                max_length=self.max_output_tokens,
                do_sample=False,
                num_beams=1,
            )
        return [
            self._restore(self.tok.decode(out[i]), prot_and_map[i][1])
            for i in range(len(srcs))
        ]


# ── Smoke test ─────────────────────────────────────────────────────────

_SMOKE_PROBES = [
    ("unicode math ops",
     "The volume is 9 cm × 4 cm × 7 cm = 252 cm³. And 45 ÷ 5 = 9."),
    ("unicode inequalities",
     "For all x ≤ 5, we have x² ≤ 25 and x² ≥ 0."),
    ("greek + set membership",
     "For all α, β ∈ ℝ, we have α + β ∈ ℝ."),
    ("gbp + eur currency",
     "The book costs £15 in London and €18 in Paris."),
    ("squared units",
     "The tank holds 500 ㎖ and the room is 20 ㎡."),
    ("curly quotes + dashes",
     "The result of Jane’s calculation was “42” — surprising."),
    ("cube root + plus-minus",
     "The solution is ∛8 ± 1 = 2 ± 1."),
    ("currency+text",
     "Each player requires a $25 jersey and 3 balls costing $47 total."),
    ("currency w/ commas",
     "The total cost was $1,234.56 for all items."),
    ("currency + LaTeX",
     "Buy $25 books and let $f(x) = x^2$ be a polynomial."),
    ("complex plane math",
     "The distance between two points $(x_1,y_1)$ and $(x_2,y_2)$ "
     "in the complex plane is $\\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$."),
    ("matrix invertible",
     "A matrix $M \\in \\mathbb{R}^{n \\times n}$ is invertible iff "
     "$\\det(M) \\neq 0$."),
    ("greek letters + currency",
     "For $100 you can buy a book about $\\pi$ and $e$."),
    ("trig",
     "The derivative of $\\sin(x)$ is $\\cos(x)$."),
    ("plain prose (no math)",
     "The Prime Minister announced sweeping reforms to the healthcare system."),
]


def _smoke(translator: LatexAwareTranslator) -> None:
    print("=== translate_with_latex smoke test ===\n")
    for label, src in _SMOKE_PROBES:
        eo = translator.translate(src)
        print(f"[{label}]")
        print(f"  EN: {src}")
        print(f"  EO: {eo}\n")


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="jensjepsen/eo-mt-v9")
    ap.add_argument("--tokenizer",
                    default="mt/data/tokenizer/spm_eneo_32k.model")
    ap.add_argument("--text", help="Translate this single string and print")
    ap.add_argument("--in", dest="in_path", type=Path,
                    help="Input JSONL with 'en' field per row")
    ap.add_argument("--out", type=Path,
                    help="Output JSONL (adds 'eo_v9' field to each row)")
    ap.add_argument("--in-field", default="en")
    ap.add_argument("--out-field", default="eo_v9")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--smoke", action="store_true",
                    help="Run built-in smoke test on 8 probes")
    args = ap.parse_args()

    tr = LatexAwareTranslator(args.checkpoint, tokenizer_path=args.tokenizer)

    if args.smoke:
        _smoke(tr)
        return

    if args.text:
        print(tr.translate(args.text))
        return

    if not (args.in_path and args.out):
        ap.error("provide --smoke, --text, or --in/--out")

    n = 0
    with args.in_path.open() as fin, args.out.open("w") as fout:
        batch_rows: list[dict] = []
        batch_srcs: list[str] = []
        for line in fin:
            r = json.loads(line)
            batch_rows.append(r)
            batch_srcs.append(r[args.in_field])
            if len(batch_srcs) >= args.batch_size:
                for row, eo in zip(batch_rows, tr.translate_batch(batch_srcs)):
                    row[args.out_field] = eo
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += len(batch_srcs)
                batch_rows, batch_srcs = [], []
                print(f"  {n} rows translated", flush=True)
        if batch_srcs:
            for row, eo in zip(batch_rows, tr.translate_batch(batch_srcs)):
                row[args.out_field] = eo
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += len(batch_srcs)
    print(f"done: {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
