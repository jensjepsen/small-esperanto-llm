"""Esperanto number-words helper.

Used by the algebra/arith generators to interleave digit form and word
form within training rows so the model learns that "dudek tri" and "23"
are equivalent.

Public API:
    num_to_eo(n)              integer → words. Signed.
    rational_to_eo(num, denom) p/q → "X Y-onoj" form. Signed.
    wordify_text(text, rng, p_word)
                              probabilistically replace standalone
                              numbers (integers or X/Y fractions) in
                              `text` with their Esperanto word form.
                              Skips numbers followed by a letter (so
                              variable coefficients like `2x` stay
                              digits, matching the chain shape).
"""
import re

_UNITS = ["", "unu", "du", "tri", "kvar", "kvin", "ses", "sep", "ok", "naŭ"]
_TENS_PREFIX = ["", "", "du", "tri", "kvar", "kvin", "ses", "sep", "ok", "naŭ"]
# 20 = dudek, 30 = tridek, ... constructed by concatenating prefix + "dek".


def _under_100(n: int) -> str:
    """0 ≤ n < 100."""
    if n == 0:
        return "nul"
    if n < 10:
        return _UNITS[n]
    if n == 10:
        return "dek"
    if n < 20:
        return "dek " + _UNITS[n - 10]
    t, u = divmod(n, 10)
    word = _TENS_PREFIX[t] + "dek"
    if u == 0:
        return word
    return word + " " + _UNITS[u]


def _under_1000(n: int) -> str:
    """0 ≤ n < 1000."""
    if n < 100:
        return _under_100(n)
    h, r = divmod(n, 100)
    h_word = "cent" if h == 1 else _UNITS[h] + "cent"
    if r == 0:
        return h_word
    return h_word + " " + _under_100(r)


def num_to_eo(n: int) -> str:
    """Integer (negative ok) to Esperanto words."""
    if n < 0:
        return "minus " + num_to_eo(-n)
    if n < 1000:
        return _under_1000(n)
    if n < 1_000_000:
        thousands, r = divmod(n, 1000)
        t_word = "mil" if thousands == 1 else _under_1000(thousands) + " mil"
        if r == 0:
            return t_word
        return t_word + " " + _under_1000(r)
    # Million+ — keep digits, not common in our training range.
    return str(n)


def rational_to_eo(num: int, denom: int) -> str:
    """Form p/q as Esperanto words.

    Examples:
      1/2  -> "duono"
      2/3  -> "du trionoj"
      7/8  -> "sep okonoj"
      91/3 -> "naŭdek unu trionoj"
      -5/4 -> "minus kvin kvaronoj"

    For denominators ≥ 1000 (rare in our data), falls back to digit form.
    """
    if denom == 0:
        return f"{num}/{denom}"  # nonsense input — pass through
    if denom < 0:
        num, denom = -num, -denom
    sign = ""
    if num < 0:
        sign = "minus "
        num = -num
    if denom >= 1000:
        return f"{sign}{num}/{denom}"  # too unwieldy as words

    # Denominator name: cardinal_root + "ono" (with plural -j when num != 1).
    if denom == 1:
        # Not really a fraction, but for robustness.
        return sign + num_to_eo(num)
    denom_root = _under_1000(denom)
    # Esperanto fraction-noun: replace trailing "o" if root ends in vowel
    # is NOT how it works; we just append "-on-" suffix directly.
    # Most denoms work as "<root>ono" by stripping trailing space and
    # appending "ono", but cleaner is to append "ono" to the last
    # numeric-word in the phrase (so "dudek tri" + "ono" -> "dudek triono").
    parts = denom_root.split()
    parts[-1] = parts[-1] + "ono"
    base = " ".join(parts)
    if num == 1:
        return sign + base
    # plural -j on the fraction-noun
    plural = base + "j"
    num_word = num_to_eo(num)
    return f"{sign}{num_word} {plural}"


# Match standalone integers and X/Y fractions. Skip when adjacent to a
# letter (variable coefficient like `2x`) or `#` (answer marker `#### N`).
_NUM_PAT = re.compile(r"(?<![A-Za-z_#])(-?\d+(?:/\d+)?)(?![A-Za-z_])")

# Binary operators with whitespace around them. Unary `-` (negative
# prefix of a number) is part of _NUM_PAT and handled by num_to_eo's
# "minus N" prefix, so we don't catch it here.
_OP_PAT = re.compile(r" ([+\-*/]) ")
_OP_WORDS = {
    "+": "plus",
    "-": "minus",
    "*": "fojoj",
    "/": "dividite per",
}


def wordify_text(text: str, rng, p_word: float = 0.15) -> str:
    """Walk through `text` and probabilistically replace numbers AND
    binary operators with their Esperanto word form. Each occurrence
    is an independent coin-flip with probability `p_word`. Skips:
      - numbers next to a letter (variable coefs: `2x`, `7y`)
      - numbers next to `#` (the `#### N` answer marker)
      - unary `-` (already part of the number; num_to_eo prefixes "minus")
    The per-occurrence decision is what produces intra-chain variation:
    same number can appear as digits one line and as words another.
    """
    if p_word <= 0:
        return text

    def num_repl(m):
        if rng.random() >= p_word:
            return m.group(0)
        s = m.group(0)
        if "/" in s:
            num_s, denom_s = s.split("/", 1)
            try:
                return rational_to_eo(int(num_s), int(denom_s))
            except ValueError:
                return s
        try:
            return num_to_eo(int(s))
        except ValueError:
            return s

    def op_repl(m):
        if rng.random() >= p_word:
            return m.group(0)
        op = m.group(1)
        return " " + _OP_WORDS[op] + " "

    text = _NUM_PAT.sub(num_repl, text)
    text = _OP_PAT.sub(op_repl, text)
    return text
