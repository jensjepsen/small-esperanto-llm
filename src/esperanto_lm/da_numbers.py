"""Danish number-words helper. Same interface as eo_numbers.

Danish quirks handled:
  - vigesimal base for 50/60/70/80/90 (halvtreds/tres/halvfjerds/firs/halvfems)
  - unit-before-tens ordering (21 = "enogtyve", not "tyveogen")
  - "en"/"et" gender (default "et" for math contexts)

Public API mirrors eo_numbers:
    num_to_da(n)
    rational_to_da(num, denom)
    wordify_text(text, rng, p_word)
"""
import re

# "en" (fælleskøn) is the cardinal form used in counting and in compound
# numbers like "enogtyve" (21). "Et" (intetkøn) only appears when 1 modifies
# a neuter noun, e.g. "et hundrede" / "et tusind" — handled specially below.
_UNITS = ["nul", "en", "to", "tre", "fire", "fem", "seks", "syv", "otte", "ni"]
_TEENS = ["ti", "elleve", "tolv", "tretten", "fjorten", "femten",
          "seksten", "sytten", "atten", "nitten"]
_TENS = {
    20: "tyve", 30: "tredive", 40: "fyrre", 50: "halvtreds",
    60: "tres", 70: "halvfjerds", 80: "firs", 90: "halvfems",
}


def _under_100(n: int) -> str:
    """0 ≤ n < 100."""
    if n < 10:
        return _UNITS[n]
    if n < 20:
        return _TEENS[n - 10]
    t, u = divmod(n, 10)
    t_word = _TENS[t * 10]
    if u == 0:
        return t_word
    # 21 = "enogtyve", 45 = "femogfyrre" — units-before-tens joined by "og"
    return f"{_UNITS[u]}og{t_word}"


def _under_1000(n: int) -> str:
    """0 ≤ n < 1000."""
    if n < 100:
        return _under_100(n)
    h, r = divmod(n, 100)
    h_word = "et hundrede" if h == 1 else f"{_UNITS[h]} hundrede"
    if r == 0:
        return h_word
    return f"{h_word} og {_under_100(r)}"


def num_to_da(n: int) -> str:
    """Integer (negative ok) to Danish words."""
    if n < 0:
        return "minus " + num_to_da(-n)
    if n < 1000:
        return _under_1000(n)
    if n < 1_000_000:
        thousands, r = divmod(n, 1000)
        t_word = "et tusind" if thousands == 1 else f"{_under_1000(thousands)} tusind"
        if r == 0:
            return t_word
        return f"{t_word} og {_under_1000(r)}"
    # Million+ — keep digits (rare in our training range)
    return str(n)


# Ordinal roots used for fraction names (X-del/dele).
# 1/2 is special: "halv" / "halve". Others use ordinal + "del".
_ORDINAL_ROOT = {
    3: "tredje", 4: "fjerde", 5: "femte", 6: "sjette", 7: "syvende",
    8: "ottende", 9: "niende", 10: "tiende",
}


def _fraction_name(denom: int, plural: bool) -> str:
    """Denominator name, e.g. 3 → 'tredjedel' / 'tredjedele'."""
    if denom == 2:
        return "halve" if plural else "halv"
    if denom in _ORDINAL_ROOT:
        base = _ORDINAL_ROOT[denom] + "del"
    else:
        # 11+ — use "N-del" with cardinal form (colloquial but readable)
        base = _under_1000(denom).replace(" ", "") + "del"
    return base + "e" if plural else base


def rational_to_da(num: int, denom: int) -> str:
    """Form p/q as Danish words.

    Examples:
      1/2  -> "en halv"
      2/3  -> "to tredjedele"
      3/4  -> "tre fjerdedele"
      -5/4 -> "minus fem fjerdedele"
    """
    if denom == 0:
        return f"{num}/{denom}"
    if denom < 0:
        num, denom = -num, -denom
    sign = ""
    if num < 0:
        sign = "minus "
        num = -num
    if denom == 1:
        return sign + num_to_da(num)
    if denom >= 1000:
        return f"{sign}{num}/{denom}"

    num_word = "en" if num == 1 else num_to_da(num)
    frac_word = _fraction_name(denom, plural=(num != 1))
    return f"{sign}{num_word} {frac_word}"


# Match standalone integers and X/Y fractions. Skip when adjacent to a
# letter (variable coefficient like `2x`) or `#` (answer marker `#### N`).
# Trailing `(?![A-Za-z_0-9])` also rejects digits so regex can't backtrack
# `10a` to match just `1` and leave `0a` behind.
_NUM_PAT = re.compile(r"(?<![A-Za-z_#0-9])(-?\d+(?:/\d+)?)(?![A-Za-z_0-9/])")

_OP_PAT = re.compile(r" ([+\-*/]) ")
_OP_WORDS = {
    "+": "plus",
    "-": "minus",
    "*": "gange",
    "/": "divideret med",
}


def wordify_text(text: str, rng, p_word: float = 0.15) -> str:
    """Probabilistically replace numbers AND binary operators with Danish
    word form. Same behavior as eo_numbers.wordify_text."""
    if p_word <= 0:
        return text

    def _num_repl(m):
        if rng.random() >= p_word:
            return m.group(0)
        s = m.group(1)
        if "/" in s:
            num_s, denom_s = s.split("/", 1)
            return rational_to_da(int(num_s), int(denom_s))
        return num_to_da(int(s))

    def _op_repl(m):
        if rng.random() >= p_word:
            return m.group(0)
        return " " + _OP_WORDS[m.group(1)] + " "

    text = _NUM_PAT.sub(_num_repl, text)
    text = _OP_PAT.sub(_op_repl, text)
    return text
