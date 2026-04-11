"""Generate arithmetic chain-of-thought SFT dataset.

Converts digit-by-digit decomposed arithmetic into Esperanto
conversation format for SFT training.
"""

import argparse
import json
import math
import random
from pathlib import Path

from rich.console import Console

console = Console()

DEFAULT_OUTPUT_DIR = Path("data/sft/arithmetic_cot")


def _num_to_eo(n: int) -> str:
    """Convert a number to its Esperanto word form."""
    if n == 0:
        return "nul"
    ones = ["", "unu", "du", "tri", "kvar", "kvin", "ses", "sep", "ok", "naŭ"]
    if n < 10:
        return ones[n]
    if n < 20:
        return f"dek {ones[n - 10]}".strip()
    if n < 100:
        tens = n // 10
        rest = n % 10
        prefix = f"{ones[tens]}dek" if tens > 1 else "dek"
        return f"{prefix} {ones[rest]}".strip()
    if n < 1000:
        hundreds = n // 100
        rest = n % 100
        prefix = f"{ones[hundreds]}cent" if hundreds > 1 else "cent"
        if rest == 0:
            return prefix
        return f"{prefix} {_num_to_eo(rest)}"
    return str(n)


def _maybe_word(n: int) -> str:
    """Randomly return number as digit or Esperanto word (for numbers < 1000)."""
    if n < 1000 and random.random() < 0.2:
        return _num_to_eo(n)
    return str(n)


def decompose_add(a: int, b: int) -> tuple[str, list[str], int]:
    da = [int(d) for d in str(a)][::-1]
    db = [int(d) for d in str(b)][::-1]
    n = max(len(da), len(db))

    carry = 0
    steps = []
    result_digits = []

    for i in range(n):
        ai = da[i] if i < len(da) else None
        bi = db[i] if i < len(db) else None

        if ai is not None and bi is not None:
            s = ai + bi + carry
            if carry:
                step = f"{ai}+{bi}+{carry}={s}"
            else:
                step = f"{ai}+{bi}={s}"
        elif ai is not None:
            s = ai + carry
            step = f"{ai}+{carry}={s}" if carry else f"{ai}={s}"
        else:
            s = bi + carry
            step = f"{bi}+{carry}={s}" if carry else f"{bi}={s}"

        new_carry = s // 10
        digit = s % 10
        result_digits.append(digit)

        step += f"c{new_carry}"
        steps.append(step)
        carry = new_carry

    if carry:
        result_digits.append(carry)
        steps.append(f"c{carry}→{carry}")

    result = int(''.join(str(d) for d in reversed(result_digits)))
    return f"{a}+{b}", steps, result


def decompose_sub(a: int, b: int) -> tuple[str, list[str], int]:
    if a < b:
        a, b = b, a
    da = [int(d) for d in str(a)][::-1]
    db = [int(d) for d in str(b)][::-1]
    n = len(da)

    borrow = 0
    steps = []
    result_digits = []

    for i in range(n):
        ai = da[i]
        bi = db[i] if i < len(db) else None
        top = ai - borrow

        if bi is not None:
            if top < bi:
                diff = (top + 10) - bi
                if borrow:
                    step = f"{ai}-{borrow}-{bi}+10={diff}"
                else:
                    step = f"{ai}-{bi}+10={diff}"
                new_borrow = 1
            else:
                diff = top - bi
                if borrow:
                    step = f"{ai}-{borrow}-{bi}={diff}"
                else:
                    step = f"{ai}-{bi}={diff}"
                new_borrow = 0
        else:
            # No more b digits - just propagate the borrow
            if borrow:
                if ai < borrow:
                    diff = (ai + 10) - borrow
                    step = f"{ai}-{borrow}+10={diff}"
                    new_borrow = 1
                else:
                    diff = ai - borrow
                    step = f"{ai}-{borrow}={diff}"
                    new_borrow = 0
            else:
                diff = ai
                step = f"{ai}={diff}"
                new_borrow = 0

        step += f"b{new_borrow}"
        result_digits.append(diff)
        steps.append(step)
        borrow = new_borrow

    while len(result_digits) > 1 and result_digits[-1] == 0:
        result_digits.pop()

    result = int(''.join(str(d) for d in reversed(result_digits)))
    return f"{a}-{b}", steps, result


def _mul_single(a: int, b: int) -> tuple[list[str], int]:
    """Multiply multi-digit a by single-digit b, return (steps, result)."""
    da = [int(d) for d in str(a)][::-1]

    carry = 0
    steps = []
    result_digits = []

    for i in range(len(da)):
        prod = da[i] * b
        if carry:
            step = f"{da[i]}*{b}+{carry}={prod + carry}"
            prod += carry
        else:
            step = f"{da[i]}*{b}={prod}"

        new_carry = prod // 10
        digit = prod % 10
        result_digits.append(digit)

        step += f"c{new_carry}"
        steps.append(step)
        carry = new_carry

    if carry:
        result_digits.append(carry)
        steps.append(f"c{carry}→{carry}")

    result = int(''.join(str(d) for d in reversed(result_digits)))
    return steps, result


def decompose_mul(a: int, b: int) -> tuple[str, list[str], int]:
    if b < 10:
        steps, result = _mul_single(a, b)
        return f"{a}*{b}", steps, result
    if a < 10:
        steps, result = _mul_single(b, a)
        return f"{a}*{b}", steps, result

    # Multi-digit × multi-digit: partial products
    db = [int(d) for d in str(b)][::-1]
    all_steps = []
    partials = []

    for i, digit in enumerate(db):
        if digit == 0:
            partials.append(0)
            continue
        steps, partial = _mul_single(a, digit)
        shifted = partial * (10 ** i)
        if i == 0:
            steps.append(f"→ {partial}")
        else:
            steps.append(f"→ {partial}{'0' * i}")
        all_steps.extend(steps)
        partials.append(shifted)

    # Sum partials (only if there are 2+ non-zero partials)
    total = sum(partials)
    nonzero = [p for p in partials if p > 0]
    if len(nonzero) > 1:
        sum_expr = "+".join(str(p) for p in nonzero)
        all_steps.append(f"{sum_expr}={total}")

    return f"{a}*{b}", all_steps, total


def decompose_div(a: int, b: int) -> tuple[str, list[str], int]:
    """Long division: a / b where b is single digit, a is multi-digit."""
    da = [int(d) for d in str(a)]
    steps = []
    result_digits = []
    remainder = 0

    for i in range(len(da)):
        current = remainder * 10 + da[i]
        digit = current // b
        remainder = current % b
        is_last = (i == len(da) - 1) and remainder == 0
        if is_last:
            step = f"{current}/{b}={digit}"
        else:
            step = f"{current}/{b}={digit}r{remainder}"
        steps.append(step)
        result_digits.append(digit)

    # Strip leading zeros
    while len(result_digits) > 1 and result_digits[0] == 0:
        result_digits.pop(0)

    result = int(''.join(str(d) for d in result_digits))
    return f"{a}/{b}", steps, result


PERCENT_Q_TEMPLATES = [
    "Kio estas {p}% de {n}?",
    "Kiom estas {p}% de {n}?",
    "Kalkulu {p}% de {n}.",
    "Kiom estas {p} procentoj de {n}?",
    "Trovu {p}% de {n}.",
]


def generate_percent(p: int, n: int) -> tuple[str, str]:
    """Generate a percentage calculation with CoT.

    p% of n = p * n / 100
    Returns (question, answer).
    """
    _, mul_steps, mul_result = decompose_mul(p, n)
    result = mul_result // 100

    mul_str = ", ".join(mul_steps)
    answer = f"{p}% de {n}: {p}*{n}: {mul_str} → {mul_result}. {mul_result}/100={result}. La respondo estas {result}. #### {result}"

    wp = _maybe_word(p)
    wn = _maybe_word(n)
    q = random.choice(PERCENT_Q_TEMPLATES).format(p=wp, n=wn)
    return q, answer


Q_TEMPLATES = {
    "add": [
        "Kalkulu {expr}.",
        "Kio estas {expr}?",
        "Kiom estas {expr}?",
        "Aldonu {a} kaj {b}.",
        "Kio estas {a} plus {b}?",
        "Kiom estas {a} plus {b}?",
    ],
    "sub": [
        "Kalkulu {expr}.",
        "Kio estas {expr}?",
        "Kiom estas {expr}?",
        "Subtrahu {b} de {a}.",
        "Kio estas {a} minus {b}?",
        "Kiom estas {a} minus {b}?",
    ],
    "mul": [
        "Kalkulu {expr}.",
        "Kio estas {expr}?",
        "Kiom estas {expr}?",
        "Multipliku {a} per {b}.",
        "Kio estas {a} oble {b}?",
        "Kiom estas {a} oble {b}?",
    ],
    "div": [
        "Kalkulu {expr}.",
        "Kio estas {expr}?",
        "Kiom estas {expr}?",
        "Dividu {a} per {b}.",
        "Kio estas {a} dividite per {b}?",
        "Kiom estas {a} dividite per {b}?",
    ],
}


def format_answer(expr: str, steps: list[str], result: int) -> str:
    """Format the chain-of-thought answer."""
    step_str = ", ".join(steps)
    return f"{expr}: {step_str}. La respondo estas {result}. #### {result}"


def apply_op(current: int, op: str, operand: int) -> tuple[str, list[str], int]:
    """Apply a single operation to current value, return (expr, steps, result)."""
    if op == "+":
        return decompose_add(current, operand)
    elif op == "-":
        if current < operand:
            current, operand = operand, current
        return decompose_sub(current, operand)
    elif op == "*":
        return decompose_mul(current, operand)
    elif op == "/":
        return decompose_div(current, operand)
    raise ValueError(f"Unknown op: {op}")


def generate_chain(num_ops: int) -> tuple[str, str, str, int, int]:
    """Generate a multi-operation chain with CoT.

    Returns (question_expr, answer_with_cot, first_op, first_a, first_b).
    """
    current = random.randint(10, 999)
    full_expr = _maybe_word(current)
    chain_steps = []
    ops_list = []

    for i in range(num_ops):
        op = random.choice(["+", "-", "*", "/"])

        if op in ("+", "-"):
            operand = random.randint(10, min(current + 100, 9999))
        elif op == "*":
            operand = random.randint(0, 99)
        else:  # division
            divisors = [d for d in range(2, 10) if current >= d and current % d == 0]
            if not divisors:
                op = "+"
                operand = random.randint(10, 999)
            else:
                operand = random.choice(divisors)

        ops_list.append((op, current, operand))
        full_expr += f"{op}{_maybe_word(operand)}"
        expr, steps, result = apply_op(current, op, operand)

        step_str = ", ".join(steps)
        chain_steps.append(f"{expr}: {step_str} → {result}")
        current = result

    answer = ". ".join(chain_steps) + f". La respondo estas {current}. #### {current}"
    return full_expr, answer, ops_list


OP_TO_KEY = {"+": "add", "-": "sub", "*": "mul", "/": "div"}

OP_WORDS_FIRST = {
    "+": ["Aldonu {a} kaj {b}", "Komencu kun {a}, aldonu {b}", "Aldonu {b} al {a}",
          "Komencu kun {a} kaj aldonu {b}"],
    "-": ["Subtrahu {b} de {a}", "Komencu kun {a}, subtrahu {b}",
          "Komencu kun {a} kaj forprenu {b}"],
    "*": ["Multipliku {a} per {b}", "Komencu kun {a}, multipliku per {b}",
          "Komencu kun {a} kaj multipliku ĝin per {b}"],
    "/": ["Dividu {a} per {b}", "Komencu kun {a}, dividu per {b}",
          "Komencu kun {a} kaj dividu ĝin per {b}"],
}

OP_WORDS_THEN = {
    "+": ["poste aldonu {b}", "kaj aldonu {b}", "poste aldonu {b} al la rezulto",
          "kaj aldonu {b}"],
    "-": ["poste subtrahu {b}", "kaj subtrahu {b}", "poste forprenu {b}",
          "kaj forprenu {b} de la rezulto"],
    "*": ["poste multipliku per {b}", "kaj multipliku la rezulton per {b}",
          "poste multipliku ĝin per {b}"],
    "/": ["poste dividu per {b}", "kaj dividu la rezulton per {b}",
          "poste dividu ĝin per {b}"],
}

PREFIXES_QUESTION = [
    "Ĉu vi povus kalkuli: ",
    "Kion ni ricevas se ni ",
    "Kio estas la rezulto se ni ",
]

PREFIXES_IMPERATIVE = [
    "",
    "Bonvolu kalkuli: ",
    "Helpu min kalkuli: ",
]


def make_natural_question(ops_list: list[tuple[str, int, int]]) -> str:
    """Build a natural language question from operation list."""
    parts = []
    for i, (op, a, b) in enumerate(ops_list):
        wa, wb = _maybe_word(a), _maybe_word(b)
        if i == 0:
            parts.append(random.choice(OP_WORDS_FIRST[op]).format(a=wa, b=wb))
        else:
            parts.append(random.choice(OP_WORDS_THEN[op]).format(b=wb))
    body = ", ".join(parts)

    if random.random() < 0.4:
        prefix = random.choice(PREFIXES_QUESTION)
        body = body[0].lower() + body[1:]
        return f"{prefix}{body}?"
    else:
        prefix = random.choice(PREFIXES_IMPERATIVE)
        if prefix:
            body = body[0].lower() + body[1:]
        return f"{prefix}{body}."


def generate_split(n_examples: int, max_tokens: int = 250) -> list[dict]:
    pairs = []

    while len(pairs) < n_examples:
        # 10% chance of percentage question
        if random.random() < 0.1:
            # Pick nice percentages that divide evenly
            p = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90])
            n = random.randint(2, 200) * (100 // math.gcd(p, 100))
            # Keep n reasonable
            if n > 9999:
                n = random.choice([100, 200, 300, 400, 500, 1000])
            q, answer = generate_percent(p, n)
            if len(q) + len(answer) <= max_tokens:
                pairs.append({"messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": answer},
                ]})
            continue

        num_ops = random.randint(1, 5)
        expr, answer, ops_list = generate_chain(num_ops)

        if num_ops == 1:
            op, a, b = ops_list[0]
            key = OP_TO_KEY.get(op, "add")
            q = random.choice(Q_TEMPLATES[key]).format(
                expr=expr, a=_maybe_word(a), b=_maybe_word(b))
        elif random.random() < 0.5:
            # Natural language form
            q = make_natural_question(ops_list)
        else:
            # Expression form
            q = random.choice(["Kalkulu {expr}.", "Kio estas {expr}?", "Kiom estas {expr}?"]).format(expr=expr)

        if len(q) + len(answer) > max_tokens:
            continue
        pairs.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": answer},
        ]})

    random.shuffle(pairs)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate arithmetic CoT dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--test-size", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    for split, size in [("train", args.train_size), ("test", args.test_size)]:
        console.print(f"[bold green]Generating {split} split ({size:,} examples)...")
        pairs = generate_split(size)
        path = args.output_dir / f"{split}.jsonl"
        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        console.print(f"[bold]  {len(pairs):,} → {path}")

    # Show samples
    console.print("\n[bold]Samples:")
    with open(args.output_dir / "train.jsonl") as f:
        lines = f.readlines()
        for line in random.sample(lines, min(5, len(lines))):
            pair = json.loads(line)
            console.print(f"  Q: {pair['messages'][0]['content']}")
            console.print(f"  A: {pair['messages'][1]['content']}")
            console.print()


if __name__ == "__main__":
    main()
