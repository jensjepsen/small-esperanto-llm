"""Generate arithmetic chain-of-thought SFT dataset.

Converts digit-by-digit decomposed arithmetic into Esperanto
conversation format for SFT training.
"""

import argparse
import json
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
    while len(da) < n: da.append(0)
    while len(db) < n: db.append(0)

    carry = 0
    steps = []
    result_digits = []

    for i in range(n):
        s = da[i] + db[i]
        if carry:
            step = f"{da[i]}+{db[i]}+{carry}={s + carry}"
            s += carry
        else:
            step = f"{da[i]}+{db[i]}={s}"

        new_carry = s // 10
        digit = s % 10
        result_digits.append(digit)

        step += f"c{new_carry}"
        steps.append(step)
        carry = new_carry

    if carry:
        result_digits.append(carry)

    result = int(''.join(str(d) for d in reversed(result_digits)))
    return f"{a}+{b}", steps, result


def decompose_sub(a: int, b: int) -> tuple[str, list[str], int]:
    if a < b:
        a, b = b, a
    da = [int(d) for d in str(a)][::-1]
    db = [int(d) for d in str(b)][::-1]
    n = len(da)
    while len(db) < n: db.append(0)

    borrow = 0
    steps = []
    result_digits = []

    for i in range(n):
        top = da[i] - borrow
        if top < db[i]:
            diff = (top + 10) - db[i]
            if borrow:
                step = f"{da[i]}-{borrow}-{db[i]}+10={diff}b1"
            else:
                step = f"{da[i]}-{db[i]}+10={diff}b1"
            borrow = 1
        else:
            diff = top - db[i]
            if borrow:
                step = f"{da[i]}-{borrow}-{db[i]}={diff}b0"
            else:
                step = f"{da[i]}-{db[i]}={diff}b0"
            borrow = 0

        result_digits.append(diff)
        steps.append(step)

    while len(result_digits) > 1 and result_digits[-1] == 0:
        result_digits.pop()

    result = int(''.join(str(d) for d in reversed(result_digits)))
    return f"{a}-{b}", steps, result


def decompose_mul(a: int, b: int) -> tuple[str, list[str], int]:
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

    result = int(''.join(str(d) for d in reversed(result_digits)))
    return f"{a}*{b}", steps, result


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
        step = f"{current}/{b}={digit}r{remainder}"
        steps.append(step)
        result_digits.append(digit)

    # Strip leading zeros
    while len(result_digits) > 1 and result_digits[0] == 0:
        result_digits.pop(0)

    result = int(''.join(str(d) for d in result_digits))
    return f"{a}/{b}", steps, result


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
            operand = random.randint(2, 9)
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
