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


Q_TEMPLATES = {
    "add": [
        "Kalkulu {expr}.",
        "Kio estas {expr}?",
        "Kiom estas {expr}?",
    ],
    "sub": [
        "Kalkulu {expr}.",
        "Kio estas {expr}?",
        "Kiom estas {expr}?",
    ],
    "mul": [
        "Kalkulu {expr}.",
        "Kio estas {expr}?",
        "Kiom estas {expr}?",
        "Multipliku {a} per {b}.",
    ],
}


def format_answer(expr: str, steps: list[str], result: int) -> str:
    """Format the chain-of-thought answer."""
    step_str = ", ".join(steps)
    return f"{expr}: {step_str}. La respondo estas {result}. #### {result}"


def generate_split(n_examples: int) -> list[dict]:
    pairs = []

    for _ in range(n_examples // 3):
        digits = random.randint(2, 5)
        a = random.randint(10**(digits-1), 10**digits - 1)
        b = random.randint(10**(digits-1), 10**digits - 1)
        expr, steps, result = decompose_add(a, b)
        q = random.choice(Q_TEMPLATES["add"]).format(expr=expr, a=a, b=b)
        pairs.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": format_answer(expr, steps, result)},
        ]})

    for _ in range(n_examples // 3):
        digits = random.randint(2, 5)
        a = random.randint(10**(digits-1), 10**digits - 1)
        b = random.randint(10**(digits-1), 10**digits - 1)
        if a < b: a, b = b, a
        expr, steps, result = decompose_sub(a, b)
        q = random.choice(Q_TEMPLATES["sub"]).format(expr=expr, a=a, b=b)
        pairs.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": format_answer(expr, steps, result)},
        ]})

    for _ in range(n_examples // 3):
        digits = random.randint(2, 4)
        a = random.randint(10**(digits-1), 10**digits - 1)
        b = random.randint(2, 9)
        expr, steps, result = decompose_mul(a, b)
        q = random.choice(Q_TEMPLATES["mul"]).format(expr=expr, a=a, b=b)
        pairs.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": format_answer(expr, steps, result)},
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
