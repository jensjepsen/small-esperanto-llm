"""Run factual Q&A benchmark against an SFT checkpoint."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM

from esperanto_lm.data import load_tokenizer, _morpheme_preprocess

console = Console()

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"
SKIP_TOKENS = {"<s>", "</s>", "<pad>", "<unk>", USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN}


def decode_tokens(tokenizer, token_ids):
    gen_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    gen_tokens = [t for t in gen_tokens if t not in SKIP_TOKENS]
    has_w = "<w>" in tokenizer.get_vocab()
    if has_w:
        text = "".join(t if t != "<w>" else " " for t in gen_tokens)
    else:
        text = " ".join(gen_tokens)
    return text.strip()


def generate(model, tokenizer, prompt, device, max_new_tokens=60):
    content = _morpheme_preprocess(prompt)
    text = f"{USER_TOKEN} {content} {ASSISTANT_TOKEN}"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.95,
            eos_token_id=end_id if end_id != tokenizer.unk_token_id else None,
        )

    new_ids = output[0][inputs["input_ids"].shape[1]:].tolist()
    if end_id in new_ids:
        new_ids = new_ids[:new_ids.index(end_id)]
    return decode_tokens(tokenizer, new_ids)


def check_answer(response: str, expected, category: str = "") -> bool:
    """Check if the expected answer appears in the response.

    expected can be a string or a list of acceptable answers.
    For arithmetic, extracts the #### answer or matches word boundaries.
    """
    response = response.lower().strip(". ")

    if category == "aritmetiko":
        # Try #### extraction first
        if "####" in response:
            final = response.split("####")[-1].strip().rstrip(".")
            if isinstance(expected, list):
                return any(final == ans.lower().strip() for ans in expected)
            return final == expected.lower().strip()
        # Fall back to word-boundary match for bare numbers
        import re
        if isinstance(expected, list):
            return any(re.search(r'\b' + re.escape(ans.lower().strip()) + r'\b', response) for ans in expected)
        return bool(re.search(r'\b' + re.escape(expected.lower().strip()) + r'\b', response))

    if isinstance(expected, list):
        return any(ans.lower().strip() in response for ans in expected)
    return expected.lower().strip() in response


def main():
    parser = argparse.ArgumentParser(description="Run factual Q&A benchmark")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/factual_qa.json"))
    parser.add_argument("--passes", type=int, default=3,
                        help="Number of passes per question (majority vote)")
    parser.add_argument("--verbose", action="store_true", help="Show each question")
    args = parser.parse_args()

    with open(args.benchmark) as f:
        questions = json.load(f)

    console.print(f"[bold green]Loading model from {args.checkpoint}...")
    tokenizer = load_tokenizer(Path(args.checkpoint))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    console.print(f"[bold]Benchmark:[/] {len(questions)} questions, {args.passes} passes each")
    console.print()

    correct_by_cat = Counter()
    total_by_cat = Counter()
    wrong = []

    for q in questions:
        question = q["question"]
        expected = q["answer"]
        category = q["category"]
        total_by_cat[category] += 1

        # Run multiple passes, take majority
        hits = 0
        responses = []
        for _ in range(args.passes):
            response = generate(model, tokenizer, question, device)
            responses.append(response)
            if check_answer(response, expected, category):
                hits += 1

        passed = hits > args.passes // 2

        if passed:
            correct_by_cat[category] += 1

        if args.verbose:
            mark = "✅" if passed else "❌"
            console.print(f"{mark} {question}")
            console.print(f"   Expected: {expected} | Got: {responses[0]}")
            if not passed and hits > 0:
                console.print(f"   ({hits}/{args.passes} passes correct)")
            console.print()
        elif not passed:
            wrong.append({
                "question": question,
                "expected": expected,
                "got": responses[0],
                "category": category,
                "hits": hits,
            })

    # Results table
    table = Table(title="Factual Q&A Benchmark Results")
    table.add_column("Category", style="bold")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Bar")

    categories = sorted(total_by_cat.keys(), key=lambda c: -correct_by_cat[c] / total_by_cat[c])
    for cat in categories:
        correct = correct_by_cat[cat]
        total = total_by_cat[cat]
        pct = correct / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        color = "green" if pct >= 70 else "yellow" if pct >= 40 else "red"
        table.add_row(cat, str(correct), str(total), f"[{color}]{pct:.0f}%[/{color}]", bar)

    total_correct = sum(correct_by_cat.values())
    total_qs = sum(total_by_cat.values())
    total_pct = total_correct / total_qs * 100
    table.add_section()
    table.add_row("[bold]TOTAL", f"[bold]{total_correct}", f"[bold]{total_qs}",
                  f"[bold]{total_pct:.1f}%", "")

    console.print(table)

    # Show wrong answers (if not verbose)
    if not args.verbose and wrong:
        console.print(f"\n[bold red]Wrong answers ({len(wrong)}):")
        for w in wrong[:20]:
            console.print(f"  {w['category']:12s} | {w['question']}")
            console.print(f"               Expected: {w['expected']} | Got: {w['got']}")
            if w['hits'] > 0:
                console.print(f"               ({w['hits']}/{args.passes} passes)")
            console.print()
        if len(wrong) > 20:
            console.print(f"  ... and {len(wrong) - 20} more")


if __name__ == "__main__":
    main()
