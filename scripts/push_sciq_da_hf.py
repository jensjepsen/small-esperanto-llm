"""Push Danish SciQ to HF Hub.

Two configs, each with train/validation/test splits:
  - default: parallel schema (EN + DA per row)
  - sft: DA messages format for direct SFT consumption
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_rows(path: Path):
    return [r for r in map(json.loads, path.open()) if r.get("da") is not None]


def to_default(rows):
    return [{
        "id": r["id"], "split": r["split"], "idx": r["idx"],
        "en_question":       r["en"]["question"],
        "en_correct_answer": r["en"]["correct_answer"],
        "en_distractor1":    r["en"]["distractor1"],
        "en_distractor2":    r["en"]["distractor2"],
        "en_distractor3":    r["en"]["distractor3"],
        "en_support":        r["en"]["support"],
        "da_question":       r["da"]["question"],
        "da_correct_answer": r["da"]["correct_answer"],
        "da_distractor1":    r["da"]["distractor1"],
        "da_distractor2":    r["da"]["distractor2"],
        "da_distractor3":    r["da"]["distractor3"],
        "da_support":        r["da"]["support"],
    } for r in rows]


def to_sft(rows):
    """Format each row as a Danish MC question:
       user  = Q + A/B/C/D options (shuffled by seed=42+idx to avoid position bias)
       assistant = correct letter + short explanation from support"""
    import random
    out = []
    for r in rows:
        da = r["da"]
        opts = [(da["correct_answer"], True),
                (da["distractor1"], False),
                (da["distractor2"], False),
                (da["distractor3"], False)]
        rng = random.Random(42 + r["idx"] + hash(r["split"]))
        rng.shuffle(opts)
        labels = "ABCD"
        letter_map = {i: labels[i] for i in range(4)}
        correct_letter = None
        opts_text = []
        for i, (opt, is_correct) in enumerate(opts):
            opts_text.append(f"{labels[i]}) {opt}")
            if is_correct:
                correct_letter = labels[i]
        user = f"Spørgsmål: {da['question']}\n" + "\n".join(opts_text)
        # Answer: just the letter + correct option (short, no support inclusion)
        assistant = f"{correct_letter}) {da['correct_answer']}"
        out.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        })
    return out


def split_dd(rows, transform):
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "validation"]
    te = [r for r in rows if r["split"] == "test"]
    return DatasetDict({"train": Dataset.from_list(transform(tr)),
                        "validation": Dataset.from_list(transform(va)),
                        "test": Dataset.from_list(transform(te))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-configs", nargs="*", default=[])
    args = ap.parse_args()

    token = args.token or os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        p = Path.home() / ".cache/huggingface/token"
        if p.exists(): token = p.read_text().strip()
    if not token:
        print("No HF token found.", file=sys.stderr); sys.exit(2)

    rows = load_rows(args.input)
    print(f"loaded {len(rows):,} rows")

    if "default" not in args.skip_configs:
        print("pushing default…", flush=True)
        split_dd(rows, to_default).push_to_hub(args.repo, config_name="default",
                                                token=token, private=args.private)
    if "sft" not in args.skip_configs:
        print("pushing sft…", flush=True)
        split_dd(rows, to_sft).push_to_hub(args.repo, config_name="sft",
                                            token=token, private=args.private)
    print(f"done → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
