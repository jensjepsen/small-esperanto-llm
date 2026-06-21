"""Build math/STEM eval sets for MT training from translated benchmarks.

Produces:
  mt/data/parallel/eval_mmlu_stem.jsonl   — MMLU val rows in STEM subjects
  mt/data/parallel/eval_sciq.jsonl        — SciQ validation rows

Each output row is one EN↔EO parallel pair (question or choice).
We pull only from PUBLIC translated datasets (GPQA is gated/private and
intentionally excluded to keep these eval files publishable). Trainer
caps total samples per set via --eval-max-samples (default 500), so
each file is OK to over-produce.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


MMLU_STEM_SUBJECTS = {
    "abstract_algebra",
    "college_mathematics",
    "high_school_mathematics",
    "elementary_mathematics",
    "college_physics",
    "conceptual_physics",
    "high_school_physics",
    "college_chemistry",
    "high_school_chemistry",
    "college_biology",
    "high_school_biology",
    "astronomy",
    "formal_logic",
    "electrical_engineering",
    "college_computer_science",
    "high_school_computer_science",
    "computer_security",
    "machine_learning",
}


def emit(en, eo, src, out):
    en = (en or "").strip()
    eo = (eo or "").strip()
    if not en or not eo or en == eo:
        return False
    out.write(json.dumps({"en": en, "eo": eo, "src": src},
                         ensure_ascii=False) + "\n")
    return True


def build_mmlu_stem(out_path: Path):
    # Use TEST split (held out from training). Training pulls
    # dev+auxiliary_train; val/test reserved here for clean eval.
    ds = load_dataset("jensjepsen/esperanto-mmlu", split="test")
    n = 0
    with open(out_path, "w") as f:
        for r in ds:
            if r["subject"] not in MMLU_STEM_SUBJECTS:
                continue
            if emit(r["en_question"], r["question"],
                    f"mmlu:{r['subject']}:q", f):
                n += 1
            for i, (en, eo) in enumerate(zip(r["en_choices"], r["choices"])):
                if emit(en, eo, f"mmlu:{r['subject']}:c{i}", f):
                    n += 1
    print(f"  mmlu_stem: {n} pairs -> {out_path}")


def build_sciq(out_path: Path):
    # Use TEST split (held out from training).
    ds = load_dataset("jensjepsen/esperanto-sciq", split="test")
    n = 0
    with open(out_path, "w") as f:
        for r in ds:
            if emit(r["en_question"], r["question"], "sciq:q", f):
                n += 1
            if emit(r["en_correct_answer"], r["correct_answer"],
                    "sciq:correct", f):
                n += 1
    print(f"  sciq: {n} pairs -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("mt/data/parallel"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    build_mmlu_stem(args.out_dir / "eval_mmlu_stem.jsonl")
    build_sciq(args.out_dir / "eval_sciq.jsonl")


if __name__ == "__main__":
    main()
