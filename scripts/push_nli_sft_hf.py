"""Push the three EO-translated NLI/commonsense-reasoning datasets to HF.

Sources:
  /mnt/data2/translations/ecqa_{train,validation,test}.jsonl
  /mnt/data2/translations/ecare_{train,validation}.jsonl
  /mnt/data2/translations/esnli_{train,valid,test}.jsonl

For each dataset, push TWO configs:
  default : full parallel EO/EN rows (all fields as translated, incl.
            round-trip English + cos_sim + chrF per field)
  sft     : chat-messages format for SFT training. Assistant response
            includes the rationale plus final answer.

Quality filter: drop rows whose key EO field(s) have LaBSE cos_sim below a
per-dataset threshold, so garbage translations don't reach the SFT mix.

Repos (private by default):
  jensjepsen/esperanto-ecqa
  jensjepsen/esperanto-ecare
  jensjepsen/esperanto-esnli

Usage:
  uv run python scripts/push_nli_sft_hf.py [--dry-run] [--only ecqa]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict

TRANSLATIONS_DIR = Path("/mnt/data2/translations")

# EO labels for e-SNLI
ESNLI_LABEL_EO = {
    "entailment": "implico",
    "neutral": "neŭtrala",
    "contradiction": "kontraŭdiro",
}

# e-CARE question-type translation
ECARE_QUESTION_EO = {"effect": "efiko", "cause": "kaŭzo"}

# LaBSE cos_sim quality thresholds (per-field)
THRESH_MAIN = 0.85   # premise/hypothesis/question
THRESH_ANS = 0.70    # short answer strings (looser — they're 1-3 words)

# Cap e-SNLI train (else it dominates the mix)
ESNLI_TRAIN_CAP = 100_000


# ── Load helpers ───────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


# ── Filters ────────────────────────────────────────────────────────────

def ecqa_pass(r: dict) -> bool:
    """ECQA: keep if question, all 5 options, and rationale translated well."""
    if not r.get("q_text_eo") or not r.get("taskA_pos_eo") or not r.get("q_ans_eo"):
        return False
    if r.get("q_text_cos_sim", 0) < THRESH_MAIN:
        return False
    if r.get("taskA_pos_cos_sim", 0) < THRESH_MAIN:
        return False
    if r.get("q_ans_cos_sim", 0) < THRESH_ANS:
        return False
    for i in range(1, 6):
        if not r.get(f"q_op{i}_eo"):
            return False
    return True


def ecare_pass(r: dict) -> bool:
    if not r.get("premise_eo") or not r.get("choice1_eo") or not r.get("choice2_eo"):
        return False
    if not r.get("conceptual_explanation_eo"):
        return False
    if r.get("premise_cos_sim", 0) < THRESH_MAIN:
        return False
    if r.get("choice1_cos_sim", 0) < THRESH_MAIN:
        return False
    if r.get("choice2_cos_sim", 0) < THRESH_MAIN:
        return False
    if r.get("conceptual_explanation_cos_sim", 0) < THRESH_MAIN:
        return False
    if r.get("question") not in ECARE_QUESTION_EO:
        return False
    return True


def esnli_pass(r: dict) -> bool:
    if not r.get("premise_eo") or not r.get("hypothesis_eo") or not r.get("rationale_eo"):
        return False
    if r.get("premise_cos_sim", 0) < THRESH_MAIN:
        return False
    if r.get("hypothesis_cos_sim", 0) < THRESH_MAIN:
        return False
    if r.get("rationale_cos_sim", 0) < THRESH_MAIN:
        return False
    if r.get("label") not in ESNLI_LABEL_EO:
        return False
    return True


# ── SFT converters ─────────────────────────────────────────────────────

def ecqa_to_sft(r: dict) -> dict | None:
    ops = [r[f"q_op{i}_eo"].strip() for i in range(1, 6)]
    q = r["q_text_eo"].strip()
    letters = ["A", "B", "C", "D", "E"]
    body = q + "\n" + "\n".join(f"{l}) {o}" for l, o in zip(letters, ops))
    answer = r["q_ans_eo"].strip()
    rationale = r["taskA_pos_eo"].strip()
    return {"messages": [
        {"role": "user", "content": body},
        {"role": "assistant", "content": f"{rationale}\nRespondo: {answer}"},
    ]}


def ecare_to_sft(r: dict) -> dict | None:
    prem = r["premise_eo"].strip()
    q_eo = ECARE_QUESTION_EO[r["question"]]
    c1 = r["choice1_eo"].strip()
    c2 = r["choice2_eo"].strip()
    label = int(r["label"])
    gold = c1 if label == 0 else c2
    rationale = r["conceptual_explanation_eo"].strip()
    body = f"Premiso: {prem}\nKio estas la {q_eo}?\nA) {c1}\nB) {c2}"
    return {"messages": [
        {"role": "user", "content": body},
        {"role": "assistant", "content": f"{rationale}\nRespondo: {gold}"},
    ]}


def esnli_to_sft(r: dict) -> dict | None:
    prem = r["premise_eo"].strip()
    hyp = r["hypothesis_eo"].strip()
    rationale = r["rationale_eo"].strip()
    label_eo = ESNLI_LABEL_EO[r["label"]]
    body = (
        f"Premiso: {prem}\nHipotezo: {hyp}\n"
        "Ĉu la premiso implicas la hipotezon, kontraŭdiras ĝin, aŭ estas neŭtrala?"
    )
    return {"messages": [
        {"role": "user", "content": body},
        {"role": "assistant", "content": f"{rationale}\nRespondo: {label_eo}"},
    ]}


# ── Per-dataset pipelines ──────────────────────────────────────────────

def process_ecqa() -> tuple[DatasetDict, DatasetDict]:
    files = {
        "train":      TRANSLATIONS_DIR / "ecqa_train.jsonl",
        "validation": TRANSLATIONS_DIR / "ecqa_validation.jsonl",
        "test":       TRANSLATIONS_DIR / "ecqa_test.jsonl",
    }
    default_dd = {}
    sft_dd = {}
    for split, path in files.items():
        rows = load_jsonl(path)
        kept = [r for r in rows if ecqa_pass(r)]
        sft_rows = [ecqa_to_sft(r) for r in kept]
        sft_rows = [r for r in sft_rows if r is not None]
        print(f"  ECQA {split:<12} in={len(rows):>6}  kept={len(kept):>6}  sft={len(sft_rows):>6}", flush=True)
        default_dd[split] = Dataset.from_list(kept)
        sft_dd[split] = Dataset.from_list(sft_rows)
    return DatasetDict(default_dd), DatasetDict(sft_dd)


def process_ecare() -> tuple[DatasetDict, DatasetDict]:
    files = {
        "train":      TRANSLATIONS_DIR / "ecare_train.jsonl",
        "validation": TRANSLATIONS_DIR / "ecare_validation.jsonl",
    }
    default_dd = {}
    sft_dd = {}
    for split, path in files.items():
        rows = load_jsonl(path)
        kept = [r for r in rows if ecare_pass(r)]
        sft_rows = [ecare_to_sft(r) for r in kept]
        sft_rows = [r for r in sft_rows if r is not None]
        print(f"  e-CARE {split:<12} in={len(rows):>6}  kept={len(kept):>6}  sft={len(sft_rows):>6}", flush=True)
        default_dd[split] = Dataset.from_list(kept)
        sft_dd[split] = Dataset.from_list(sft_rows)
    return DatasetDict(default_dd), DatasetDict(sft_dd)


def process_esnli(cap_train: int = ESNLI_TRAIN_CAP) -> tuple[DatasetDict, DatasetDict]:
    files = {
        "train":      TRANSLATIONS_DIR / "esnli_train.jsonl",
        "validation": TRANSLATIONS_DIR / "esnli_valid.jsonl",
        "test":       TRANSLATIONS_DIR / "esnli_test.jsonl",
    }
    default_dd = {}
    sft_dd = {}
    for split, path in files.items():
        rows = load_jsonl(path)
        kept = [r for r in rows if esnli_pass(r)]
        # Cap train to ESNLI_TRAIN_CAP (deterministic sample)
        if split == "train" and cap_train and len(kept) > cap_train:
            rng = random.Random(42)
            rng.shuffle(kept)
            kept = kept[:cap_train]
        sft_rows = [esnli_to_sft(r) for r in kept]
        sft_rows = [r for r in sft_rows if r is not None]
        label_dist = Counter(r["label"] for r in kept)
        print(
            f"  e-SNLI {split:<12} in={len(rows):>6}  kept={len(kept):>6}  "
            f"sft={len(sft_rows):>6}  labels={dict(label_dist)}", flush=True)
        default_dd[split] = Dataset.from_list(kept)
        sft_dd[split] = Dataset.from_list(sft_rows)
    return DatasetDict(default_dd), DatasetDict(sft_dd)


# ── Main ───────────────────────────────────────────────────────────────

DATASETS = [
    ("ecqa",  "jensjepsen/esperanto-ecqa",  process_ecqa),
    ("ecare", "jensjepsen/esperanto-ecare", process_ecare),
    ("esnli", "jensjepsen/esperanto-esnli", process_esnli),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only process these tags (e.g. --only ecqa ecare)")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--no-private", dest="private", action="store_false")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HF_HUB_TOKEN")
    if not token:
        tp = Path.home() / ".cache/huggingface/token"
        if tp.exists():
            token = tp.read_text().strip()
    if not token and not args.dry_run:
        print("no HF token", file=sys.stderr)
        sys.exit(2)

    for tag, repo, fn in DATASETS:
        if args.only and tag not in args.only:
            continue
        print(f"\n=== {tag} → {repo} ===", flush=True)
        default_dd, sft_dd = fn()
        if args.dry_run:
            continue
        print(f"pushing default config to {repo} …", flush=True)
        default_dd.push_to_hub(repo, config_name="default", token=token, private=args.private)
        print(f"pushing sft config to {repo} …", flush=True)
        sft_dd.push_to_hub(repo, config_name="sft", token=token, private=args.private)
        print(f"→ https://huggingface.co/datasets/{repo}", flush=True)


if __name__ == "__main__":
    main()
