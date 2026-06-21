"""Merge all EN↔EO parallel sources into a single MT training dataset.

Aggregates three families of sources:

1. Existing parallel mt/data/parallel/*_dedup.jsonl from the original
   v5b recipe (ccmatrix, xlent, tatoeba, wikimatrix, opus100, bible,
   opensubtitles, wikimedia, ted2020, opusbooks).

2. FineTranslations — pulled from /mnt/data2/datasets/finetranslations/
   (large file, kept off / to avoid disk pressure).

3. Gemini-translated MC benchmarks on HF (today's work) — each row
   contributes multiple parallel pairs:
     - jensjepsen/esperanto-sciq          (Q, correct, 3 distractors, support)
     - jensjepsen/esperanto-balanced-copa (premise, choice1, choice2)
     - jensjepsen/esperanto-piqa          (goal, sol1, sol2)
     - jensjepsen/esperanto-mmlu          (Q, choices[4])
     - jensjepsen/esperanto-gpqa-diamond  (Q, correct, 3 distractors) — off by default
     - jensjepsen/esperanto-triviaqa      (Q, answer)

All splits (train/validation/test/dev) are included; eval is done on a
separate FLORES devtest set so there's no leakage concern.

Dedup across the union, then push to HF. GPQA is OFF by default so the
merged dataset can be public. Each row: {en, eo, src}.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_dataset


PARALLEL_DIR = Path("mt/data/parallel")
FINETRANSLATIONS_PATH = Path(
    "/mnt/data2/datasets/finetranslations/finetranslations.jsonl"
)
EXISTING_FILES = [
    "ccmatrix_filtered_dedup.jsonl",
    "xlent_dedup.jsonl",
    "tatoeba_train_dedup.jsonl",
    "wikimatrix_dedup.jsonl",
    "opus100_train_dedup.jsonl",
    "bible_uedin_dedup.jsonl",
    "opensubtitles_v2024_dedup.jsonl",
    "wikimedia_dedup.jsonl",
    "ted2020_dedup.jsonl",
    "opusbooks_train_dedup.jsonl",
]


def emit_pair(en, eo, src, out):
    """Append one EN↔EO pair if both strings are non-empty and not equal."""
    en = (en or "").strip()
    eo = (eo or "").strip()
    if not en or not eo or en == eo:
        return False
    out.append({"en": en, "eo": eo, "src": src})
    return True


def load_existing(parallel_dir: Path) -> list[dict]:
    rows = []
    for fname in EXISTING_FILES:
        path = parallel_dir / fname
        if not path.exists():
            print(f"  [skip] {fname} (not found)", flush=True)
            continue
        n = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("en") or not r.get("eo"):
                    continue
                r.setdefault("src", fname.replace(".jsonl", ""))
                rows.append({"en": r["en"], "eo": r["eo"], "src": r["src"]})
                n += 1
        print(f"  {fname}: {n:,}", flush=True)
    return rows


def load_finetranslations(path: Path = FINETRANSLATIONS_PATH) -> list[dict]:
    if not path.exists():
        print(f"  [skip] {path} (not found — run download_finetranslations.py)",
              flush=True)
        return []
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("en") and r.get("eo"):
                rows.append({"en": r["en"], "eo": r["eo"],
                             "src": r.get("src", "finetranslations")})
    print(f"  finetranslations.jsonl: {len(rows):,}", flush=True)
    return rows


def load_sciq(rows):
    for split in ("train", "validation", "test"):
        ds = load_dataset("jensjepsen/esperanto-sciq", split=split)
        before = len(rows)
        for r in ds:
            emit_pair(r["en_question"], r["question"],
                      f"sciq:{split}:q", rows)
            emit_pair(r["en_correct_answer"], r["correct_answer"],
                      f"sciq:{split}:correct", rows)
            for i, (en, eo) in enumerate(zip(
                [r["en_distractor1"], r["en_distractor2"],
                 r["en_distractor3"]],
                r["distractors"],
            )):
                emit_pair(en, eo, f"sciq:{split}:d{i+1}", rows)
            emit_pair(r["en_support"], r["support"],
                      f"sciq:{split}:support", rows)
        print(f"  sciq/{split}: +{len(rows) - before:,}", flush=True)


def load_copa(rows):
    for split in ("train", "test"):
        ds = load_dataset("jensjepsen/esperanto-balanced-copa", split=split)
        before = len(rows)
        for r in ds:
            emit_pair(r["en_premise"], r["premise"],
                      f"copa:{split}:premise", rows)
            emit_pair(r["en_choice1"], r["choice1"],
                      f"copa:{split}:c1", rows)
            emit_pair(r["en_choice2"], r["choice2"],
                      f"copa:{split}:c2", rows)
        print(f"  copa/{split}: +{len(rows) - before:,}", flush=True)


def load_piqa(rows):
    for split in ("train", "validation", "test"):
        ds = load_dataset("jensjepsen/esperanto-piqa", split=split)
        before = len(rows)
        for r in ds:
            emit_pair(r["en_goal"], r["goal"],
                      f"piqa:{split}:goal", rows)
            emit_pair(r["en_sol1"], r["sol1"], f"piqa:{split}:s1", rows)
            emit_pair(r["en_sol2"], r["sol2"], f"piqa:{split}:s2", rows)
        print(f"  piqa/{split}: +{len(rows) - before:,}", flush=True)


def load_mmlu(rows):
    for split in ("dev", "validation", "test"):
        try:
            ds = load_dataset("jensjepsen/esperanto-mmlu", split=split)
        except Exception as e:
            print(f"  [skip] mmlu/{split}: {e}", flush=True)
            continue
        before = len(rows)
        for r in ds:
            emit_pair(r["en_question"], r["question"],
                      f"mmlu:{split}:q", rows)
            for i, (en, eo) in enumerate(zip(r["en_choices"], r["choices"])):
                emit_pair(en, eo, f"mmlu:{split}:c{i}", rows)
        print(f"  mmlu/{split}: +{len(rows) - before:,}", flush=True)


def load_gpqa(rows):
    ds = load_dataset("jensjepsen/esperanto-gpqa-diamond", split="train")
    before = len(rows)
    for r in ds:
        emit_pair(r["en_question"], r["question"], "gpqa:train:q", rows)
        emit_pair(r["en_correct_answer"], r["correct_answer"],
                  "gpqa:train:correct", rows)
        for i, (en, eo) in enumerate(zip(r["en_distractors"], r["distractors"])):
            emit_pair(en, eo, f"gpqa:train:d{i+1}", rows)
    print(f"  gpqa/train: +{len(rows) - before:,}", flush=True)


def load_triviaqa(rows):
    for split in ("validation", "train"):
        try:
            ds = load_dataset("jensjepsen/esperanto-triviaqa", split=split)
        except Exception as e:
            print(f"  [skip] triviaqa/{split}: {e}", flush=True)
            continue
        before = len(rows)
        for r in ds:
            emit_pair(r["en_question"], r["question"],
                      f"triviaqa:{split}:q", rows)
            emit_pair(r["en_answer"], r["answer"],
                      f"triviaqa:{split}:a", rows)
        print(f"  triviaqa/{split}: +{len(rows) - before:,}", flush=True)


def dedup(rows: list[dict]) -> list[dict]:
    """Dedup on the (en, eo) pair, whitespace-normalized."""
    seen = set()
    out = []
    for r in rows:
        en_key = " ".join(r["en"].split())
        eo_key = " ".join(r["eo"].split())
        key = (en_key, eo_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-existing", action="store_true", default=True,
                    help="Include the v5b parallel files (default on)")
    ap.add_argument("--no-existing", action="store_false",
                    dest="include_existing")
    ap.add_argument("--include-finetranslations", action="store_true",
                    default=True,
                    help="Include FineTranslations JSONL (default on)")
    ap.add_argument("--include-gpqa", action="store_true", default=False,
                    help="Include GPQA pairs (forces private push). "
                         "Default OFF so merged dataset can be public.")
    ap.add_argument("--push-to", default="jensjepsen/esperanto-mt-parallel",
                    help="HF repo to push to (empty to skip)")
    ap.add_argument("--private", action="store_true", default=False,
                    help="Push private (forced True if GPQA included)")
    ap.add_argument("--out",
                    default="/mnt/data2/datasets/mt-parallel/merged.jsonl",
                    help="Local JSONL output (default /mnt/data2 — large)")
    args = ap.parse_args()

    rows: list[dict] = []

    if args.include_existing:
        print("[1] existing parallel files:")
        rows.extend(load_existing(PARALLEL_DIR))

    if args.include_finetranslations:
        print("\n[2] finetranslations:")
        rows.extend(load_finetranslations())

    print("\n[3] Gemini-translated benchmarks:")
    load_sciq(rows)
    load_copa(rows)
    load_piqa(rows)
    load_mmlu(rows)
    if args.include_gpqa:
        load_gpqa(rows)
    load_triviaqa(rows)

    print(f"\n[4] total before dedup: {len(rows):,}")
    rows = dedup(rows)
    print(f"    after dedup: {len(rows):,}")

    src_counts = Counter(r["src"].split(":")[0] for r in rows)
    print(f"\n[5] source distribution (top-level):")
    for src, n in src_counts.most_common(15):
        print(f"  {src}: {n:,}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[6] wrote {len(rows):,} rows -> {out_path}")

    if args.push_to:
        private = args.private or args.include_gpqa
        if args.include_gpqa and not args.private:
            print("  [note] forcing private push because GPQA included")
        ds = Dataset.from_list(rows)
        ds.push_to_hub(args.push_to, private=private)

        # Push a minimal README with ONLY the names of merged sources.
        from huggingface_hub import HfApi
        api = HfApi()
        names = []
        if args.include_existing:
            names.extend(f.replace(".jsonl", "") for f in EXISTING_FILES)
        if args.include_finetranslations:
            names.append("HuggingFaceFW/finetranslations (epo_Latn)")
        names.extend([
            "jensjepsen/esperanto-sciq",
            "jensjepsen/esperanto-balanced-copa",
            "jensjepsen/esperanto-piqa",
            "jensjepsen/esperanto-mmlu",
        ])
        if args.include_gpqa:
            names.append("jensjepsen/esperanto-gpqa-diamond")
        names.append("jensjepsen/esperanto-triviaqa")
        readme = "\n".join(f"- {n}" for n in names) + "\n"
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.push_to,
            repo_type="dataset",
        )
        print(f"  pushed -> {args.push_to} (private={private})")


if __name__ == "__main__":
    main()
