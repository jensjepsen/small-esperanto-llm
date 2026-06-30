"""Convert the 5 pretrain-benchmark QA datasets to SFT messages format and
push each to HF as a new dataset.

These are loaded raw via `load_benchmark_qa_dataset` in data.py for
pretrain; this script makes them available in chat-template SFT form so
the model learns to surface the pretrained knowledge through the
<|assistant|> turn.

Naming: <existing-repo>-sft (e.g. `esperanto-sciq` → `esperanto-sciq-sft`).
Schema: {"messages": [{"role": "user", "content": q},
                      {"role": "assistant", "content": correct_answer}]}

Skip rows where q or a is empty after stripping.
"""
import argparse
import os
from pathlib import Path

from datasets import Dataset, load_dataset


def _emit(q, a, out):
    q = (q or "").strip()
    a = (str(a) if a is not None else "").strip()
    if not q or not a:
        return
    out.append({
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    })


def convert_sciq(ds, out):
    for r in ds:
        _emit(r["question"], r["correct_answer"], out)


def convert_copa(ds, out):
    for r in ds:
        a = r["choice1"] if r["label"] == 0 else r["choice2"]
        _emit(r["premise"], a, out)


def convert_piqa(ds, out):
    for r in ds:
        a = r["sol1"] if r["label"] == 0 else r["sol2"]
        _emit(r["goal"], a, out)


def convert_mmlu(ds, out):
    for r in ds:
        choices = r["choices"]
        idx = int(r["answer"])
        if 0 <= idx < len(choices):
            _emit(r["question"], choices[idx], out)


def convert_triviaqa(ds, out):
    for r in ds:
        _emit(r["question"], r["answer"], out)


SOURCES = [
    # (source_repo, split, target_repo, converter)
    ("jensjepsen/esperanto-sciq",          "train",            "jensjepsen/esperanto-sciq-sft",          convert_sciq),
    ("jensjepsen/esperanto-balanced-copa", "train",            "jensjepsen/esperanto-balanced-copa-sft", convert_copa),
    ("jensjepsen/esperanto-piqa",          "train",            "jensjepsen/esperanto-piqa-sft",          convert_piqa),
    ("jensjepsen/esperanto-mmlu",          "auxiliary_train",  "jensjepsen/esperanto-mmlu-sft",          convert_mmlu),
    ("jensjepsen/esperanto-triviaqa",      "train",            "jensjepsen/esperanto-triviaqa-sft",      convert_triviaqa),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--no-private", action="store_false", dest="private")
    ap.add_argument("--dry-run", action="store_true",
                    help="Convert but skip the push (just report row counts).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only process target repos containing these strings.")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN") or (
        Path.home() / ".cache/huggingface/token").read_text().strip()

    summary = []
    for src, split, dst, conv in SOURCES:
        if args.only and not any(s in dst for s in args.only):
            continue
        print(f"--- loading {src} / {split} ...", flush=True)
        try:
            ds = load_dataset(src, split=split)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            summary.append((dst, 0, "load-failed"))
            continue
        rows = []
        conv(ds, rows)
        n_in = len(ds)
        n_out = len(rows)
        print(f"  {n_in:,} → {n_out:,} SFT rows after empty-strip", flush=True)
        if args.dry_run:
            summary.append((dst, n_out, "dry-run"))
            continue
        msg_ds = Dataset.from_list(rows)
        print(f"  pushing to {dst} (private={args.private})...", flush=True)
        msg_ds.push_to_hub(dst, token=token, private=args.private)
        summary.append((dst, n_out, "ok"))
        print(f"  done", flush=True)

    print("\n=== SUMMARY ===")
    total = 0
    for dst, n, status in summary:
        print(f"  {dst:55s}  {n:>7,} rows  {status}")
        total += n
    print(f"  {'TOTAL':55s}  {total:>7,}")


if __name__ == "__main__":
    main()
