"""Convert HF EO math datasets to multi-turn funcall JSONL.

Reads ``jensjepsen/esperanto-gsm8k`` and ``jensjepsen/esperanto-orca-math``
from HuggingFace and writes one JSONL per dataset using the converter in
``esperanto_lm.funcall.converter``. Rows are dropped if no convertible
arithmetic is found, if any claim fails sympy verification, or (for
orca) if algebraic ``x = N`` leftovers remain.

Usage::

    uv run python scripts/convert_funcall_data.py --out-dir /tmp
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import load_dataset

from esperanto_lm.funcall.converter import (
    convert_gsm_answer,
    convert_orca_answer,
)


def run(ds_name: str, converter, out_path: Path) -> int:
    print(f"\n===== {ds_name} =====", flush=True)
    ds = load_dataset(ds_name, split="train")
    n_ok = 0
    with open(out_path, "w") as f:
        for row in ds:
            user = row["messages"][0]["content"]
            converted = converter(row["messages"][1]["content"])
            if converted is None:
                continue
            rec = {"messages": [{"role": "user", "content": user}] + converted}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
    pct = n_ok / len(ds) * 100
    print(f"converted: {n_ok}/{len(ds)} ({pct:.1f}%) -> {out_path}")
    return n_ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp"))
    p.add_argument(
        "--gsm-name",
        default="jensjepsen/esperanto-gsm8k",
        help="HF dataset name for GSM8K EO source",
    )
    p.add_argument(
        "--orca-name",
        default="jensjepsen/esperanto-orca-math",
        help="HF dataset name for orca-math EO source",
    )
    p.add_argument(
        "--skip-gsm", action="store_true",
        help="Skip GSM8K conversion (only run orca)",
    )
    p.add_argument(
        "--skip-orca", action="store_true",
        help="Skip orca conversion (only run GSM)",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_gsm:
        run(
            args.gsm_name,
            convert_gsm_answer,
            args.out_dir / f"{args.gsm_name.split('/')[-1]}_funcall.jsonl",
        )
    if not args.skip_orca:
        run(
            args.orca_name,
            convert_orca_answer,
            args.out_dir / f"{args.orca_name.split('/')[-1]}_funcall.jsonl",
        )


if __name__ == "__main__":
    main()
