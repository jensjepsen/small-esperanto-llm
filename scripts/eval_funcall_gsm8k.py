"""Tool-loop greedy eval on the EO GSM8K test split (HF, properly aligned).

Uses ``jensjepsen/esperanto-gsm8k`` test split (1,279 rows). Gold answer
is extracted from the ``#### N`` marker in each row's assistant message,
so the EO question and gold are guaranteed aligned within a row.

Logs full per-item text + tool call trace so spurious-match analysis
can be done from the log without re-running the model.

Usage::

    CKPT=runs/v8_full_mix/final N=100 \\
        uv run python scripts/eval_funcall_gsm8k.py \\
        | tee /tmp/eval_v8_gsm8k_hf.log
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import load_dataset

from esperanto_lm.funcall.runtime import ToolInferenceRunner


def extract_final(text: str, last_result: str) -> str:
    """Parse the final-answer number from the model's rendered output.

    Strategy: take the assistant text after the LAST tool result, look
    for ``#### N``, else the last bare number in that tail, else the
    last tool result itself.
    """
    parts = text.split("<|/tool_result|>")
    tail = parts[-1] if parts else text
    tail = tail.split("<|tool_call|>")[0]
    m = re.search(r"####\s*([\-0-9]+(?:\.\d+)?)", tail)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", tail)
    if nums:
        return nums[-1]
    return last_result or ""


def matches(p: str, g: str) -> bool:
    if not p:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-6
    except Exception:
        return p.strip() == g.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        default=os.environ.get("CKPT"),
        required=os.environ.get("CKPT") is None,
        help="Path to SFT checkpoint (or set $CKPT)",
    )
    ap.add_argument(
        "--n", type=int, default=int(os.environ.get("N", "100")),
        help="Number of test items (or set $N)",
    )
    ap.add_argument(
        "--dataset", default="jensjepsen/esperanto-gsm8k",
        help="HF dataset name (test split is used)",
    )
    ap.add_argument(
        "--max-hops", type=int, default=5,
        help="Maximum tool-call hops per question",
    )
    ap.add_argument(
        "--max-new-per-hop", type=int, default=200,
        help="Maximum new tokens generated per hop",
    )
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split="test")
    pairs = []
    for r in ds:
        a = r["messages"][1]["content"]
        m = re.search(r"####\s*([\-0-9,]+)", a)
        if not m:
            continue
        pairs.append({"eo": r["messages"][0]["content"],
                      "gold": m.group(1).replace(",", "").strip()})
    pairs = pairs[: args.n]
    print(f"using {args.checkpoint}  N={len(pairs)}", flush=True)

    runner = ToolInferenceRunner(args.checkpoint)
    ok = 0
    tool_correct = 0
    calls_total = 0
    t0 = time.time()
    for i, p in enumerate(pairs, 1):
        res = runner.chat(
            p["eo"],
            max_hops=args.max_hops,
            max_new_per_hop=args.max_new_per_hop,
        )
        calls = [t for t in res["trace"] if t[0] == "tool"]
        last_res = calls[-1][2] if calls else ""
        pred = extract_final(res["text"], last_res)
        ok_i = matches(pred, p["gold"])
        if ok_i:
            ok += 1
        calls_total += len(calls)
        if any(matches(t[2], p["gold"]) for t in calls):
            tool_correct += 1
        elap = time.time() - t0
        mark = "OK" if ok_i else "X "
        print(
            f"[{i:3d}/{len(pairs)}] {mark} gold={p['gold']:>6} pred={pred:>8}"
            f"  calls={len(calls)}  acc={ok/i:.3f}  elap={elap:.0f}s",
            flush=True,
        )
        # Full per-item dump so spurious-match analysis doesn't need re-run
        print(f"  Q: {p['eo'][:200]}")
        print(f"  CALLS: {[(t[1], t[2]) for t in calls]}")
        print(f"  TEXT: {res['text'][-400:]}")
        print(flush=True)

    print(f"\n=== {args.checkpoint} pass@1: {ok}/{len(pairs)} = "
          f"{ok/len(pairs)*100:.1f}% ===")
    print(f"avg tool calls: {calls_total/len(pairs):.1f}")
    print(f"tool-correct (any call hit gold): {tool_correct}/{len(pairs)} = "
          f"{tool_correct/len(pairs)*100:.1f}%")


if __name__ == "__main__":
    main()
