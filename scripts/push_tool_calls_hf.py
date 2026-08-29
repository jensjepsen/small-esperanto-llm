"""Push the Danish tool-call dataset to HF Hub.

Two configs, each with train + eval splits:
  - default: raw fields (tool_catalog, user_utterance, assistant_reasoning,
             assistant_call, tool_result, assistant_followup + metadata) —
             source of truth, reformattable via scripts/rerender_tool_calls.py
  - sft:     messages list ready to feed into scripts/train_sft_packed.py
             (produced with format=separated)
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_rows(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def _to_json(v):
    """Serialize as JSON string for Arrow compat. None → None (preserved as null)."""
    return None if v is None else json.dumps(v, ensure_ascii=False)


def build_default(rows):
    """Raw rows — variable-shape JSON fields serialized as strings so Arrow
    can hold them in single-typed columns. Consumers `json.loads()` them
    back (schema is uniform: catalog is always a list of tool dicts; call
    is dict or list-of-dicts or None; result is any JSON value or None).
    """
    return [{
        "tool_catalog":         _to_json(r["tool_catalog"]),   # list[dict]
        "target_tool":          r["target_tool"],              # string
        "user_utterance":       r["user_utterance"],           # string
        "assistant_reasoning":  r["assistant_reasoning"],      # string
        "assistant_call":       _to_json(r["assistant_call"]),  # dict / list / None
        "tool_result":          _to_json(r.get("tool_result")),  # any JSON / None
        "assistant_followup":   r.get("assistant_followup"),    # string / None
        "difficulty":           r["difficulty"],
        "scenario":             r["scenario"],
        "structural_hash":      r["structural_hash"],
        "n_tools_in_catalog":   r["n_tools_in_catalog"],
    } for r in rows]


def build_sft(rows):
    """Messages field — native list[{role, content}]. Arrow handles
    varying list length as long as struct schema is uniform (it is).
    """
    return [{"messages": r["messages"]} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-train", type=Path,
                    default=Path("data/tool_calls/v1_train.jsonl"))
    ap.add_argument("--raw-eval", type=Path,
                    default=Path("data/tool_calls/v1_eval.jsonl"))
    ap.add_argument("--msg-train", type=Path,
                    default=Path("data/tool_calls/v1_train_messages.jsonl"))
    ap.add_argument("--msg-eval", type=Path,
                    default=Path("data/tool_calls/v1_eval_messages.jsonl"))
    ap.add_argument("--repo", default="jensjepsen/danish-tool-calls-v1")
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--skip-configs", nargs="*", default=[])
    args = ap.parse_args()

    token = args.token or os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        p = Path.home() / ".cache/huggingface/token"
        if p.exists():
            token = p.read_text().strip()
    if not token:
        print("No HF token found.", file=sys.stderr)
        sys.exit(2)

    if "default" not in args.skip_configs:
        print(f"loading raw train/eval …", flush=True)
        raw_train = load_rows(args.raw_train)
        raw_eval = load_rows(args.raw_eval)
        print(f"  train={len(raw_train):,}  eval={len(raw_eval):,}")
        dd = DatasetDict({
            "train": Dataset.from_list(build_default(raw_train)),
            "eval":  Dataset.from_list(build_default(raw_eval)),
        })
        print("pushing default config …", flush=True)
        dd.push_to_hub(args.repo, config_name="default",
                       token=token, private=args.private)

    if "sft" not in args.skip_configs:
        print(f"loading messages train/eval …", flush=True)
        sft_train = load_rows(args.msg_train)
        sft_eval = load_rows(args.msg_eval)
        print(f"  train={len(sft_train):,}  eval={len(sft_eval):,}")
        dd = DatasetDict({
            "train": Dataset.from_list(build_sft(sft_train)),
            "eval":  Dataset.from_list(build_sft(sft_eval)),
        })
        print("pushing sft config …", flush=True)
        dd.push_to_hub(args.repo, config_name="sft",
                       token=token, private=args.private)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
