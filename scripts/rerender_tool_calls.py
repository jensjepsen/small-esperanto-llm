"""Re-render tool-call rows into a different SFT format without re-calling
Gemini. Reads raw fields from a JSONL (produced by gen_tool_call_sft.py) and
writes a new JSONL with a rebuilt `messages` field.

Supported formats:
  default        catalog+utterance in user; reasoning + bare JSON in assistant
  code_fence     same as default but JSON is inside ```json ... ``` fence
  system_catalog catalog in a `system` message, utterance in user, rest as default
  call_only      assistant emits ONLY the JSON, no reasoning
  reasoning_after JSON call first, then reasoning below (opposite order)

Usage:
    python scripts/rerender_tool_calls.py --in data/tool_calls/v1.jsonl \\
        --out data/tool_calls/v1_code_fence.jsonl --format code_fence
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _catalog_str(cat: list[dict]) -> str:
    return json.dumps(cat, ensure_ascii=False, indent=2)


def _call_blob(call: Any, fenced: bool = False) -> str:
    if isinstance(call, list):
        body = "\n".join(json.dumps(c, ensure_ascii=False) for c in call)
    else:
        body = json.dumps(call, ensure_ascii=False)
    if fenced:
        return f"```json\n{body}\n```"
    return body


def render_default(row: dict) -> list[dict]:
    cat = row["tool_catalog"]; utt = row["user_utterance"]
    rea = row["assistant_reasoning"]; call = row["assistant_call"]
    user = f"Værktøjer:\n{_catalog_str(cat)}\n\n{utt}"
    if call is None:
        asst = rea.strip()
    else:
        asst = f"{rea.strip()}\n\n{_call_blob(call)}"
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": asst}]


def render_code_fence(row: dict) -> list[dict]:
    cat = row["tool_catalog"]; utt = row["user_utterance"]
    rea = row["assistant_reasoning"]; call = row["assistant_call"]
    user = f"Værktøjer:\n{_catalog_str(cat)}\n\n{utt}"
    if call is None:
        asst = rea.strip()
    else:
        asst = f"{rea.strip()}\n\n{_call_blob(call, fenced=True)}"
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": asst}]


def render_system_catalog(row: dict) -> list[dict]:
    cat = row["tool_catalog"]; utt = row["user_utterance"]
    rea = row["assistant_reasoning"]; call = row["assistant_call"]
    sys_msg = f"Du har adgang til følgende værktøjer:\n{_catalog_str(cat)}"
    if call is None:
        asst = rea.strip()
    else:
        asst = f"{rea.strip()}\n\n{_call_blob(call)}"
    return [{"role": "system", "content": sys_msg},
            {"role": "user", "content": utt},
            {"role": "assistant", "content": asst}]


def render_call_only(row: dict) -> list[dict]:
    cat = row["tool_catalog"]; utt = row["user_utterance"]
    call = row["assistant_call"]; rea = row["assistant_reasoning"]
    user = f"Værktøjer:\n{_catalog_str(cat)}\n\n{utt}"
    # For clarify/refuse (no call), keep reasoning — otherwise drop it.
    asst = rea.strip() if call is None else _call_blob(call)
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": asst}]


def render_reasoning_after(row: dict) -> list[dict]:
    cat = row["tool_catalog"]; utt = row["user_utterance"]
    rea = row["assistant_reasoning"]; call = row["assistant_call"]
    user = f"Værktøjer:\n{_catalog_str(cat)}\n\n{utt}"
    if call is None:
        asst = rea.strip()
    else:
        asst = f"{_call_blob(call)}\n\n{rea.strip()}"
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": asst}]


def render_agent_loop(row: dict) -> list[dict]:
    """Four-turn conversation: user → assistant(call) → tool(result) → assistant(followup).

    Falls back to 2-turn default for clarify/refuse (no tool_result exists).
    """
    cat = row["tool_catalog"]; utt = row["user_utterance"]
    rea = row["assistant_reasoning"]; call = row["assistant_call"]
    result = row.get("tool_result"); followup = row.get("assistant_followup")
    user = f"Værktøjer:\n{_catalog_str(cat)}\n\n{utt}"
    if call is None:
        # clarify / refuse — same 2-turn shape as default
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": rea.strip()}]
    asst_call = f"{rea.strip()}\n\n{_call_blob(call)}"
    tool_content = json.dumps(result, ensure_ascii=False)
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": asst_call},
            {"role": "tool", "content": tool_content},
            {"role": "assistant", "content": (followup or "").strip()}]


def render_separated(row: dict) -> list[dict]:
    """Modern-agent-style separation with tool_call as its own role:
        user → assistant(reasoning only) → tool_call → tool_result → assistant(followup)

    - Reasoning turn is prose only (no JSON), so the model learns to
      "think, then invoke" as two distinct actions.
    - tool_call and tool_result are structured JSON in their own roles —
      chat template renders each differently, easy to mask loss on
      tool_result (world state, not model output).
    - Multi-chain: interleaves tool_call/tool_result pairs.
    - Clarify/refuse: falls back to 2-turn (no call, no result).
    """
    cat = row["tool_catalog"]; utt = row["user_utterance"]
    rea = row["assistant_reasoning"]; call = row["assistant_call"]
    result = row.get("tool_result"); followup = row.get("assistant_followup")
    user = f"Værktøjer:\n{_catalog_str(cat)}\n\n{utt}"

    if call is None:
        # clarify / refuse — no call, no result
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": rea.strip()}]

    msgs: list[dict] = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": rea.strip()},
    ]

    # Multi-chain: pair each call with its result.
    if isinstance(call, list):
        results = result if isinstance(result, list) else [result] * len(call)
        for c, r in zip(call, results):
            msgs.append({"role": "tool_call",
                         "content": json.dumps(c, ensure_ascii=False)})
            msgs.append({"role": "tool_result",
                         "content": json.dumps(r, ensure_ascii=False)})
    else:
        msgs.append({"role": "tool_call",
                     "content": json.dumps(call, ensure_ascii=False)})
        msgs.append({"role": "tool_result",
                     "content": json.dumps(result, ensure_ascii=False)})

    msgs.append({"role": "assistant",
                 "content": (followup or "").strip()})
    return msgs


FORMATTERS = {
    "default": render_default,
    "code_fence": render_code_fence,
    "system_catalog": render_system_catalog,
    "call_only": render_call_only,
    "reasoning_after": render_reasoning_after,
    "agent_loop": render_agent_loop,
    "separated": render_separated,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=list(FORMATTERS), default="separated",
                    help="Message layout. Default 'separated' produces the "
                         "5-message shape (user → assistant → tool_call → "
                         "tool_result → assistant) that plugs directly into "
                         "train_sft_packed.py.")
    args = ap.parse_args()

    fmt = FORMATTERS[args.format]
    inp = Path(args.inp); out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with inp.open() as fi, out.open("w") as fo:
        for line in fi:
            if not line.strip():
                continue
            row = json.loads(line)
            row = dict(row)  # shallow copy
            row["messages"] = fmt(row)
            row["render_format"] = args.format
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"rewrote {n} rows → {out}  format={args.format}")


if __name__ == "__main__":
    main()
