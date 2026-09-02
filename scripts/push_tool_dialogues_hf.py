"""Push the Danish tool-dialogue corpus to the Hub.

Named `danish-tool-dialogues-v1`, NOT tool-calls-v2: it is not a new version of
`danish-tool-calls-v1` but a different dataset built from a different source by
a different method. Per the naming convention, versions are reserved for
supersession and siblings get their own names. The two are complementary --
tool-calls-v1 is single-turn synthetic Danish, this is multi-turn translated
ToolMind/Glaive with reasoning.

SPLITS. Two eval buckets, because "can it call a tool it was trained on" and
"can it call a tool it has never seen" are different questions and only the
second measures generalisation:

    train              conversations whose tools all appear in training
    eval_seen_tools    held-out CONVERSATIONS over tools that do appear
    eval_unseen_tools  conversations using a held-out TOOL NAME

The unseen bucket is chosen by hashing the tool NAME, and a conversation lands
there if ANY of its tools is held out -- so a held-out name cannot leak into
train through a multi-tool catalogue. eval_seen isolates the other axis: same
tools, unseen dialogue.

CONFIGS. `default` (tools + conversations + meta), `sft` (messages, ready for
--sft-data), `en` (the untranslated source row, for provenance and A/B), and
`rejected` (rows that failed the gate, with their verdicts, published rather
than silently dropped).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi

REPO = "jensjepsen/danish-tool-dialogues-v1"
SPLITS = ["train", "eval_seen_tools", "eval_unseen_tools"]


def bucket(s: str, mod: int = 100) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def tool_names(row) -> list[str]:
    """Names in the CATALOGUE."""
    return [((t.get("function") or {}).get("name") or "")
            for t in row.get("tools", []) if isinstance(t, dict)]


def called_names(row) -> list[str]:
    """Names actually INVOKED. The split hinges on this, not on the catalogue:
    a row whose catalogue merely mentions a held-out tool while calling a
    common one tests nothing about unseen tools."""
    out = []
    for m in row.get("conversations", []):
        for tc in (m.get("tool_calls") or []):
            n = (tc.get("function") or {}).get("name")
            if n:
                out.append(n)
    return out


def _cfg(cfg, base, splits):
    lines = [f"- config_name: {cfg}", "  data_files:"]
    for sp in splits:
        lines += [f"  - split: {sp}", f"    path: {base}/{sp}-*"]
    return "\n".join(lines)


def card(counts, n_rejected, tool_stats, fails):
    rows = "\n".join(f"| `{s}` | {counts[s]:,} |" for s in SPLITS)
    fl = "\n".join(f"| `{k}` | {v:,} |" for k, v in fails.most_common())
    return f"""---
language:
- da
license: apache-2.0
task_categories:
- text-generation
tags:
- danish
- function-calling
- tool-use
- multi-turn
configs:
{_cfg("default", "data", SPLITS)}
{_cfg("sft", "sft", SPLITS)}
{_cfg("en", "en", SPLITS)}
- config_name: rejected
  data_files:
  - split: train
    path: rejected/train-*
---

# danish-tool-dialogues-v1

Danish multi-turn tool-use conversations with reasoning, translated from the
Glaive subset of
[`Nanbeige/ToolMind`](https://huggingface.co/datasets/Nanbeige/ToolMind)
(Apache-2.0) by `scripts/translate_toolmind_da.py`.

Complements `danish-tool-calls-v1`, which is single-turn and synthetic. Here
the conversations run several turns, tool results are fed back, and the
assistant reasons before calling.

| split | rows |
|---|---|
{rows}

`{tool_stats['n_tools']:,}` distinct tools; `{tool_stats['n_heldout']:,}` names
are held out entirely, so `eval_unseen_tools` measures whether the model can
call a tool it has never been trained on. `eval_seen_tools` holds out
conversations rather than tools, isolating dialogue novelty from tool novelty.

## What is Danish and what is not

**Danish**: user turns, the assistant's reasoning and replies, tool and
parameter descriptions, enum values, and natural-language argument values —
including content the user chose, such as a note title.

**Unchanged**: tool names, parameter keys, every JSON key, and machine values
(acronyms, dates, numbers, emails, URLs, ISO codes, `snake_case` identifiers).
A Danish user talks to an English-named API, and pinning the surface keeps the
data exactly verifiable — a reward function can compare keys and names for
equality.

Enum values *are* translated, but once: the spec's list is translated and every
invocation inherits that exact string, so the contract stays coherent rather
than English. A call carrying `"cirkel"` is valid only if the spec offers
`"cirkel"`.

Reasoning is plain assistant text. The source `<think>` wrapper is removed.

## Gates

Every row is checked mechanically: structure identical to the source (keys,
nesting, types), tool names and parameter keys byte-identical, enum spec and
invocations in agreement, argument values traceable to the conversation that
introduced them, machine identifiers surviving inside Danish prose, and
translated fields detected as Danish by `langdetect`. Six planted controls, one
per check, must fail on every run — a gate that never fires is
indistinguishable from clean data.

{f"| failure | rows |{chr(10)}|---|---|{chr(10)}{fl}" if fl else ""}

`{n_rejected:,}` rows failed and are published under the `rejected` config with
their verdicts rather than dropped silently. Most are a single skipped short
description; they are usable with care.

## Known limitations

Conversations whose *subject* is language (`translate_text`, `detect_language`)
were removed at source — translating them destroys the premise, since a user
asking to translate an English sentence ends up quoting a Danish one while the
call still says `source_language="English"`.

Grammatical gender and inflection errors occur at a low rate and no mechanical
gate detects them.

## Configs

`default` (tools + conversations as JSON strings, plus counts and tool names),
`sft` (messages only, for completion-only training), `en` (the untranslated
source row), `rejected`.

`tools` and `conversations` are JSON strings rather than nested structs: the
tool schemas are heterogeneous enough that Arrow cannot infer a single type
across 18k rows. `json.loads` them. `sft` is structured, since its
message objects are uniform.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("scratch/toolmind_da_v2"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--heldout-pct", type=int, default=6,
                    help="share of TOOL NAMES reserved for eval_unseen_tools")
    ap.add_argument("--eval-pct", type=int, default=4,
                    help="share of remaining rows for eval_seen_tools")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_toolmind_sft import to_messages

    recs = [json.loads(l) for l in (args.src / "translated.jsonl").open()
            if l.strip()]
    verdicts = {}
    for line in (args.src / "gate_verdicts.jsonl").open():
        v = json.loads(line)
        verdicts[v["idx"]] = v["bad"]
    clean = [r for r in recs if not verdicts.get(r.get("idx"), ["unknown"])]
    rejected = [r for r in recs if verdicts.get(r.get("idx"), ["unknown"])]
    fails = Counter(b.split("(")[0] for r in rejected
                    for b in verdicts.get(r["idx"], []))
    print(f"clean {len(clean):,}  rejected {len(rejected):,}", flush=True)

    names = sorted({n for r in clean for n in tool_names(r["da"]) if n})
    heldout = {n for n in names if bucket(n) < args.heldout_pct}
    print(f"tools: {len(names):,} distinct, {len(heldout):,} held out",
          flush=True)

    data = {s: [] for s in SPLITS}
    dropped = 0
    for r in clean:
        da = r["da"]
        tn = [n for n in tool_names(da) if n]
        cn = [n for n in called_names(da) if n]
        if any(n in heldout for n in cn):
            split = "eval_unseen_tools"      # actually CALLS a held-out tool
        elif any(n in heldout for n in tn):
            # A held-out name sits in the catalogue but is never called. It
            # cannot go to train -- the model would see the name and
            # description in a prompt, so the tool is no longer unseen -- and
            # it belongs in no eval bucket either, because the target call
            # uses a tool the model has seen. Drop it.
            dropped += 1
            continue
        elif bucket(str(r["idx"]) + "seen") < args.eval_pct:
            split = "eval_seen_tools"
        else:
            split = "train"
        msgs = to_messages(da)
        if msgs is None:
            continue
        data[split].append({
            "tools": da.get("tools", []),
            "conversations": da.get("conversations", []),
            "messages": msgs,
            "en": r["orig"],
            "meta": {"idx": r["idx"], "n_tools": len(tn),
                     "n_turns": len(da.get("conversations", [])),
                     "tool_names": tn},
        })
    counts = {s: len(v) for s, v in data.items()}
    print("splits:", counts, f"(dropped {dropped:,} catalogue-only rows)",
          flush=True)

    # eval_seen must test tools the model HAS trained on. A rare tool whose
    # only rows landed in the eval sample would otherwise sit here untrained,
    # quietly making this bucket a second unseen-tool test.
    train_names = {n for r in data["train"] for n in r["meta"]["tool_names"]}
    keep, moved = [], 0
    for r in data["eval_seen_tools"]:
        if all(n in train_names for n in r["meta"]["tool_names"]):
            keep.append(r)
        else:
            data["train"].append(r)
            moved += 1
    data["eval_seen_tools"] = keep
    if moved:
        print(f"moved {moved:,} eval_seen rows to train (their tools were not "
              f"otherwise trained)", flush=True)
    counts = {s: len(v) for s, v in data.items()}

    # VERIFY THE SPLIT, do not assume it. A held-out tool leaking into train
    # would make eval_unseen_tools measure recall instead of generalisation.
    train_tools = {n for r in data["train"] for n in r["meta"]["tool_names"]}

    leak = train_tools & heldout
    unseen_called = {n for r in data["eval_unseen_tools"]
                     for n in called_names(
                         {"conversations": r["conversations"]})}
    seen_tools = {n for r in data["eval_seen_tools"]
                  for n in r["meta"]["tool_names"]}
    print(f"held-out tools appearing in TRAIN (must be 0): {len(leak)}")
    print(f"eval_unseen CALLED tools never in train: "
          f"{len(unseen_called - train_tools)}/{len(unseen_called)}")
    print(f"eval_seen tools also in train: "
          f"{len(seen_tools & train_tools)}/{len(seen_tools)}")
    assert not leak, f"held-out tools leaked into train: {sorted(leak)[:5]}"

    if args.dry_run:
        print("\ndry run")
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    # `tools` and `conversations` ship as JSON STRINGS. Arrow cannot infer one
    # schema across them: `arguments` is a dict in almost every row and a list
    # in two, and parameter objects vary in shape, which fails with "cannot mix
    # list and non-list values". ToolMind already stores tool results this way,
    # so a consumer is doing json.loads on that field regardless. `sft` stays
    # structured because {role, content} is uniform.
    J = lambda o: json.dumps(o, ensure_ascii=False)          # noqa: E731
    views = {
        "default": lambda R: [{"tools": J(x["tools"]),
                               "conversations": J(x["conversations"]),
                               "n_tools": x["meta"]["n_tools"],
                               "n_turns": x["meta"]["n_turns"],
                               "tool_names": x["meta"]["tool_names"],
                               "idx": x["meta"]["idx"]} for x in R],
        "sft": lambda R: [{"messages": x["messages"]} for x in R],
        "en": lambda R: [{"en": J(x["en"]), "idx": x["meta"]["idx"]}
                         for x in R],
    }
    for cfg, fn in views.items():
        for sp in SPLITS:
            Dataset.from_list(fn(data[sp])).push_to_hub(
                args.repo, config_name=cfg, split=sp,
                commit_message=f"{cfg}/{sp} ({counts[sp]} rows)")
            print(f"  pushed {cfg}/{sp} ({counts[sp]})", flush=True)
    Dataset.from_list([{"en": J(r["orig"]), "da": J(r["da"]),
                        "verdict": verdicts.get(r["idx"], []),
                        "idx": r["idx"]} for r in rejected]).push_to_hub(
        args.repo, config_name="rejected", split="train",
        commit_message=f"rejected ({len(rejected)} rows)")
    print(f"  pushed rejected ({len(rejected)})", flush=True)

    stats = {"n_tools": len(names), "n_heldout": len(heldout)}
    api.upload_file(
        path_or_fileobj=card(counts, len(rejected), stats, fails).encode(),
        path_in_repo="README.md", repo_id=args.repo, repo_type="dataset",
        commit_message="dataset card")
    print(f"-> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
