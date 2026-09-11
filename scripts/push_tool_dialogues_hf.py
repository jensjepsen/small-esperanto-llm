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
# A symbolized twin of an EVAL row goes to its own split, so `tool_unseen` and
# `tool_unseen_sym` measure the same conversations with and without recallable
# parameter names -- their difference IS the description-reading signal. A twin
# of a TRAIN row just joins train. Twins are produced by stage 6
# (symbolize_twins.py) and arrive flagged; this script only routes them.
SYM_OF = {"eval_seen_tools": "eval_seen_sym",
          "eval_unseen_tools": "eval_unseen_sym"}


def bucket(s: str, mod: int = 100) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def signature(fn) -> str:
    """A function's identity: its parameter property names, sorted.

    Names are not identities in this corpus -- 379 of 875 carry more than one
    parameter schema and `search_movies` carries 63, because glaive invented
    each dialogue independently. The held-out split still hashes the NAME,
    which is the stronger guarantee (the model provably never saw the string),
    but `eval_seen_tools` has to be checked on the signature: otherwise a row
    can present a variant of `search_movies` the model never trained on while
    counting as a seen tool.
    """
    props = ((fn.get("parameters") or {}).get("properties") or {})
    return f"{fn.get('name')}({','.join(sorted(props))})"


def tool_signatures(row) -> list[str]:
    """Signatures in the CATALOGUE."""
    return [signature(t.get("function") or {})
            for t in row.get("tools", []) if isinstance(t, dict)
            and (t.get("function") or {}).get("name")]


def called_signatures(row) -> list[str]:
    """Signatures actually INVOKED, matched to the catalogue by name."""
    specs = {}
    for t in row.get("tools", []) or []:
        f = t.get("function") if isinstance(t, dict) else None
        if f and f.get("name"):
            specs[f["name"]] = f
    out = []
    for m in row.get("conversations", []):
        for tc in (m.get("tool_calls") or []):
            n = (tc.get("function") or {}).get("name")
            if n and n in specs:
                out.append(signature(specs[n]))
    return out


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


def card(counts, n_rejected, tool_stats, fails, splits=SPLITS, cfgs=None):
    rows = "\n".join(f"| `{s}` | {counts[s]:,} |" for s in splits)
    fl = "\n".join(f"| `{k}` | {v:,} |" for k, v in fails.most_common())
    # Only advertise configs that were actually pushed. A card naming
    # `en`/`rejected` when neither exists makes the dataset fail to load.
    cfgs = cfgs or ["default", "sft", "en", "rejected"]
    cfg_block = "\n".join(
        _cfg(c, {"default": "data"}.get(c, c),
             ["train"] if c == "rejected" else splits)
        for c in cfgs)
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
{cfg_block}
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
    ap.add_argument("--catalogue-size", type=int, default=0,
                    help="Shuffle each catalogue and pad it with distractor "
                         "tools to this size. 0 = source order, in which the "
                         "called tool is listed FIRST in 98.4%% of multi-tool "
                         "rows -- 'call tool #1' then scores 99.2%% right-tool "
                         "and selection is neither taught nor measured.")
    ap.add_argument("--catalogue-min", type=int, default=2,
                    help="Lower bound for the per-row catalogue size.")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="drop the reasoning prose before each tool call")
    ap.add_argument("--answers", type=Path, default=None,
                    help="answer cache from gen_tool_answer_turns.py; splices "
                         "result+answer onto dangling terminal calls")
    ap.add_argument("--abstention", type=Path, default=None,
                    help="jsonl of abstention rows, pushed as their own config. "
                         "Pushed HERE because this script rewrites the dataset "
                         "card, which de-registers any config it does not know "
                         "about -- a separate push of `abstention` silently "
                         "disappeared the next time the main configs went up.")
    ap.add_argument("--rendered", type=Path, default=None,
                    help="CONSUME stage 4/5 output (sft_answered.jsonl) "
                         "instead of re-rendering. This script used to "
                         "re-render from translated.jsonl, so every stage-4 "
                         "behaviour had to be threaded through twice -- which "
                         "is how the pedagogy filters came to apply to "
                         "sft.jsonl and to nothing that reached the Hub. "
                         "Joined on `idx`, which the renderer now emits.")
    ap.add_argument("--digest", action="store_true",
                    help="print a per-split SHA of the assembled messages. "
                         "Two invocations that differ only in --rendered MUST "
                         "print the same digests; that is the equivalence "
                         "check for consuming stage 4 rather than redoing it.")
    ap.add_argument("--keep-defective", action="store_true",
                    help="skip the pedagogy filters; reproduces v6 and "
                         "earlier, which shipped without them.")
    ap.add_argument("--private", action="store_true",
                    help="create the repo private. Set it at CREATE "
                         "time: flipping an existing public repo to "
                         "private later does not un-publish what has "
                         "already been fetched or cached.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_tool_answer_turns import cache_key, dangling  # noqa: E402
    from render_toolmind_sft import (to_messages, build_tool_pool,  # noqa: E402
                                     pedagogy_reject)

    recs = [json.loads(l) for l in (args.src / "translated.jsonl").open()
            if l.strip()]
    # gate_verdicts.jsonl is the TRANSLATION pipeline's record of which rows
    # its gate failed, and it keeps the failures in translated.jsonl so they
    # can be inspected. The procedural generator gates and judges inline and
    # rewrites translated.jsonl with only the survivors, so there is no
    # verdict file and nothing rejected to carry -- every row on disk is
    # already clean. Absent file therefore means "all clean", not "unknown".
    verdicts, gp = {}, args.src / "gate_verdicts.jsonl"
    if gp.exists():
        for line in gp.open():
            v = json.loads(line)
            verdicts[v["idx"]] = v["bad"]
        clean = [r for r in recs if not verdicts.get(r.get("idx"), ["unknown"])]
        rejected = [r for r in recs if verdicts.get(r.get("idx"), ["unknown"])]
    else:
        print(f"no gate_verdicts.jsonl in {args.src}: treating all "
              f"{len(recs):,} rows as gate-clean (procedural build)", flush=True)
        clean, rejected = list(recs), []
    fails = Counter(b.split("(")[0] for r in rejected
                    for b in verdicts.get(r["idx"], []))
    print(f"clean {len(clean):,}  rejected {len(rejected):,}", flush=True)

    # THIS SCRIPT RE-RENDERS; it does not consume stage 4's output. So the
    # pedagogy filters have to run here too, or the corpus that ships differs
    # from the corpus that was inspected -- the filters would apply to
    # sft.jsonl and to nothing that reaches the Hub. Imported from the
    # renderer rather than restated, so the two cannot drift.
    pool_recs = list(clean)             # pool built BEFORE filtering
    if not args.keep_defective:
        before, why = len(clean), Counter()
        kept = []
        for r in clean:
            reason = pedagogy_reject(r)
            if reason:
                why[reason] += 1
            else:
                kept.append(r)
        clean = kept
        print(f"pedagogy filters: {len(clean):,} of {before:,} kept, "
              f"dropped {dict(why)}", flush=True)

    names = sorted({n for r in clean for n in tool_names(r["da"]) if n})
    heldout = {n for n in names if bucket(n) < args.heldout_pct}
    print(f"tools: {len(names):,} distinct, {len(heldout):,} held out",
          flush=True)

    # `returns` arrives baked into the rows by the generator -- the renderer
    # and this script only assemble catalogues now.
    answers, n_answered = {}, 0
    if args.answers:
        for line in args.answers.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            answers[rec["k"]] = (rec["resultat"], rec["svar"])
        print(f"answer cache: {len(answers):,} generated turns", flush=True)
    pool = (build_tool_pool([r["da"] for r in pool_recs])
            if args.catalogue_size else [])
    if args.catalogue_size:
        print(f"distractor pool: {len(pool):,} non-held-out tools; catalogues "
              f"shuffled and padded to {args.catalogue_size}", flush=True)
    # `pre_rel` exists because the RENDERER is now a second source of this
    # label. ToolMind recovers it in the answer stage and it rides the
    # translated record; the procedural generator DECLARES it as
    # `_answer_field` and the renderer turns that into the same {at, fields}
    # shape. Keeping only `messages` here dropped the second kind on the
    # floor -- the field was written into sft.jsonl and never reached the
    # dataset, so `tool_answer` would have scored those rows against nothing.
    pre, twins, pre_rel, twin_rel = {}, {}, {}, {}
    if args.rendered:
        for line in args.rendered.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "idx" not in rec:
                continue
            if rec.get("sym"):
                twins[rec["idx"]] = rec["messages"]
                if rec.get("answer_relevance"):
                    twin_rel[rec["idx"]] = rec["answer_relevance"]
            else:
                pre[rec["idx"]] = rec["messages"]
                if rec.get("answer_relevance"):
                    pre_rel[rec["idx"]] = rec["answer_relevance"]
        print(f"consuming stage 4/5: {len(pre):,} rendered rows"
              + (f" + {len(twins):,} symbolized twins" if twins else "")
              + f" from {args.rendered.name}", flush=True)
        missing = [r["idx"] for r in clean if r["idx"] not in pre]
        if missing:
            print(f"  {len(missing):,} gate-clean rows absent from the "
                  f"rendered file (dropped downstream); skipping them",
                  flush=True)

    all_splits = SPLITS + (list(SYM_OF.values()) if True else [])
    data = {s: [] for s in all_splits}
    dropped = 0
    for _i, r in enumerate(clean):
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
        # r["idx"], not _i: the seed must be the row's source identity,
        # or filtering redraws every surviving row's catalogue.
        if args.rendered:
            msgs = pre.get(r["idx"])
            if msgs is None:
                continue            # filtered out downstream; not ours to keep
        else:
            msgs = to_messages(da, pool, r["idx"], args.catalogue_size,
                               args.catalogue_min,
                               reasoning=not args.no_reasoning)
            if msgs is None:
                continue
        if answers and not args.rendered:
            # Stage 5 already spliced these when --rendered is used;
            # re-splicing would duplicate the answer turn.
            # Splice the generated result + answer onto the row's dangling
            # terminal call. Rendered here rather than read from a pre-made
            # file because the catalogue is built in THIS loop -- the answer
            # is keyed on the call and the tool's returns contract, both of
            # which the row already carries, so the two stay in step.
            d = dangling({"messages": msgs})
            if d:
                at, call, spec, q = d
                got = answers.get(cache_key(call, q, spec))
                if got:
                    res, ans = got
                    msgs[at + 1:at + 1] = [
                        {"role": "tool_result",
                         "content": json.dumps(res, ensure_ascii=False)},
                        {"role": "assistant", "content": ans}]
                    n_answered += 1
        tw = twins.get(r["idx"])
        if tw is not None:
            data[SYM_OF.get(split, split)].append({
                "tools": da.get("tools", []),
                "conversations": da.get("conversations", []),
                "messages": tw,
                "en": r.get("orig"),
                "meta": {"idx": r["idx"], "n_tools": len(tn),
                         "n_turns": len(da.get("conversations", [])),
                         "tool_names": tn,
                         "tool_signatures": tool_signatures(da),
                         "called_signatures": called_signatures(da),
                         "symbolized": True,
                         # symbolize_twins remaps the field names through the
                         # twin's return-symbol map, so the twin carries its
                         # OWN relevance -- it is not the untwinned one.
                         "answer_relevance": twin_rel.get(r["idx"]) or []},
            })
        data[split].append({
            "tools": da.get("tools", []),
            "conversations": da.get("conversations", []),
            "messages": msgs,
            "en": r.get("orig"),
            "meta": {"idx": r["idx"], "n_tools": len(tn),
                     "n_turns": len(da.get("conversations", [])),
                     "tool_names": tn,
                     "tool_signatures": tool_signatures(da),
                     "called_signatures": called_signatures(da),
                     # which payload fields each generated answer was written
                     # to cite; the eval scores precision against these rather
                     # than re-deriving them from the reference text.
                     # Rendered first: for a declared label that is the only
                     # place it exists, and for a recovered one the two agree.
                     "answer_relevance": (pre_rel.get(r["idx"])
                                          or r.get("answer_relevance") or [])},
        })
    counts = {s: len(v) for s, v in data.items()}
    print("splits:", counts, f"(dropped {dropped:,} catalogue-only rows)",
          flush=True)
    if args.answers:
        print(f"attached a generated answer turn to {n_answered:,} rows",
              flush=True)

    # eval_seen must test tools the model HAS trained on. A rare tool whose
    # only rows landed in the eval sample would otherwise sit here untrained,
    # quietly making this bucket a second unseen-tool test.
    # Checked on SIGNATURES, not names. A row whose catalogue says
    # `search_movies` may be using a variant with different parameters that the
    # model never trained on -- that is an unseen function wearing a seen
    # label, and scoring it as a seen tool understates the split's difficulty.
    train_sigs = {x for r in data["train"]
                  for x in r["meta"]["tool_signatures"]}
    keep, moved = [], 0
    for r in data["eval_seen_tools"]:
        if all(x in train_sigs for x in r["meta"]["tool_signatures"]):
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

    if args.digest:
        import hashlib as _h
        print("--- per-split message digests ---", flush=True)
        for sp in SPLITS:
            blob = json.dumps([row["messages"] for row in data[sp]],
                              ensure_ascii=False, sort_keys=True)
            ans = sum(1 for row in data[sp]
                      for i, m in enumerate(row["messages"])
                      if m["role"] == "tool_result"
                      and i + 1 < len(row["messages"])
                      and row["messages"][i + 1]["role"] == "assistant")
            print(f"  {sp:<20} {len(data[sp]):6,} rows  {ans:6,} answer turns  "
                  f"sha256={_h.sha256(blob.encode()).hexdigest()[:16]}",
                  flush=True)

    if args.dry_run:
        print("\ndry run")
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True,
                    private=args.private)
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
    # A Danish-NATIVE corpus has no English source, so the `en` config would be
    # a column of nulls advertised in the card as provenance. Drop it rather
    # than ship it empty; same for `rejected`, which a pipeline that discards
    # its failures never populates.
    has_en = any(x.get("en") is not None for sp in all_splits for x in data[sp])
    if not has_en:
        views.pop("en")
        print("no English source on any row: skipping the `en` config",
              flush=True)
    live = [sp for sp in all_splits if data[sp]]
    for cfg, fn in views.items():
        for sp in live:
            Dataset.from_list(fn(data[sp])).push_to_hub(
                args.repo, config_name=cfg, split=sp,
                commit_message=f"{cfg}/{sp} ({counts[sp]} rows)")
            print(f"  pushed {cfg}/{sp} ({counts[sp]})", flush=True)
    if rejected:
        Dataset.from_list([{"en": J(r.get("orig")), "da": J(r["da"]),
                            "verdict": verdicts.get(r["idx"], []),
                            "idx": r["idx"]} for r in rejected]).push_to_hub(
            args.repo, config_name="rejected", split="train",
            commit_message=f"rejected ({len(rejected)} rows)")
        print(f"  pushed rejected ({len(rejected)})", flush=True)

    stats = {"n_tools": len(names), "n_heldout": len(heldout)}
    api.upload_file(
        path_or_fileobj=card(counts, len(rejected), stats, fails,
                             live, cfgs=list(views) + (["rejected"] if rejected
                                                       else [])).encode(),
        path_in_repo="README.md", repo_id=args.repo, repo_type="dataset",
        commit_message="dataset card")
    if args.abstention:
        # AFTER the card upload, not before. push_to_hub adds its config to the
        # dataset card's YAML; uploading our own README afterwards replaces that
        # card and de-registers the config, which is how `abstention` vanished
        # twice while the push log said it had been pushed.
        extra = [json.loads(l) for l in args.abstention.open() if l.strip()]
        Dataset.from_list([{"messages": r["messages"], "kind": r["kind"]}
                           for r in extra]).push_to_hub(
            args.repo, config_name="abstention", split="train",
            commit_message="abstention rows")
        print(f"  pushed abstention ({len(extra):,})", flush=True)

    print(f"-> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
