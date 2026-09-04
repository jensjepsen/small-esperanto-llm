"""Render the translated ToolMind rows into the trainer's chat format.

The `<think>` wrapper is DROPPED and its text becomes ordinary assistant
content. The tags are an artefact of how ToolMind was generated, not something
we want a 400M Danish model to emit: keeping them would make the model produce
a literal `<think>` string it was never taught to close reliably (16 of 6,486
assistant turns in the source are missing their closing tag), and would put
reasoning behind a marker that nothing downstream parses. Reasoning that flows
straight into a tool call is exactly what `format_conversation` already
encodes -- `<|assistant|> reasoning <|tool_call|>{...}<|/tool_call|> <|end|>`
as one autoregressive burst -- so the wrapper adds nothing.

Role mapping into the trainer's protocol:

    user            -> user            (catalogue prepended to the FIRST one)
    assistant text  -> assistant       (think tags stripped)
    assistant calls -> tool_call, one per call
    tool            -> tool_result     (masked from the loss by the trainer)

Usage:
  python scripts/render_toolmind_sft.py --in scratch/toolmind_da_v2 --n 5
  python scripts/render_toolmind_sft.py --in scratch/toolmind_da_v2 --out sft.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sft_packed import (  # noqa: E402
    ASSISTANT_TOKEN, SPECIAL_TOKENS, TOOL_CALL_OPEN, TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN, USER_TOKEN, _build_label_masker,
    _build_preprocess_and_tokenize, format_conversation,
)

# The translator rendered the tag in five different ways -- <think> 4,346,
# <taenk> 1,619, <tanke> 85, <taenke> 9, <taenker> 5 -- because it is inside
# text being translated. Match the family, not the literal. Tolerates unclosed
# tags too (16 rows have an opening tag and no close).
THINK = re.compile(r"</?\s*(?:think|thought|t\u00e6nk\w*|tanke\w*)\s*>", re.I)
# The per-row request numbers lines as "1. [kind] text", and the model echoes
# the hint back -- TRANSLATED, which the English-only stripper in the
# translator never caught: [svar] 1,408, [taenk] 1,219, plus variants. Strip a
# leading bracket tag; real assistant text does not open with one.
LEAD_TAG = re.compile(r"^\s*\[[^\]\n]{1,14}\]\s*")
CATALOG_LABEL = "Værktøjer"


def _strip_result_tags(content: str) -> str:
    """Remove leaked "[resultat]" tags from tool-result payloads.

    The translator echoed its own "[kind]" hint into 662 of 8,220 tool_result
    messages (8.05%), inside the JSON leaves rather than at the front of the
    string: {"status": "[resultat] succes", "data": {...}}. The assistant-turn
    strip never reached them because they are not at the start of the message.

    Tool results carry no loss, so this is input noise rather than a learned
    pattern -- but the model reads it, and it is trivially removable. Parsed
    and re-serialised leaf by leaf so a tag sitting mid-structure is caught and
    the JSON shape is preserved; non-JSON results fall back to a plain strip.
    """
    if not content:
        return content
    try:
        obj = json.loads(content)
    except Exception:
        return LEAD_TAG.sub("", content).strip()

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str):
            return LEAD_TAG.sub("", o).strip()
        return o
    return json.dumps(walk(obj), ensure_ascii=False)


def strip_think(text: str) -> str:
    out = LEAD_TAG.sub("", text or "")
    out = THINK.sub("", out)
    return LEAD_TAG.sub("", out).strip()   # tag can sit inside the think block


# ── catalogue realism ───────────────────────────────────────────────────────
#
# The source corpus lists the tool that gets called FIRST in 98.4% of
# multi-tool rows, and 69.7% of rows offer only one tool at all. So "call the
# first catalogued tool" scores 99.2% right-tool on the eval, beating the
# trained model's 84-93%: the ordering leaks the answer and there is almost
# never a choice to make.
#
# The model learns exactly that. Given a hand-built catalogue of four novel
# tools it called the FIRST one for every question -- dice-rolling for a query
# about coffee -- slot-filling whatever numbers appeared in the prompt. It also
# explains why eval_unseen scored ABOVE eval_seen across five measurements:
# unseen has more single-tool catalogues (74.0% vs 71.4%), i.e. it is easier,
# not better generalised.
#
# Two fixes, both render-time and free:
#   SHUFFLE   the catalogue per row, so position carries no signal
#   DISTRACT  pad it with unrelated tools, so selection is a real task
#
# Distractors are drawn ONLY from non-held-out tools. Injecting a held-out
# spec into a training catalogue would let the model read it during training,
# and eval_unseen_tools would stop meaning "never seen" -- the split's whole
# claim. The held-out rule is duplicated from push_tool_dialogues_hf.bucket()
# rather than imported, because the renderer runs before the split exists.
HELDOUT_PCT = 6


def _bucket(s: str, mod: int = 100) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def is_heldout_tool(name: str) -> bool:
    return _bucket(name or "") < HELDOUT_PCT


def build_tool_pool(rows) -> list[dict]:
    """Distinct non-held-out tool specs, for use as distractors."""
    pool, seen = [], set()
    for r in rows:
        for t in r.get("tools", []) or []:
            f = t.get("function") if isinstance(t, dict) else None
            if not f:
                continue
            n = f.get("name")
            if not n or n in seen or is_heldout_tool(n):
                continue
            seen.add(n)
            pool.append(f)
    return pool


def make_catalogue(tools, pool, idx, target, rng_seed=0, minimum=2):
    """Shuffled catalogue of `tools` padded with distractors.

    `target` is a MAXIMUM, not a fixed width: the size is drawn per row from
    [minimum, target]. A constant width is itself learnable -- the model can
    condition on "there are always six" -- and real catalogues vary, so a
    corpus that never varies teaches a habit that breaks at inference. It also
    keeps `gold is first` at chance-for-THAT-row rather than at one global
    constant, so no single positional prior fits the corpus.

    The floor is the number of genuine tools: padding never removes a real
    option, and rows whose source catalogue is already larger keep it.

    Deterministic in `idx` so a re-render reproduces the corpus exactly.
    """
    names = {t.get("name") for t in tools}
    rng = random.Random(rng_seed * 1_000_003 + idx)
    out = list(tools)
    if pool and target > len(out):
        want = max(len(out), minimum, rng.randint(minimum, target))
        cand = [f for f in pool if f.get("name") not in names]
        rng.shuffle(cand)
        out.extend(cand[:max(0, want - len(out))])
    rng.shuffle(out)
    return out


def to_messages(row, pool=None, idx=0, target=0, minimum=2) -> list[dict] | None:
    tools = [t.get("function") for t in row.get("tools", [])
             if isinstance(t, dict) and t.get("function")]
    if target:
        tools = make_catalogue(tools, pool or [], idx, target,
                               minimum=minimum)
    catalog = json.dumps(tools, ensure_ascii=False)
    msgs: list[dict] = []
    for m in row.get("conversations", []):
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            body = strip_think(content)
            if not msgs:                      # catalogue rides the first turn
                body = f"{CATALOG_LABEL}:\n{catalog}\n\n{body}"
            msgs.append({"role": "user", "content": body})
        elif role == "assistant":
            body = strip_think(content)
            calls = m.get("tool_calls") or []
            # ALWAYS open a model turn with <|assistant|>, even when there is
            # no reasoning to put in it. Every inference path prompts with
            # f"{USER}{q}{END}{ASST}", so the model only ever generates after
            # an <|assistant|> marker. Emitting a bare <|tool_call|> straight
            # after <|user|> -- which 1,797 of 7,834 calls did, 22.9% -- trains
            # a context that never occurs at inference.
            if body or calls:
                msgs.append({"role": "assistant", "content": body})
            for tc in calls:
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                msgs.append({"role": "tool_call",
                             "content": json.dumps(fn, ensure_ascii=False)})
        elif role == "tool":
            msgs.append({"role": "tool_result",
                         "content": _strip_result_tags(content)})
        else:
            return None                       # unknown role: drop the row
    if not any(m["role"] in ("assistant", "tool_call") for m in msgs):
        return None                           # nothing for the model to learn
    if msgs and msgs[0]["role"] != "user":
        return None                           # must open with the catalogue
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path,
                    default=Path("scratch/toolmind_da_v2"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n", type=int, default=0, help="print N rendered rows")
    ap.add_argument("--catalogue-size", type=int, default=0,
                    help="MAX catalogue size; the actual size is drawn per "
                         "row from [--catalogue-min, this] and the catalogue is "
                         "SHUFFLED. 0 = leave as-is (the source "
                         "lists the called tool first in 98.4%% of multi-tool "
                         "rows, so position leaks the answer and 'call tool #1' "
                         "scores 99.2%% right-tool).")
    ap.add_argument("--catalogue-min", type=int, default=2,
                    help="Lower bound for the per-row catalogue size.")
    ap.add_argument("--clean-only", action="store_true",
                    help="render only rows the gate passed, read from "
                         "gate_verdicts.jsonl. Without it EVERY translated row "
                         "is rendered, failures included.")
    ap.add_argument("--tokenizer",
                    default="jensjepsen/danish-lm-400m-sft-v34-mid")
    ap.add_argument("--subfolder", default="step-30240-agg-0.264")
    args = ap.parse_args()

    recs = [json.loads(l) for l in (args.src / "translated.jsonl").open()
            if l.strip()]
    if args.clean_only:
        vp = args.src / "gate_verdicts.jsonl"
        if not vp.exists():
            raise SystemExit(f"--clean-only needs {vp}; run --gate-only first")
        verdicts = {}
        for line in vp.open():
            v = json.loads(line)
            verdicts[v["idx"]] = v["bad"]
        before = len(recs)
        recs = [r for r in recs if not verdicts.get(r.get("idx"), ["unknown"])]
        print(f"clean-only: {len(recs):,} of {before:,} rows passed the gate",
              flush=True)
    rows = [r["da"] for r in recs]
    print(f"loaded {len(rows):,} translated rows", flush=True)

    pool = build_tool_pool(rows) if args.catalogue_size else []
    if args.catalogue_size:
        print(f"distractor pool: {len(pool):,} distinct non-held-out tools; "
              f"padding catalogues to {args.catalogue_size} and shuffling",
              flush=True)
    rendered, drops = [], Counter()
    for i, r in enumerate(rows):
        m = to_messages(r, pool, i, args.catalogue_size,
                        args.catalogue_min)
        if m is None:
            drops["unrenderable"] += 1
            continue
        rendered.append({"messages": m})
    print(f"rendered {len(rendered):,}  dropped {dict(drops) or 0}", flush=True)

    # VERIFY AGAINST THE TRAINER, not against an idea of it. Every row must
    # tokenise, produce a label mask, and train on something -- and no user or
    # tool-result text may appear among the trained tokens.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                        subfolder=args.subfolder)
    unk = tok.unk_token_id
    tid = tok.convert_tokens_to_ids
    mask = _build_label_masker(tid(ASSISTANT_TOKEN), tid(TOOL_RESULT_OPEN),
                               tid(TOOL_RESULT_CLOSE), unk,
                               user_id=tid(USER_TOKEN),
                               tool_call_id=tid(TOOL_CALL_OPEN))
    tk = _build_preprocess_and_tokenize(tok, SPECIAL_TOKENS, 8048,
                                        morpheme_preprocess=False)
    bad = Counter()
    check = rendered[:400]
    for row in check:
        text = format_conversation(row["messages"])
        ids = tk(text)["input_ids"]
        labels = mask(ids)
        if labels is None:
            bad["no-model-turn"] += 1
            continue
        if all(l == -100 for l in labels):
            bad["nothing-trained"] += 1
            continue
        # POSITIONAL, not textual. Checking whether the user's words appear
        # among the trained tokens flags correct behaviour: the model quotes
        # the user in its reasoning ("brugeren spurgte om...") and copies the
        # user's phrasing into arguments ({"content": "Jeg vil gerne..."}).
        # Both are exactly what it should learn. The real invariant is that
        # the <|user|> and <|tool_result|> SPANS carry no loss.
        u_id, tr_id = tid(USER_TOKEN), tid(TOOL_RESULT_OPEN)
        a_id, tc_id = tid(ASSISTANT_TOKEN), tid(TOOL_CALL_OPEN)
        in_world = False
        for t, l in zip(ids, labels):
            if t in (a_id, tc_id):
                in_world = False
            elif t in (u_id, tr_id):
                in_world = True
            if in_world and l != -100:
                bad["world-span-trained"] += 1
                break
        if "<think>" in text or "</think>" in text:
            bad["think-tag-survived"] += 1
    print(f"\ntrainer check over {len(check)} rows: "
          f"{'CLEAN' if not bad else dict(bad)}")

    if args.n:
        for row in rendered[:args.n]:
            print("\n" + "=" * 72)
            for m in row["messages"]:
                print(f"  [{m['role']:<11}] {m['content'][:220]}")

    if args.out:
        with args.out.open("w") as f:
            for row in rendered:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n-> {args.out}  ({len(rendered):,} rows)")


if __name__ == "__main__":
    main()
