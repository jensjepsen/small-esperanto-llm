"""Flatten da_sci_reasoning_v1 + da_fact_check_v1 into SFT rows.

Sources:
  - data/da_sci_reasoning_v1/rows.jsonl — per article:
      {pageid, title, text, items: {worked_calc, mechanism, counterfactual}}
  - data/da_fact_check_v1/rows.jsonl — per article:
      {pageid, title, text, claims: [{claim, verdict, reasoning}]}

Emitted subtypes (all in one output stream, tagged):
  * stem_worked_calc_open / _closed
  * stem_mechanism_open / _closed
  * stem_counterfactual_open / _closed
  * stem_fact_check_open / _closed

Per source item we emit BOTH open and closed views. Prompt composition
sampled per row from prompt_templates.json:
    open  : wrapper_open.format(title, text, TASK=task_template.format(**fields))
    closed: wrapper_closed.format(TASK=task_template.format(**fields))

Output row schema:
    {messages: [{role, content}], subtype, view, pageid, passage_len}

Length filter: drop rows over --max-tokens using the DA tokenizer.
"""
from __future__ import annotations
import argparse
import json
import random
import re
from pathlib import Path

# Qs that reference the article ("ifølge artiklen", "artiklen nævner", etc.)
# don't make sense in closed view (no article present). Skip closed for these.
ARTICLE_REF_RE = re.compile(
    r"\b(artiklen|artiklens|artikelens|ifølge artiklen|i artiklen|"
    r"teksten|i teksten)\b", re.I)

# Natural-language task descriptions. Singular with article for A/D
# ("lav {desc}"); bare plural for B ("lav 5 forskellige {desc}").
TASK_DESC_SG = {
    "worked_calc":    "et regnestykke med trin-for-trin løsning",
    "mechanism":      "et spørgsmål om hvordan noget virker, med trinvis forklaring",
    "counterfactual": "et 'hvad hvis'-spørgsmål med årsagsforklaring",
    "factcheck":      "en påstand med sandhedsvurdering og begrundelse",
}
TASK_DESC_PL = {
    "worked_calc":    "regnestykker med trin-for-trin løsning",
    "mechanism":      "spørgsmål om hvordan noget virker, med trinvis forklaring",
    "counterfactual": "'hvad hvis'-spørgsmål med årsagsforklaring",
    "factcheck":      "påstande med sandhedsvurdering og begrundelse",
}


def build_prompt(templates, view, title, text, task_pool, task_kwargs, rng):
    """Compose the user prompt.

    task_pool=None → use the raw field (q/claim) verbatim as TASK.
    Otherwise sample a task template and .format() with task_kwargs.

    Closed view → no wrapper (bare TASK — that's the natural real-user
    prompt). Open view → sample a wrapper_open template.
    """
    if task_pool is None:
        task_str = next(iter(task_kwargs.values()))
    else:
        task_tpl = rng.choice(templates[task_pool])
        task_str = task_tpl.format(**task_kwargs) if task_kwargs else task_tpl
    if view == "open":
        wrapper = rng.choice(templates["wrapper_open"])
        return wrapper.format(text=text, TASK=task_str)
    else:
        return task_str


def sci_reasoning_rows(raw, templates, rng):
    it = raw["items"]
    # No task template for sci_* — q is already a full standalone question.
    for key, task_pool in [
        ("worked_calc", None),
        ("mechanism", None),
        ("counterfactual", None),
    ]:
        items = it.get(key) or []
        for item in items:
            q = item.get("q"); a = item.get("a")
            if not (isinstance(q, str) and isinstance(a, str)): continue
            # Answers may be lists (JSON leak) — join into prose
            if a.startswith("[") and a.endswith("]"):
                try:
                    parsed = json.loads(a)
                    if isinstance(parsed, list):
                        a = "\n".join(str(x) for x in parsed)
                except Exception:
                    pass
            # Skip closed view if Q references the article/text.
            views = ("open", "closed") if not ARTICLE_REF_RE.search(q) else ("open",)
            for view in views:
                subtype = f"stem_{key}_{view}"
                prompt = build_prompt(templates, view, raw["title"], raw["text"],
                                      task_pool, {"q": q}, rng)
                yield {
                    "messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": a}],
                    "subtype": subtype,
                    "view": view,
                    "pageid": raw["pageid"],
                    "passage_len": raw["text_len"],
                }


def _valid_qa(item):
    q = item.get("q"); a = item.get("a")
    if not (isinstance(q, str) and isinstance(a, str)): return None
    if a.startswith("[") and a.endswith("]"):
        try:
            parsed = json.loads(a)
            if isinstance(parsed, list): a = "\n".join(str(x) for x in parsed)
        except Exception: pass
    return q, a


def rev_rows(raw, templates, rng):
    """Reverse-direction: model generates Qs/claims from an article.
    Emits A (single Q+A), B (batched Q+A per subtype), C (balanced fact-check
    batch), D (Q only) variants.
    """
    text = raw["text"]
    pageid = raw["pageid"]
    passage_len = raw["text_len"]
    items_by_key = raw.get("items", {})  # sci_reasoning source
    claims = raw.get("claims", [])       # fact_check source

    # A: single Q+A per item (sci) or per claim (fact_check)
    for key in ("worked_calc", "mechanism", "counterfactual"):
        for item in items_by_key.get(key) or []:
            qa = _valid_qa(item)
            if not qa: continue
            q, a = qa
            tpl = rng.choice(templates["rev_A_single_qa"])
            prompt = tpl.format(task_desc=TASK_DESC_SG[key], text=text)
            answer = f"SPØRGSMÅL: {q}\n\nSVAR: {a}"
            yield {
                "messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": answer}],
                "subtype": f"stem_gen_single_{key}",
                "view": "gen",
                "pageid": pageid, "passage_len": passage_len,
            }
    for cl in claims:
        c = cl.get("claim"); v = cl.get("verdict"); why = cl.get("reasoning")
        if not (isinstance(c, str) and v in ("SAND", "FALSK") and isinstance(why, str)):
            continue
        tpl = rng.choice(templates["rev_A_single_qa"])
        prompt = tpl.format(task_desc=TASK_DESC_SG["factcheck"], text=text)
        answer = f"PÅSTAND: {c}\n\nVURDERING: {v}. {why}"
        yield {
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}],
            "subtype": "stem_gen_single_factcheck",
            "view": "gen",
            "pageid": pageid, "passage_len": passage_len,
        }

    # B: batched multi Q+A per sci subtype
    for key in ("worked_calc", "mechanism", "counterfactual"):
        items = [_valid_qa(it) for it in (items_by_key.get(key) or [])]
        items = [x for x in items if x]
        if len(items) < 2: continue  # need ≥2 for meaningful batch
        n = len(items)
        tpl = rng.choice(templates["rev_B_multi_qa"])
        prompt = tpl.format(n=n, task_desc=TASK_DESC_PL[key], text=text)
        answer = "\n\n".join(f"{i+1}. SPØRGSMÅL: {q}\n   SVAR: {a}"
                              for i, (q, a) in enumerate(items))
        yield {
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}],
            "subtype": f"stem_gen_multi_{key}",
            "view": "gen",
            "pageid": pageid, "passage_len": passage_len,
        }

    # C: balanced fact-check batch (n sand + n falsk, mixed)
    sand = [c for c in claims if c.get("verdict") == "SAND"
            and isinstance(c.get("claim"), str) and isinstance(c.get("reasoning"), str)]
    falsk = [c for c in claims if c.get("verdict") == "FALSK"
             and isinstance(c.get("claim"), str) and isinstance(c.get("reasoning"), str)]
    n_each = min(len(sand), len(falsk))
    if n_each >= 1:
        pick_s = rng.sample(sand, min(n_each, len(sand)))
        pick_f = rng.sample(falsk, min(n_each, len(falsk)))
        mixed = pick_s + pick_f
        rng.shuffle(mixed)
        tpl = rng.choice(templates["rev_C_factcheck_balanced"])
        prompt = tpl.format(n=n_each, text=text)
        answer = "\n\n".join(
            f"{i+1}. {c['verdict']}: {c['claim']}\n   Begrundelse: {c['reasoning']}"
            for i, c in enumerate(mixed))
        yield {
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}],
            "subtype": "stem_gen_factcheck_balanced",
            "view": "gen",
            "pageid": pageid, "passage_len": passage_len,
        }

    # D: single Q only (no answer)
    for key in ("worked_calc", "mechanism", "counterfactual"):
        for item in items_by_key.get(key) or []:
            qa = _valid_qa(item)
            if not qa: continue
            q, _ = qa
            tpl = rng.choice(templates["rev_D_single_q_only"])
            prompt = tpl.format(task_desc=TASK_DESC_SG[key], text=text)
            yield {
                "messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": q}],
                "subtype": f"stem_gen_qonly_{key}",
                "view": "gen",
                "pageid": pageid, "passage_len": passage_len,
            }
    for cl in claims:
        c = cl.get("claim"); v = cl.get("verdict")
        if not (isinstance(c, str) and v in ("SAND", "FALSK")): continue
        tpl = rng.choice(templates["rev_D_single_q_only"])
        prompt = tpl.format(task_desc=TASK_DESC_SG["factcheck"], text=text)
        yield {
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": c}],
            "subtype": "stem_gen_qonly_factcheck",
            "view": "gen",
            "pageid": pageid, "passage_len": passage_len,
        }


def fact_check_rows(raw, templates, rng):
    for claim in raw.get("claims", []):
        c = claim.get("claim"); v = claim.get("verdict"); why = claim.get("reasoning")
        if not (isinstance(c, str) and v in ("SAND", "FALSK") and isinstance(why, str)):
            continue
        answer = f"{v}. {why}"
        views = ("open", "closed") if not ARTICLE_REF_RE.search(c) else ("open",)
        for view in views:
            subtype = f"stem_fact_check_{view}"
            prompt = build_prompt(templates, view, raw["title"], raw["text"],
                                  "stem_fact_check", {"claim": c}, rng)
            yield {
                "messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": answer}],
                "subtype": subtype,
                "view": view,
                "pageid": raw["pageid"],
                "passage_len": raw["text_len"],
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sci-data", type=Path,
                    default=Path("data/da_sci_reasoning_v1/rows.jsonl"))
    ap.add_argument("--fc-data", type=Path,
                    default=Path("data/da_fact_check_v1/rows.jsonl"))
    ap.add_argument("--templates", type=Path,
                    default=Path("data/da_stem_v1/prompt_templates.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/da_stem_v1/sft.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="Drop rows above this. 0 disables.")
    ap.add_argument("--tokenizer", default="jensjepsen/danish-tokenizer")
    args = ap.parse_args()

    templates = json.loads(args.templates.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tok = None
    if args.max_tokens > 0:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer for length filter...", flush=True)
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    counts = {}
    dropped = {}
    n_total = 0

    with args.out.open("w") as out:
        for src_name, path, builders in [
            ("sci", args.sci_data, [sci_reasoning_rows, rev_rows]),
            ("fc",  args.fc_data,  [fact_check_rows, rev_rows]),
        ]:
            if not path.exists():
                print(f"  SKIP {src_name}: no {path}"); continue
            print(f"processing {src_name} from {path}...", flush=True)
            for line in path.open():
                raw = json.loads(line)
                if raw.get("reject"): continue
                for builder in builders:
                    # Distinct RNG per (src, builder, pageid) so rev vs
                    # forward pick independent wrappers/templates.
                    rng = random.Random(f"{args.seed}:{src_name}:{builder.__name__}:{raw['pageid']}")
                    for row in builder(raw, templates, rng):
                        if tok is not None:
                            n_tok = (len(tok(row["messages"][0]["content"]).input_ids)
                                     + len(tok(row["messages"][1]["content"]).input_ids))
                            if n_tok > args.max_tokens:
                                dropped[row["subtype"]] = dropped.get(row["subtype"], 0) + 1
                                continue
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts[row["subtype"]] = counts.get(row["subtype"], 0) + 1
                        n_total += 1

    print(f"\nTotal: {n_total:,} SFT rows → {args.out}")
    for st in sorted(counts):
        d = dropped.get(st, 0)
        drop_pct = 100*d/(counts[st]+d) if counts[st]+d else 0
        print(f"  {st:32s} {counts[st]:>7,}  (dropped {d}, {drop_pct:.1f}%)")


if __name__ == "__main__":
    main()
