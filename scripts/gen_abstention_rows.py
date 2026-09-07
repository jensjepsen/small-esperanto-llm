"""Rows where the right move is NOT to answer.

Two gaps the probes found, neither of which the corpus teaches:

  A  the result does not contain what was asked for. Probed with a CVR-number
     absent from every passage, the model answered a different question with
     real facts instead of saying it was not there. Constructed by deleting the
     relevant field from a payload we already generated -- the ground truth is
     then a property of the payload, not of a reference answer.

  B  no tool in the catalogue can answer. Probed on four invented tools, the
     model called `dog_years` for a meeting-room question. Constructed by
     building the catalogue from distractors only, omitting the tool the
     question needs. A wrong call is worse than a missing answer.

The refusal text is GENERATED, not templated. A corpus where every refusal is
"Det fremgår ikke af resultatet" teaches the string rather than the behaviour --
the same trap as the filtered word-problem templates -- so the model is asked
for varied phrasing and the run reports how varied it actually came out.

Both types stay a minority of the mix. Over-trained refusal produces a model
that declines answerable questions, which is why the tool evals need a
"should have called" counter-metric before this is trained on at scale.
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tool_answer_turns import _cited, _leaves  # noqa: E402

MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"

# Danish markers that the reply declines rather than asserts. A starting set,
# not a definition: once there is volume, the honest version derives these from
# the accepted corpus. Kept small deliberately, and the planted controls below
# check it rejects a confident answer as well as accepting a refusal.
# Fields whose value is a constant, so "what was the status" has no content to
# be missing. The GRPO builder excludes the same set for the same reason.
BOILERPLATE = {"status", "message", "success", "result", "code", "error",
               "response", "info", "data"}

DECLINE = re.compile(
    r"\b(ikke|ingen|kan desværre|har ikke|mangler|fremgår ikke|"
    r"findes ikke|umuligt|ude af stand)\b", re.I)
CALLISH = re.compile(r"<\|tool_call\|>|\{\s*\"name\"\s*:", re.S)


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in (Path.home() / "or", Path.home() / ".or"):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("no OpenRouter key (~/or)")


SYS_A = """Du laver træningsdata til en dansk assistent, der bruger værktøjer.

Du får brugerens spørgsmål og værktøjets FULDE svar. Lav to ting:

1. "felt": navnet på det felt i svaret, som brugerens spørgsmål faktisk beder
   om. Præcis ét felt. Er der intet felt der besvarer spørgsmålet, så skriv "".

2. "svar": det assistenten ville sige, HVIS netop det felt manglede i
   værktøjets svar. Resten af felterne er der stadig.
   - Sig ligeud at oplysningen ikke er der.
   - Find ALDRIG på værdien. Gæt ikke, og regn den ikke ud.
   - Remse ikke de andre felter op som erstatning.
   - Sig ALDRIG at du kan se det felt, der mangler. Det er væk.
   - Sig ALDRIG at et af de FELTER, DER ER TILBAGE, mangler. De er der.
     ("Jeg kan se afstanden, men mangler enhederne" er forkert, når det er
      afstanden der mangler og enheden der er tilbage.)
   - Lov ikke at kunne levere det manglende senere.
   - Skriv ikke om JSON, kald eller parametre.
   - Ingen engelske ord.

Det er afgørende at "felt" er det felt spørgsmålet handler om. Vælger du et
tilfældigt felt, bliver svaret et afslag på et spørgsmål der sagtens kunne
besvares -- og det er værre end ingenting.

Du får FLERE opgaver på én gang. Svarene skal være indbyrdes forskellige --
forskellig åbning, forskellig sætningsbygning, forskellig længde."""

SYS_B = """Du laver træningsdata til en dansk assistent, der bruger værktøjer.

Brugeren beder om noget, som INGEN af de tilgængelige værktøjer kan klare.

Skriv assistentens svar på dansk, 1-2 sætninger:
- Sig at du ikke kan hjælpe med det med de værktøjer, du har.
- Kald IKKE et værktøj, og lad som om du gjorde det.
- Brug ikke et forkert værktøj, fordi det ligner. Et forkert kald er værre
  end intet svar.
- Nævn gerne kort hvad du faktisk kan, hvis noget er i nærheden.
- Du må gerne henvise til dine værktøjer -- det er naturligt her, når du
  forklarer at du mangler et. Skriv ikke om JSON, kald eller parametre.
- Ingen engelske ord.

Du får FLERE opgaver på én gang. Svarene skal være indbyrdes forskellige --
forskellig åbning, forskellig sætningsbygning, forskellig længde. Nogle korte
og afvisende, nogle der tilbyder et alternativ, nogle der spørger tilbage.
Begynd ikke to svar på samme måde."""


def gate_a(answer, removed_value, payload, question, field=""):
    # The user already said it. Removing `title` from a film lookup when the
    # question is "fortæl om filmen Inception" leaves a row that refuses to
    # supply something the user just typed -- and the rest of the payload
    # (year, director, genre, rating) still answers the question. A target
    # whose value appears in the question is not the thing being asked for.
    if removed_value is not None and question:
        v = str(removed_value).strip()
        if len(v) > 2 and v.lower() in question.lower():
            return "target-value-is-in-the-question"
    if not answer or not answer.strip():
        return "empty"
    a = answer.strip()
    if len(a.split()) > 60:
        return "too-long"
    if CALLISH.search(a):
        return "contains-a-call"
    if not DECLINE.search(a):
        return "does-not-decline"
    # Schema identifiers are not Danish prose. "Jeg kan ikke oprette en opgave,
    # da jeg ikke har et task_id" names the field instead of the thing, which
    # is the same register slip as talking about the tool call.
    for key in [field] + list(payload):
        if key and len(key) > 3 and re.search(rf"\b{re.escape(key)}\b", a, re.I):
            return f"names-a-schema-field:{key}"
    # The value was deleted from the payload, so any appearance is invented.
    if removed_value is not None and _cited(a, removed_value) \
            and str(removed_value) not in question:
        return "states-the-missing-value"
    # SYNONYM SURVIVORS. v5 contracts unioned several names for one concept --
    # current_price alongside data.stock_price, tip alongside tip_amount -- so
    # deleting one leaves the answer sitting in the payload under another name
    # and the row becomes a refusal to an answerable question. 4 of 6 sampled
    # rows failed this way. Checked on the VALUE, which is what makes the
    # question answerable, not on the field name.
    if removed_value is not None:
        for _p, v in _leaves(payload):
            if isinstance(v, bool) or v in (None, ""):
                continue
            if str(v).strip().lower() == str(removed_value).strip().lower():
                return "synonym-field-survives"
    return None


def gate_b(answer, gold_tool):
    if not answer or not answer.strip():
        return "empty"
    a = answer.strip()
    if len(a.split()) > 60:
        return "too-long"
    if CALLISH.search(a):
        return "contains-a-call"
    if not DECLINE.search(a):
        return "does-not-decline"
    if gold_tool and gold_tool.lower() in a.lower():
        return "names-the-missing-tool"
    return None


CONTROLS_A = [
    ("Der er 8 kopper tilbage.", 8, {"working": True}, "", "does-not-decline"),
    ("Det fremgår ikke, men der er 8 kopper tilbage.", 8, {}, "",
     "states-the-missing-value"),
    ("", None, {}, "", "empty"),
    ("Jeg kan ikke se det. <|tool_call|>{\"name\": \"x\"}", None, {}, "",
     "contains-a-call"),
]
CLEAN_A = [("Antallet af kopper fremgår ikke af resultatet.", 8,
            {"working": True}, "")]
CONTROLS_B = [
    ("Jeg slår det op for dig.", "get_weather", "does-not-decline"),
    ("Jeg kan ikke, men get_weather kunne måske.", "get_weather",
     "names-the-missing-tool"),
]
CLEAN_B = [("Det kan jeg desværre ikke hjælpe med her.", "get_weather")]


def check_controls():
    for ans, val, pay, q, want in CONTROLS_A:
        got = gate_a(ans, val, pay, q)
        if got is None or not got.startswith(want):
            raise SystemExit(f"A control {want!r} -> {got!r}")
    for ans, val, pay, q in CLEAN_A:
        if gate_a(ans, val, pay, q) is not None:
            raise SystemExit(f"A gate rejects clean: {gate_a(ans,val,pay,q)}")
    for ans, tool, want in CONTROLS_B:
        got = gate_b(ans, tool)
        if got is None or not got.startswith(want):
            raise SystemExit(f"B control {want!r} -> {got!r}")
    for ans, tool in CLEAN_B:
        if gate_b(ans, tool) is not None:
            raise SystemExit(f"B gate rejects clean: {gate_b(ans, tool)}")
    print(f"gate: {len(CONTROLS_A)+len(CONTROLS_B)} planted defects caught, "
          f"{len(CLEAN_A)+len(CLEAN_B)} clean pass", flush=True)


async def ask_batch(session, sys_prompt, users, tries=3):
    """One request, N tasks, N answers -- so the model can vary against itself.

    Per-call `variér formuleringen` does nothing: each request is independent,
    so the model has no idea it opened the last thirteen answers with `Jeg kan
    desværre`. Showing it the whole batch is what makes the instruction
    actionable.
    """
    numbered = "\n\n".join(f"OPGAVE {i+1}:\n{u}" for i, u in enumerate(users))
    body = {"model": MODEL, "temperature": 1.0, "max_tokens": 2000,
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": numbered}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "svar", "strict": True, "schema": {
                    "type": "object",
                    "properties": {"svar": {"type": "array",
                                            "items": {"type": "string"}}},
                    "required": ["svar"], "additionalProperties": False}}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                out = json.loads(d["choices"][0]["message"]["content"])["svar"]
                if len(out) == len(users):
                    return out
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None


async def ask_batch_json(session, sys_prompt, users, tries=3):
    """Batched, but each answer is an object rather than a string."""
    numbered = "\n\n".join(f"OPGAVE {i+1}:\n{u}" for i, u in enumerate(users))
    body = {"model": MODEL, "temperature": 1.0, "max_tokens": 2500,
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": numbered}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "svar", "strict": True, "schema": {
                    "type": "object",
                    "properties": {"svar": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"felt": {"type": "string"},
                                       "svar": {"type": "string"}},
                        "required": ["felt", "svar"],
                        "additionalProperties": False}}},
                    "required": ["svar"], "additionalProperties": False}}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                out = json.loads(d["choices"][0]["message"]["content"])["svar"]
                if len(out) == len(users):
                    return out
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None


def opening(text, n=3):
    return " ".join((text or "").split()[:n]).lower().strip(".,!?")


async def ask(session, sys_prompt, user, tries=3):
    body = {"model": MODEL, "temperature": 0.9, "max_tokens": 400,
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "svar", "strict": True, "schema": {
                    "type": "object", "properties": {"svar": {"type": "string"}},
                    "required": ["svar"], "additionalProperties": False}}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                return json.loads(d["choices"][0]["message"]["content"])["svar"]
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None


CATALOG_LABEL = "Værktøjer"


def load_local(path, limit, seed=0):
    """(catalogue, question, call, payload) from OBSERVED results in a corpus.

    Reads the real glaive payloads rather than ones we generated, so the
    synonym-survivor rate measured here is the source's, not ours.
    """
    # SAMPLE, do not take the head. glaive's early rows are a handful of
    # scenarios repeated -- taking the first N gave three NY->LA distance rows
    # and three USD->EUR rows out of fifteen, and that source monotony was
    # being read as generator monotony.
    lines = [l for l in Path(path).open() if l.strip()]
    random.Random(seed).shuffle(lines)
    out = []
    for line in lines:
        da = (json.loads(line).get("da") or {})
        conv = da.get("conversations") or []
        cat = [t.get("function") for t in da.get("tools") or []
               if isinstance(t, dict) and t.get("function")]
        if not cat:
            continue
        q = next((m.get("content") for m in conv
                  if m.get("role") == "user" and (m.get("content") or "").strip()),
                 None)
        if not q:
            continue
        last = None
        for i, m in enumerate(conv):
            for tc in (m.get("tool_calls") or []):
                last = tc.get("function")
            if m.get("role") != "tool" or not last:
                continue
            try:
                payload = json.loads(m.get("content") or "")
            except Exception:
                continue
            if isinstance(payload, dict) and len(payload) >= 2:
                out.append((cat, q, last, payload))
            break
        if len(out) >= limit:
            break
    return out


def _from_rows(ds, limit, seed=0):
    """(catalogue, question, call, payload) from rendered SFT rows.

    Type B needs a catalogue with distractors in it, and that only exists after
    rendering -- raw glaive rows mostly offer a single tool, which is why the
    local-corpus path rejected 60 of 60 as catalogue-too-small.
    """
    ds = list(ds)
    random.Random(seed).shuffle(ds)
    out = []
    for r in ds:
        ms = r["messages"]
        at = next((i for i, m in enumerate(ms)
                   if m["role"] == "tool_result"), None)
        if at is None or at + 1 >= len(ms) or ms[at + 1]["role"] != "assistant":
            continue
        cat_msg = next((m for m in ms
                        if (m.get("content") or "").startswith("Værktøjer:")),
                       None)
        if not cat_msg:
            continue
        try:
            head, q = cat_msg["content"].split("\n\n", 1)
            cat = json.loads(head.split("Værktøjer:", 1)[1].strip())
            payload = json.loads(ms[at]["content"])
            call = json.loads(ms[at - 1]["content"])
        except Exception:
            continue
        if not isinstance(payload, dict) or len(payload) < 2:
            continue
        out.append((cat, q, call, payload))
        if len(out) >= limit:
            break
    return out


def load_source(repo, split, limit, seed=0):
    from datasets import load_dataset
    return _from_rows(load_dataset(repo, "sft", split=split), limit, seed)


def load_rendered(path, limit, seed=0):
    return _from_rows((json.loads(l) for l in Path(path).open() if l.strip()),
                      limit, seed)


async def main_async(args):
    import aiohttp
    check_controls()
    src = (load_rendered(args.rows, args.n * 6, args.seed) if args.rows
           else load_local(args.local, args.n * 4, args.seed) if args.local
           else load_source(args.repo, args.split, args.n * 2, args.seed))
    print(f"{len(src):,} source rows", flush=True)
    rng = random.Random(args.seed)
    rows, reasons = [], Counter()
    sem = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}",
                     "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=300)) as s:

        async def make_a_batch(items):
            """A batch of A rows, with the relevant field IDENTIFIED first.

            The first version deleted a RANDOM field, which usually left the
            question answerable -- and the generator, told the field was gone,
            wrote a refusal anyway. All four sampled rows were refusals to
            questions the payload still answered. The field the question asks
            for has to be identified before anything is deleted.
            """
            users = [f"SPØRGSMÅL:\n{q}\n\nVÆRKTØJETS FULDE SVAR:\n"
                     f"{json.dumps(payload, ensure_ascii=False)}"
                     for _cat, q, _call, payload in items]
            async with sem:
                got = await ask_batch_json(s, SYS_A, users)
            if got is None:
                reasons["a:api"] += len(items)
                return
            for (cat, q, call, payload), o in zip(items, got):
                field = (o or {}).get("felt") or ""
                ans = (o or {}).get("svar") or ""
                if field not in payload:
                    reasons["a:field-not-in-payload"] += 1
                    continue
                if field.lower() in BOILERPLATE:
                    reasons["a:boilerplate-target"] += 1
                    continue
                thin = {k: v for k, v in payload.items() if k != field}
                if not thin:
                    reasons["a:payload-would-be-empty"] += 1
                    continue
                why = gate_a(ans, payload[field], thin, q, field)
                if why:
                    reasons[f"a:{why}"] += 1
                    continue
                op, w0 = opening(ans), opening(ans, 1)
                rows.append({"kind": "absent-field", "removed": field,
                             "messages": [
                    {"role": "user",
                     "content": f"{CATALOG_LABEL}:\n"
                                f"{json.dumps(cat, ensure_ascii=False)}\n\n{q}"},
                    {"role": "assistant", "content": ""},
                    {"role": "tool_call",
                     "content": json.dumps(call, ensure_ascii=False)},
                    {"role": "tool_result",
                     "content": json.dumps(thin, ensure_ascii=False)},
                    {"role": "assistant", "content": ans}]})

        seen_open, seen_first = Counter(), Counter()
        seen_open_a, seen_first_a = Counter(), Counter()
        seen_rows = set()

        async def make_b_batch(items):
            """A batch of B rows: one request, so the answers can differ."""
            prepared = []
            for cat, q, call, payload in items:
                gold = call.get("name")
                others = [t for t in cat if t.get("name") != gold]
                if len(others) < 2:
                    reasons["b:catalogue-too-small"] += 1
                    continue
                prepared.append((cat, q, gold, others))
            if not prepared:
                return
            users = [f"SPØRGSMÅL:\n{q}\n\nTILGÆNGELIGE VÆRKTØJER:\n"
                     f"{json.dumps([t.get('name') for t in others], ensure_ascii=False)}"
                     for _cat, q, _gold, others in prepared]
            async with sem:
                answers = await ask_batch(s, SYS_B, users)
            if answers is None:
                reasons["b:api"] += len(prepared)
                return
            for (cat, q, gold, others), ans in zip(prepared, answers):
                why = gate_b(ans, gold)
                if why:
                    reasons[f"b:{why}"] += 1
                    continue
                # DYNAMIC cap, not a banned-phrase list: no opening may take
                # more than a fifth of the corpus. Derived from what has been
                # accepted so far rather than from an enumeration of phrases we
                # happen to dislike.
                op, w0 = opening(ans), opening(ans, 1)
                rows.append({"kind": "no-capable-tool", "messages": [
                    {"role": "user",
                     "content": f"{CATALOG_LABEL}:\n"
                                f"{json.dumps(others, ensure_ascii=False)}"
                                f"\n\n{q}"},
                    {"role": "assistant", "content": ans}]})

        half = len(src) // 2
        bsrc = src[half:][:args.n]
        K = 8
        batches = [bsrc[i:i + K] for i in range(0, len(bsrc), K)]
        asrc = src[:half][:args.n]
        abatches = [asrc[i:i + K] for i in range(0, len(asrc), K)]
        await asyncio.gather(
            *[make_a_batch(b) for b in abatches],
            *[make_b_batch(b) for b in batches])

    # SUBSAMPLE rather than reject inline. An inline cap proportional to rows
    # already accepted cannot bootstrap -- it refuses everything until it has
    # something to be a fraction of -- and a cap proportional to the TARGET
    # stops binding at scale (200 on 1,040 rows let three openings take 58%).
    # Generating first and selecting after wastes no paid generation and gives
    # an exact distribution.
    def enforce(rs, share=0.10):
        """Cap each opening at a depth D, chosen so no opening exceeds `share`.

        Neither obvious cap works: a fraction of the INPUT overshoots (0.10 of
        1,580 candidates is 17% of the 940 that survive) and a fraction of
        accepted-so-far cannot bootstrap. Round-robin alone is not enough
        either -- take a prefix of it and the later cycles contain only the
        big groups. Solving for the depth is exact: keep at most D from every
        group, with D the largest value whose result still satisfies the share.
        """
        groups = {}
        for r in rs:
            groups.setdefault(opening(r["messages"][-1]["content"]), []).append(r)
        sizes = sorted((len(v) for v in groups.values()), reverse=True)
        best = 1
        for d in range(1, max(sizes) + 1):
            total = sum(min(n, d) for n in sizes)
            if min(sizes[0], d) / max(1, total) <= share:
                best = d
        return [r for v in groups.values() for r in v[:best]]

    before = len(rows)
    rows = [r for k in {x["kind"] for x in rows}
            for r in enforce([x for x in rows if x["kind"] == k])]
    print(f"opening cap: {before:,} -> {len(rows):,} rows", flush=True)

    kinds = Counter(r["kind"] for r in rows)
    print(f"\nkept {len(rows):,}  ({dict(kinds)})", flush=True)
    if reasons:
        print("rejected:")
        for w, c in reasons.most_common():
            print(f"  {w:<34} {c:,}")

    # diversity: a corpus of one phrase teaches the phrase
    for k in kinds:
        ans = [r["messages"][-1]["content"] for r in rows if r["kind"] == k]
        opens = Counter(" ".join(a.split()[:3]).lower() for a in ans)
        print(f"\n{k}: {len(ans)} answers, {len(opens)} distinct openings "
              f"({100*len(opens)/max(1,len(ans)):.0f}% unique)")
        for o, c in opens.most_common(3):
            print(f"    {c:>3}x  {o!r}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}", flush=True)
    for r in rows[:args.show]:
        print("=" * 74)
        for m in r["messages"]:
            print(f"  [{m['role']:<11}] {(m['content'] or '')[:220]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="jensjepsen/danish-tool-dialogues-v5")
    ap.add_argument("--split", default="train")
    ap.add_argument("--rows", type=Path, default=None,
                    help="a rendered sft jsonl: catalogues carry distractors, "
                         "which type B needs")
    ap.add_argument("--local", type=Path, default=None,
                    help="a translated.jsonl: use OBSERVED payloads instead of "
                         "the published (partly generated) ones")
    ap.add_argument("--n", type=int, default=20, help="per kind")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
