"""Scaffold: generate ratio word problems in Esperanto via Gemini Flash Lite.

Pipeline:
  generate batch -> parse JSON -> verify arithmetic -> dedup -> append JSONL

Two cheap verifier checks per problem:
  1. Every `LHS = RHS` line in chain_eo must hold under sandboxed eval.
  2. The number in the chain's final line must match the JSON `answer`.

Diversity gate: hash the equation skeleton (numbers stripped, names stripped) and
reject if seen >= MAX_DUP_PER_SKELETON times.

Usage:
  GOOGLE_API_KEY=... uv run python scripts/generate_ratio_problems.py \\
    --n 100 --out data/word_problems/ratios.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Pull names + objects from the ontology so we get real EO coverage instead of
# the model defaulting to "Anna kaj Bert dividas bombonojn" forever.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_NAMES_FILE = PROJECT_ROOT / "src/esperanto_lm/ontology/sampler.py"
_CONCEPTS = PROJECT_ROOT / "src/esperanto_lm/ontology/data/concepts.jsonl"

# Body parts and other non-divisible items to exclude from "things people divide".
_BAD_OBJECTS = {
    "brako", "dento", "dorso", "fingro", "kapo", "kolo", "korpo", "mano",
    "okulo", "orelo", "piedo", "ventro", "vosto", "ŝultro", "buŝo", "haŭto",
    "nazo", "lipo", "lango", "frunto", "mentono", "trunko", "kruro",
    "genuo", "kubuto", "ostoj", "sango", "haro", "ungo", "muskolo",
    "cerbo", "koro", "pulmo", "stomako", "rumpa", "hepato",
    # weather/sky: also poor fit for "divide between people"
    "ĉielo", "ĉielarko", "aŭroro", "fajro", "flako", "vento", "vojo",
    "duno", "vulkano", "ŝtuparo",
}


def load_names() -> list[str]:
    """Pull PERSON_NAMES from sampler.py + capitalize."""
    src = _NAMES_FILE.read_text()
    m = re.search(r"PERSON_NAMES\s*=\s*\[(.*?)\]", src, re.DOTALL)
    names = re.findall(r'"([a-zćĉĝĥĵŝŭ]+)"', m.group(1)) if m else []
    return [n.capitalize() for n in names]


def load_objects() -> list[str]:
    """Pull countable EO objects from concepts.jsonl, sans body parts."""
    out = []
    for line in _CONCEPTS.open():
        r = json.loads(line)
        et = r.get("entity_type", "")
        if isinstance(et, list):
            et = et[0] if et else ""
        lem = r.get("lemma", "")
        if (et in ("artifact", "natural_object", "inanimate")
                and lem.endswith("o")
                and 3 <= len(lem) <= 12
                and "-" not in lem and " " not in lem
                and lem not in _BAD_OBJECTS):
            out.append(lem)
    return sorted(set(out))


PERSON_NAMES = load_names()
OBJECT_POOL = load_objects()

# Per-strategy solver examples. Each batch is locked to ONE strategy so the
# model doesn't default to the most familiar one (parts-method) every time.
# Verifier accepts any chain whose LHS=RHS equalities check and whose final
# number matches the JSON answer, so any strategy is fine as long as math holds.
STRATEGY_EXAMPLES = {
    "parts": """STRATEGIO: dividu la totalon en partojn.
Ekzemplo:
  "ni dividu en 5 partojn (2+3=5).
  unu parto = 30 / 5 = 6.
  bert ricevas 3 partojn: 3 * 6 = 18.
  #### 18"
""",
    "algebra": """STRATEGIO: starigu algebran ekvacion kun variabla parto.
Ekzemplo:
  "estu x la valoro de unu parto.
  do anna ricevas 2x kaj bert ricevas 3x.
  la totalo: 2x + 3x = 30.
  do 5x = 30.
  x = 30 / 5 = 6.
  bert ricevas: 3x = 3 * 6 = 18.
  #### 18"
""",
    "fraction": """STRATEGIO: esprimu kiel frakcion de la tuto.
Ekzemplo:
  "la totalo de la proporciopartoj: 2 + 3 = 5.
  bert ricevas frakcion 3/5 de la totalo.
  do bert ricevas: 3 / 5 * 30 = 18.
  bert ricevas: 18.
  #### 18"
""",
    "diff": """STRATEGIO: kalkulu unue ambaŭ partojn, poste la DIFERENCON aŭ SUMON laŭ demando.
Ekzemplo (demando: kiom pli ricevas bert ol anna?):
  "ni dividu en 5 partojn (2+3=5).
  unu parto = 30 / 5 = 6.
  anna ricevas: 2 * 6 = 12.
  bert ricevas: 3 * 6 = 18.
  diferenco: 18 - 12 = 6.
  #### 6"
""",
}

# Per-framing examples. Each batch also picks a question framing so we don't
# only ever ask "how many does Z get?".
FRAMING_HINTS = {
    "direct": "Demandu kiom da aĵoj ricevas unu specifa persono.",
    "larger": "Demandu kiom da aĵoj ricevas la persono kun la PLI GRANDA parto.",
    "smaller": "Demandu kiom da aĵoj ricevas la persono kun la PLI MALGRANDA parto.",
    "diff": "Demandu kiom PLI da aĵoj havas unu ol la alia (diferenco).",
    "given-one": ("Sciigu ke unu persono ricevis X aĵojn, kaj demandu la "
                  "TOTALON aŭ kiom ricevis la alia."),
    "context": ("Vortumu kiel rakonton: lernejo, restoracio, festo, vendejo, "
                "familio, klubo. Pasinta tempo. Konkretaj detaloj."),
}


_RATIOS = ["1:2", "2:3", "3:4", "1:3", "2:5", "3:5", "4:5", "1:4",
           "3:7", "1:2:3", "2:3:5", "1:1:2", "1:3:4"]


def build_prompt(n: int, strategy: str, framing: str, rng: random.Random) -> str:
    # Pre-sample n names and n objects so the model is FORCED to use diverse
    # surface forms across problems instead of defaulting to Anna/Bert/bombonoj.
    names_pool = rng.sample(PERSON_NAMES, min(2 * n + 2, len(PERSON_NAMES)))
    objs_pool = rng.sample(OBJECT_POOL, min(n + 4, len(OBJECT_POOL)))
    ratios_pool = rng.sample(_RATIOS, min(n + 2, len(_RATIOS)))
    return f"""Generu {n} esperantajn matematikajn problemojn pri proporcio (ratio).
Ĉiu problemo dividas entjeran kvanton laŭ donita proporcio.

DEVIGE uzu personojn EL ĈI TIU LISTO (varias por ĉiu problemo):
  {", ".join(names_pool)}

DEVIGE uzu aĵojn EL ĈI TIU LISTO (varias por ĉiu problemo):
  {", ".join(objs_pool)}

Proporcioj el: {", ".join(ratios_pool)}
Totaloj: entjeroj 10–300, divideblaj de la sumo de la proporcio.

KADRO POR LA DEMANDOJ: {FRAMING_HINTS[framing]}

{STRATEGY_EXAMPLES[strategy]}
Por ĉiu problemo, redonu JSON-objekton kun:
- "type": "ratio"
- "question_eo": la problemo (1–3 frazoj, varia stilo, ĝusta esperanta gramatiko)
- "chain_eo": solvo laŭ la supra STRATEGIO. ĈIU aritmetika paŝo sur PROPRA LINIO kun `=`. Fina linio devas esti "#### N".
- "answer": la fina nombro (entjero)

Respondu NUR JSON-listo de {n} objektoj, sen ```markdown, sen klariga teksto.
"""


# Sandbox arithmetic eval: only +, -, *, /, parens, digits, decimal points.
_SAFE_EXPR = re.compile(r"^[\d\s+\-*/().]+$")


def safe_eval(expr: str) -> float | None:
    expr = expr.strip()
    if not _SAFE_EXPR.match(expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


# Pull every numeric `LHS = RHS` line out of a chain. Handles `LHS = MID = RHS`
# by collapsing into pairs: (LHS, MID), (MID, RHS).
_EQ_LINE = re.compile(r"([\d\s+\-*/().]+(?:\s*=\s*[\d\s+\-*/().]+)+)")
_FINAL_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
_TRAILING_NUM = re.compile(r"(-?\d+(?:\.\d+)?)[^\d]*$")


def verify_chain(chain: str, answer) -> tuple[bool, str]:
    """Returns (ok, reason)."""
    # 1) every stated arithmetic equality must hold
    for match in _EQ_LINE.finditer(chain):
        parts = [p.strip() for p in match.group(1).split("=")]
        # need at least one operator on the LHS for it to count as a claim
        if not re.search(r"[+\-*/]", parts[0]):
            continue
        for i in range(len(parts) - 1):
            lhs = safe_eval(parts[i])
            rhs = safe_eval(parts[i + 1])
            if lhs is None or rhs is None:
                continue  # unparseable side, skip rather than fail (EO words mixed in)
            if abs(lhs - rhs) > 1e-6:
                return False, f"arith-mismatch: {parts[i]} != {parts[i+1]}"

    # 2) JSON answer must match the final number in the chain
    m = _FINAL_HASH.search(chain)
    if not m:
        m = _TRAILING_NUM.search(chain.strip())
    if not m:
        return False, "no-final-number-in-chain"
    chain_final = float(m.group(1))
    try:
        ans = float(answer)
    except (TypeError, ValueError):
        return False, f"non-numeric-answer: {answer!r}"
    if abs(chain_final - ans) > 1e-6:
        return False, f"chain-vs-json-mismatch: {chain_final} != {ans}"
    return True, ""


# Diversity: strip names, numbers, and aĵo-words; hash what's left.
_NAMES = ("anna", "bert", "petro", "maria", "helena", "hugo", "sara",
          "lukas", "eva", "ema")
_ITEMS = ("bombonoj", "pomoj", "libroj", "moneroj", "biskvitoj", "ovoj",
          "krajonoj", "glasoj", "ŝtonoj")


def skeleton(question: str) -> str:
    s = question.lower()
    for w in _NAMES + _ITEMS:
        s = s.replace(w, "<X>")
    s = re.sub(r"\d+", "<N>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_response(text: str) -> list[dict]:
    """Strip ```json fences and JSON-parse the response."""
    text = text.strip()
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if chunk.startswith("json"):
                text = chunk[4:].strip()
                break
            if chunk.startswith("["):
                text = chunk
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to recover by finding outermost [...]
        i = text.find("[")
        j = text.rfind("]")
        if i >= 0 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                pass
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="target accepted problems")
    ap.add_argument("--batch-size", type=int, default=5,
                    help="problems per Gemini call")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    ap.add_argument("--max-dup", type=int, default=3,
                    help="reject if skeleton seen this many times")
    ap.add_argument("--max-calls", type=int, default=0,
                    help="hard cap on API calls (0 = unlimited)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--strategies", default="parts,algebra,fraction,diff",
                    help="comma-list, rotated per call")
    ap.add_argument("--framings", default="direct,larger,smaller,diff,given-one,context",
                    help="comma-list, rotated per call")
    args = ap.parse_args()
    strategies = args.strategies.split(",")
    framings = args.framings.split(",")
    bad = [s for s in strategies if s not in STRATEGY_EXAMPLES]
    if bad:
        print(f"unknown strategies: {bad}; valid: {list(STRATEGY_EXAMPLES)}", file=sys.stderr)
        sys.exit(2)
    bad = [f for f in framings if f not in FRAMING_HINTS]
    if bad:
        print(f"unknown framings: {bad}; valid: {list(FRAMING_HINTS)}", file=sys.stderr)
        sys.exit(2)

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(2)

    from google import genai
    client = genai.Client(api_key=api_key)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    skeleton_counts = Counter()
    existing = 0
    if args.out.exists():
        with args.out.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                    skeleton_counts[skeleton(row["question_eo"])] += 1
                    existing += 1
                except Exception:
                    continue
        print(f"resume: {existing} already in {args.out}", flush=True)

    stats = Counter()
    accepted = existing
    calls = 0
    t0 = time.time()
    out_f = args.out.open("a")

    while accepted < args.n:
        if args.max_calls and calls >= args.max_calls:
            print(f"hit --max-calls={args.max_calls}; stopping", flush=True)
            break
        strategy = strategies[calls % len(strategies)]
        framing = framings[calls % len(framings)]
        rng = random.Random(calls * 1009 + accepted)  # per-call deterministic
        calls += 1
        prompt = build_prompt(args.batch_size, strategy, framing, rng)
        try:
            resp = client.models.generate_content(model=args.model, contents=prompt)
            text = resp.text or ""
        except Exception as e:
            stats["api-error"] += 1
            print(f"  [call {calls}] API error: {e}", flush=True)
            time.sleep(2)
            continue

        items = parse_response(text)
        if not items:
            stats["parse-fail"] += 1
            print(f"  [call {calls}] parse fail; first 200 chars: {text[:200]!r}",
                  flush=True)
            continue

        for it in items:
            stats["total"] += 1
            q = it.get("question_eo", "").strip()
            chain = it.get("chain_eo", "").strip()
            ans = it.get("answer")
            if not q or not chain or ans is None:
                stats["missing-field"] += 1
                continue
            ok, why = verify_chain(chain, ans)
            if not ok:
                stats[f"verify:{why.split(':')[0]}"] += 1
                continue
            sk = skeleton(q)
            if skeleton_counts[sk] >= args.max_dup:
                stats["dup-skeleton"] += 1
                continue
            skeleton_counts[sk] += 1
            row = {"type": "ratio", "question_eo": q, "chain_eo": chain,
                   "answer": float(ans) if "." in str(ans) else int(ans),
                   "strategy": strategy, "framing": framing}
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            accepted += 1
            stats["accepted"] += 1

        rate = (accepted - existing) / max(1, time.time() - t0) * 60
        print(f"  [call {calls}] accepted={accepted}/{args.n}  "
              f"({rate:.1f}/min)  stats={dict(stats)}",
              flush=True)

    out_f.close()
    print(f"\ndone: {accepted}/{args.n} accepted, {calls} API calls, "
          f"{time.time()-t0:.0f}s wall")
    print(f"  stats: {dict(stats)}")
    print(f"  unique skeletons: {len(skeleton_counts)}")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()
