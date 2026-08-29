"""Generate Esperanto narrative wrappers for word-problem composition via
Gemini Flash Lite. Each wrapper has {MATH_STATEMENT} + {QUESTION} placeholders;
filling them later (in word_problems_diverse.py) produces the final question.

Output JSON schema:
  [
    {"id": "school-bake-1", "tone": "school-narrative",
     "template": "Antaŭ la lerneja ... {MATH_STATEMENT}. ... {QUESTION}"},
    ...
  ]

Usage:
  GOOGLE_API_KEY=$(cat ~/gem) uv run --extra gemini python scripts/gen_wp_wrappers.py \\
      --n 200 --out data/wp_wrappers.json
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROMPT = """Generu {n} esperantajn rakontajn ŝablonojn por matematikaj
problemoj. Ĉiu ŝablono enhavas KUNTEKSTON kaj du anstataŭigojn:
  {{MATH_STATEMENT}} = matematika deklaro (la nombroj kaj nomoj venos poste)
  {{QUESTION}} = KOMPLETA demanda klaŭzo, ekz. "kiom da pomoj ricevas Maria"
                 (NE estas nominala frazo — ĝi enhavas verbon)

KRITIKAJ STRUKTURAJ REGULOJ:
1. {{QUESTION}} estas KLAŬZO, ne nomo de aĵo. ⛔ NE skribu "kalkuli la
   rezulton de {{QUESTION}}" — tio postulus nomon, ne klaŭzon.
2. La ŝablono mem NE havu sian propran demandon antaŭ {{QUESTION}}.
   ⛔ Malpermesite: "Kiom da pomoj? {{QUESTION}}", "Ĉu vere? {{QUESTION}}"
   ✓ Permesite: ŝablonoj kun NUR komenco/setupo, kaj {{QUESTION}} kiel
                la sola demanda klaŭzo.
3. {{QUESTION}} devas esti la lasta semantika elemento aŭ tre proksima al la
   fino. La fina punkto-marko (? . !) aldoniĝos poste — ne aldonu ĝin
   post {{QUESTION}}.
4. Komenco-frazo regule kapitaligita.

DEVIGE varias laŭ:
- TONO: rakonta (pasinta) / scenara (estanta) / hipoteza (se...us) /
  ĵurnalista / poezia / komika / formala / klariga
- KUNTEKSTO: lernejo, vendejo, festo, sporto, familio, biblioteko, ĝardeno,
  vojaĝo, kuirejo, klubo, urbo, kamparo, oficejo, restoracio, laboratorio,
  muzeo, koncerto, kongreso, esplormisio
- LONGECO: 1 ĝis 3 frazoj
- STRUKTURO de la fluo: setup → math → konekt-vorto + {{QUESTION}}

NE inkluzivu nombrojn, nomojn de personoj, aŭ specifajn aĵojn (tiuj venos
kun {{MATH_STATEMENT}}).

EKZEMPLOJ DE BONAJ ŜABLONOJ:
  ✓ "Antaŭ la lerneja bazaro, du gelernantoj kunlaboris kun la kuirklubo.
     {{MATH_STATEMENT}}. Post la vendado, {{QUESTION}}"
  ✓ "Estis nuba mateno. {{MATH_STATEMENT}}. Tial, {{QUESTION}}"
  ✓ "Laŭ la novaĵoj: {{MATH_STATEMENT}}. La demando estas: {{QUESTION}}"
  ✓ "{{MATH_STATEMENT}}. Surbaze de tio, {{QUESTION}}"

MALBONA EKZEMPLO (NE generu tiajn):
  ⛔ "{{MATH_STATEMENT}}. Kiom da mono restis? {{QUESTION}}"
     — ĉar la ŝablono jam havas demandon antaŭ {{QUESTION}}.
  ⛔ "Mi devas kalkuli {{QUESTION}}"
     — ĉar {{QUESTION}} estas klaŭzo, ne objekto de "kalkuli".

Respondu JSON-listo de {n} objektoj kun ŝlosiloj "id", "tone", "template".
NE uzu ```markdown, NE aldonu klarigan tekston.
"""


def parse_response(text: str):
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
        i, j = text.find("["), text.rfind("]")
        if i >= 0 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                pass
    return []


def validate_wrapper(w: dict) -> tuple[bool, str]:
    if not isinstance(w, dict):
        return False, "not-dict"
    tmpl = w.get("template", "")
    if not isinstance(tmpl, str) or not tmpl.strip():
        return False, "no-template"
    n_math = tmpl.count("{MATH_STATEMENT}")
    n_q = tmpl.count("{QUESTION}")
    if n_math != 1:
        return False, f"math-count={n_math}"
    if n_q != 1:
        return False, f"q-count={n_q}"
    # no rogue placeholders
    other_placeholders = re.findall(r"\{[^}]+\}", tmpl)
    if any(p not in ("{MATH_STATEMENT}", "{QUESTION}") for p in other_placeholders):
        return False, f"rogue-placeholders={other_placeholders}"
    # must contain at least a few words of context
    if len(tmpl.split()) < 4:
        return False, "too-short"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="target accepted wrappers")
    ap.add_argument("--batch-size", type=int, default=20, help="per Gemini call")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    ap.add_argument("--max-calls", type=int, default=50)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY", file=sys.stderr); sys.exit(2)
    from google import genai
    client = genai.Client(api_key=api_key)

    accepted = []
    rejects = 0
    seen_templates = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for call in range(1, args.max_calls + 1):
        if len(accepted) >= args.n:
            break
        prompt = PROMPT.format(n=args.batch_size)
        try:
            resp = client.models.generate_content(model=args.model, contents=prompt)
            text = resp.text or ""
        except Exception as e:
            print(f"  [call {call}] API error: {e}", flush=True)
            time.sleep(2); continue

        items = parse_response(text)
        if not items:
            print(f"  [call {call}] parse fail; head: {text[:200]!r}", flush=True)
            continue

        new_this_call = 0
        for w in items:
            ok, reason = validate_wrapper(w)
            if not ok:
                rejects += 1
                continue
            tmpl = w["template"].strip()
            if tmpl in seen_templates:
                rejects += 1
                continue
            seen_templates.add(tmpl)
            # ensure ID exists
            if not w.get("id"):
                w["id"] = f"wrapper-{len(accepted)+1}"
            accepted.append({"id": w.get("id"),
                              "tone": w.get("tone", "unknown"),
                              "template": tmpl})
            new_this_call += 1
        rate = len(accepted) / max(0.1, time.time() - t0) * 60
        print(f"  [call {call}] accepted={len(accepted)}/{args.n}  "
              f"(+{new_this_call} this call, {rejects} total rejects, "
              f"{rate:.0f}/min)", flush=True)

    # truncate to exactly N
    accepted = accepted[: args.n]
    args.out.write_text(json.dumps(accepted, ensure_ascii=False, indent=2))
    print(f"\ndone: {len(accepted)} wrappers → {args.out}")
    print(f"  rejects: {rejects} ({100*rejects/max(1,rejects+len(accepted)):.0f}%)")
    print(f"  wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
