"""Can the model answer from PROSE tool results, not just field payloads?

probe_field_selection.py covers structured results -- pick the right key out of
several. This covers the retrieval shape: the tool returns passages, the answer
lives inside one of them, and the others are plausible neighbours.

Different failure modes are possible here and each case isolates one:

  pick        the fact is in passage 2 of 3, the others are near-misses
  conflict    two passages disagree; a good answer picks one, not an average
  absent      the answer is NOT in any passage -- abstention is the correct move
  combine     the answer needs a fact from two passages
  long        one long passage, the fact buried mid-way
  number      the passages carry several numbers, one is asked for

Scored per case against `want` (must appear) and `avoid` (must not). The
`absent` case inverts it: citing anything specific is the failure, and saying
so is the pass.
"""
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CALL_RE = re.compile(r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)", re.S)

SEARCH = {
    "name": "search_documents",
    "description": "Søg i dokumentsamlingen og få de mest relevante uddrag",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Søgeordene"}},
        "required": ["query"]},
    "returns": {"type": "object", "properties": {
        "passages": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"description": "Dokumentets titel"},
            "text": {"description": "Uddrag fra dokumentet"}}}}}},
}

CASES = [
    ("pick", "Hvor mange ansatte har Nordvind A/S?",
     {"passages": [
         {"title": "Nordvind A/S — historie",
          "text": "Nordvind A/S blev grundlagt i 1987 i Esbjerg og byggede "
                  "sin første vindmøllepark i 1991."},
         {"title": "Nordvind A/S — nøgletal",
          "text": "Selskabet har i dag 412 ansatte fordelt på fire "
                  "afdelinger og omsatte sidste år for 1,8 mia. kr."},
         {"title": "Sydvind ApS — nøgletal",
          "text": "Sydvind ApS har 96 ansatte og er hovedsageligt aktiv i "
                  "Sønderjylland."}]},
     ["412"], ["96", "1987", "1991"]),

    ("conflict", "Hvornår åbnede Storebæltsbroen?",
     {"passages": [
         {"title": "Broer i Danmark",
          "text": "Storebæltsbroen åbnede for biltrafik i 1998."},
         {"title": "Infrastrukturhistorie",
          "text": "Jernbanedelen af Storebæltsforbindelsen blev taget i brug "
                  "i 1997, et år før vejforbindelsen."}]},
     ["1998", "1997"], []),

    ("absent", "Hvad er Nordvind A/S' CVR-nummer?",
     {"passages": [
         {"title": "Nordvind A/S — nøgletal",
          "text": "Selskabet har 412 ansatte og omsatte sidste år for "
                  "1,8 mia. kr."},
         {"title": "Nordvind A/S — bestyrelse",
          "text": "Bestyrelsen består af fem medlemmer og ledes af "
                  "Kirsten Dahl."}]},
     [], ["412", "1,8", "fem", "5"]),

    ("combine", "Hvem leder bestyrelsen i det selskab der har 412 ansatte?",
     {"passages": [
         {"title": "Nordvind A/S — nøgletal",
          "text": "Nordvind A/S har 412 ansatte og fire afdelinger."},
         {"title": "Nordvind A/S — bestyrelse",
          "text": "Bestyrelsen ledes af Kirsten Dahl, der har siddet i "
                  "posten siden 2019."},
         {"title": "Sydvind ApS — bestyrelse",
          "text": "Bestyrelsen ledes af Mogens Riis."}]},
     ["Kirsten"], ["Mogens"]),

    ("long", "Hvilket år flyttede biblioteket til Rådhuspladsen?",
     {"passages": [
         {"title": "Byens bibliotek gennem tiden",
          "text": "Biblioteket blev oprettet i 1904 i en lejet lejlighed på "
                  "Vestergade. Samlingen voksede hurtigt, og i 1912 fik man "
                  "en bevilling til reoler og et læsevaerelse. Under krigen "
                  "blev dele af samlingen opmagasineret. I 1961 flyttede "
                  "biblioteket til en nyopført bygning på Rådhuspladsen, "
                  "hvor det stadig ligger. En renovering fandt sted i 1998, "
                  "og i 2014 åbnede en ny børneafdeling."}]},
     ["1961"], ["1904", "1912", "1998", "2014"]),

    ("number", "Hvad kostede renoveringen af broen?",
     {"passages": [
         {"title": "Broprojektet — økonomi",
          "text": "Renoveringen kostede 340 mio. kr. og blev finansieret "
                  "over ti år."},
         {"title": "Broprojektet — trafik",
          "text": "Broen benyttes dagligt af 27.000 køretøjer, og "
                  "hastighedsgrænsen er 80 km/t."}]},
     ["340"], ["27.000", "27000", "80"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.float16).cuda().eval()
    eos = [i for i in (tok.eos_token_id,
                       tok.convert_tokens_to_ids("<|end|>")) if i is not None]

    def gen(p, n=220):
        e = tok(p, return_tensors="pt", add_special_tokens=False,
                return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=n, do_sample=False,
                               eos_token_id=eos,
                               pad_token_id=tok.pad_token_id or 0,
                               repetition_penalty=1.1)
        return tok.decode(o[0][e["input_ids"].shape[1]:],
                          skip_special_tokens=False).strip()

    cat = json.dumps([SEARCH], ensure_ascii=False)
    ok = 0
    for kind, q, payload, want, avoid in CASES:
        p = f"<|user|>Værktøjer:\n{cat}\n\n{q}<|end|><|assistant|>"
        out = gen(p)
        m = CALL_RE.search(out)
        called = "(no call)"
        if m:
            try:
                called = json.dumps(json.loads(m.group(1).strip()),
                                    ensure_ascii=False)
            except Exception:
                called = m.group(1).strip()[:80]
        res = json.dumps(payload, ensure_ascii=False)
        p2 = (p + out.split("<|/tool_call|>")[0] + "<|/tool_call|><|end|>"
              + f"<|tool_result|>{res}<|/tool_result|><|assistant|>")
        ans = gen(p2, 200).replace("<|end|>", "").strip()
        low = ans.lower()
        hit = all(w.lower() in low for w in want) if want else True
        bad = any(a.lower() in low for a in avoid)
        if kind == "absent":
            # correct behaviour is to say it is not in the documents
            says_no = any(s in low for s in
                          ("ikke", "fremgår ikke", "kan ikke", "ingen"))
            good = says_no and not bad
        elif kind == "conflict":
            good = any(w.lower() in low for w in want)   # either, not both wrong
        else:
            good = hit and not bad
        ok += good
        print("=" * 78)
        print(f"[{kind}] {q}")
        print(f"  call   : {called}")
        print(f"  answer : {ans[:300]}")
        print(f"  verdict: {'PASS' if good else 'FAIL'}"
              + (f"   (wanted {want}" if want else "   (wanted abstention")
              + (f", avoid {avoid})" if avoid else ")"))
    print("=" * 78)
    print(f"passed {ok}/{len(CASES)}")


if __name__ == "__main__":
    main()
