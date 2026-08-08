"""Probe the task-expansion subtypes on an SFT ckpt.

Loads the ckpt, feeds a small DA-wiki passage through each task-type prompt
variant (rc, reason, textman), streams model output for eyeball QA.
"""
from __future__ import annotations
import argparse, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"

PASSAGE_TITLE = "Isbjørn"
PASSAGE_TEXT = """Isbjørnen (Ursus maritimus) er et stort rovdyr i bjørnefamilien, som lever i Arktis. Den er
verdens største landrovdyr — hanner vejer typisk 350-700 kg og bliver 2,4-3 meter lange, mens
hunner er cirka halvt så store. Isbjørnen har hvide til lysegule hår og en tyk lag spæk, som
holder den varm i temperaturer helt ned til -50 grader Celsius.

Isbjørnen lever hovedsageligt af sæler, som den fanger ved at vente ved åndehuller i isen. En
enkelt sæl kan give en isbjørn energi nok til to uger. Den er en fremragende svømmer og kan
tilbagelægge over 100 kilometer i åbent vand. Klimaændringerne truer arten alvorligt, fordi den
er afhængig af havisen for at kunne jage — når isen smelter tidligere om foråret, får isbjørnene
sværere ved at bygge de nødvendige fedtreserver op inden sommerens fasteperiode.

Den samlede bestand estimeres til 22.000-31.000 individer fordelt på 19 delbestande. IUCN har
klassificeret isbjørnen som "sårbar". Den er beskyttet af den internationale aftale fra 1973
mellem de fem stater med isbjørnbestande: Canada, USA, Rusland, Norge og Danmark (Grønland)."""


PROBES = [
    ("rc-multi_fact",       f"Læs artiklen og besvar spørgsmålet.\n\nARTIKEL:\n{PASSAGE_TEXT}\n\nSpørgsmål: Hvordan hænger klimaændringer sammen med isbjørnens fasteperiode?"),
    ("rc-numeric-lookup",   f"Baseret på nedenstående tekst, hvor mange delbestande er den samlede isbjørnbestand fordelt på, og hvad er den samlede bestandsestimat?\n\nTekst:\n{PASSAGE_TEXT}"),
    ("rc-numeric-derive",   f"Baseret på nedenstående tekst, hvor mange kilogram er en gennemsnitlig han-isbjørn tungere end en hun-isbjørn?\n\nTekst:\n{PASSAGE_TEXT}"),
    ("rc-attribution",      f"{PASSAGE_TEXT}\n\n---\nHvem har klassificeret isbjørnen som 'sårbar', og hvornår blev den internationale beskyttelsesaftale indgået?"),
    ("reason-causal",       f"Læs følgende og forklar hvorfor.\n\n{PASSAGE_TEXT}\n\nHvorfor er en enkelt sæl så vigtig for isbjørnens energibalance?"),
    ("reason-multi_step",   f"ARTIKEL: {PASSAGE_TEXT}\n\nHvis isbjørnen får energi fra sæler og en enkelt sæl dækker to uger, og der er 52 uger i et år, hvor mange sæler skal isbjørnen minimum fange på et år for at overleve? Vis dine trin."),
    ("reason-fact_check",   f"Nedenstående er en artikel om {PASSAGE_TITLE}:\n{PASSAGE_TEXT}\n\nEr følgende påstand SAND eller FALSK ifølge teksten? Begrund.\nPåstand: Isbjørnen er verdens største rovdyr til lands."),
    ("reason-analogy",      f"Baseret på artiklen om {PASSAGE_TITLE}:\n{PASSAGE_TEXT}\n\nLav en analogi der forklarer isbjørnens afhængighed af havisen ved at sammenligne med noget mere velkendt. Forklar hvor analogien holder og hvor den bryder sammen."),
    ("textman-summary",     f"Opsummer artiklen i 3 bulletpoints.\n\nARTIKEL:\n{PASSAGE_TEXT}"),
    ("textman-extraction",  f"Udtræk personer, steder, datoer og tal som JSON med nøglerne people, places, dates, numbers.\n\n{PASSAGE_TEXT}"),
    ("textman-style_casual",f"Omskriv artiklens indledning i afslappet talesprog.\n\n{PASSAGE_TEXT}"),
    ("textman-tweet",       f"{PASSAGE_TEXT}\n\nOmskriv artiklens essens som en tweet på under 280 tegn."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--max-new", type=int, default=350)
    ap.add_argument("--only", default=None, help="Comma-separated probe names")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.float16).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id] + ([end_id] if end_id != tok.unk_token_id else [])

    want = set(args.only.split(",")) if args.only else None

    for name, prompt in PROBES:
        if want and name not in want: continue
        print(f"\n{'='*80}\n▶ {name}\n{'='*80}", flush=True)
        wrapped = f"{USER}{prompt}{END}{ASST}"
        enc = tok(wrapped, return_tensors="pt", add_special_tokens=False,
                  return_token_type_ids=False).to("cuda")
        streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=eos_ids,
                           streamer=streamer)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
