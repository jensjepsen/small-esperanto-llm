"""Probe the task-expansion subtypes on an SFT ckpt.

Loads the ckpt, feeds a small DA-wiki passage through each task-type prompt
variant (rc, reason, textman), streams model output for eyeball QA.
"""
from __future__ import annotations
import argparse, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"

PASSAGES = {
    "isbjørn": """Isbjørnen (Ursus maritimus) er et stort rovdyr i bjørnefamilien, som lever i Arktis. Den er
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
mellem de fem stater med isbjørnbestande: Canada, USA, Rusland, Norge og Danmark (Grønland).""",

    "roentgen": """Wilhelm Conrad Röntgen (1845-1923) var en tysk fysiker, der i 1895 opdagede en ny type
elektromagnetisk stråling, som han kaldte X-stråler — i dag også kendt som røntgenstråler.
Opdagelsen skete ved et uheld, mens han eksperimenterede med kathodestrålerør i sit laboratorium
i Würzburg.

Röntgen bemærkede, at en fluorescerende skærm i nærheden af røret begyndte at lyse, selv når røret
var dækket med sort pap. Han fandt hurtigt ud af, at strålingen kunne trænge gennem bløde
materialer som papir og hud, men blev standset af tættere stoffer som knogler og metal. Han tog
det første røntgenbillede nogensinde — af sin kones hånd, hvor hendes vielsesring var tydeligt
synlig.

For denne opdagelse modtog Röntgen den allerførste Nobelpris i fysik i 1901. Han donerede hele
prispengesummen på 50.000 svenske kroner til Würzburgs universitet. Røntgen nægtede at tage patent
på sin opdagelse, fordi han mente, at videnskaben tilhørte menneskeheden. I dag anvendes
røntgenstråler bredt i medicin, industri og videnskab.""",

    "vandkredsløb": """Vandkredsløbet er den kontinuerlige bevægelse af vand mellem jordens overflade, atmosfære og
undergrund. Solen driver kredsløbet: solstrålingen fordamper vand fra oceaner, søer og floder til
vanddamp i atmosfæren. Denne proces kaldes fordampning. Cirka 90 procent af al fordampning sker
fra havoverfladen.

I atmosfæren kondenserer vanddampen til små dråber, som danner skyer. Når dråberne bliver store
nok, falder de tilbage til jorden som nedbør — regn, sne eller hagl. Cirka to tredjedele af den
årlige nedbør på jorden falder på havet, mens den resterende tredjedel falder på land.

Vand, der lander på land, følger flere veje: noget infiltrerer jorden og bliver til grundvand,
noget løber som overfladeafstrømning tilbage til floder og havet, og noget optages af planter,
som senere afgiver det til atmosfæren gennem transpiration. Det samlede vandvolumen på jorden er
konstant — cirka 1,4 milliarder kubikkilometer — og har været det i milliarder af år.""",
}

# Passage-specific probe sets
def probes_for(passage_key):
    passage = PASSAGES[passage_key]
    title = passage_key
    if passage_key == "isbjørn":
        return [
            ("rc-numeric-lookup",   f"Baseret på nedenstående tekst, hvor mange delbestande er den samlede isbjørnbestand fordelt på, og hvad er den samlede bestandsestimat?\n\nTekst:\n{passage}"),
            ("rc-numeric-derive",   f"Baseret på nedenstående tekst, hvor mange kilogram er en gennemsnitlig han-isbjørn tungere end en hun-isbjørn?\n\nTekst:\n{passage}"),
            ("reason-multi_step",   f"ARTIKEL: {passage}\n\nHvis isbjørnen får energi fra sæler og en enkelt sæl dækker to uger, og der er 52 uger i et år, hvor mange sæler skal isbjørnen minimum fange på et år for at overleve? Vis dine trin."),
            ("reason-analogy",      f"Baseret på artiklen om {title}:\n{passage}\n\nLav en analogi der forklarer isbjørnens afhængighed af havisen ved at sammenligne med noget mere velkendt. Forklar hvor analogien holder og hvor den bryder sammen."),
            ("textman-extraction",  f"Udtræk personer, steder, datoer og tal som JSON med nøglerne people, places, dates, numbers.\n\n{passage}"),
        ]
    if passage_key == "roentgen":
        return [
            ("rc-attribution",      f"{passage}\n\n---\nHvem opdagede røntgenstråler og i hvilken by fandt opdagelsen sted?"),
            ("rc-numeric-lookup",   f"Læs artiklen og besvar: I hvilket år modtog Röntgen sin Nobelpris, og hvor stort var prisbeløbet?\n\n{passage}"),
            ("rc-causal_inference", f"ARTIKEL: {passage}\n\nHvorfor kunne Röntgen tage et billede af sin kones vielsesring gennem hendes hånd?"),
            ("reason-fact_check",   f"Er følgende påstand SAND eller FALSK ifølge teksten? Begrund.\n\nPåstand: Röntgen tog patent på sin opdagelse for at sikre indtægter til Würzburgs universitet.\n\nTekst:\n{passage}"),
            ("reason-argumentation", f"Baseret på artiklen:\n{passage}\n\nEr det en god idé for videnskabsfolk at afstå fra at tage patent på deres opdagelser, som Röntgen gjorde? Argumenter for og imod."),
            ("textman-summary",     f"Opsummer artiklen i 3 bulletpoints.\n\n{passage}"),
            ("textman-extraction",  f"Udtræk personer, steder, datoer og tal som JSON med nøglerne people, places, dates, numbers.\n\n{passage}"),
            ("textman-tweet",       f"{passage}\n\nOmskriv artiklens essens som en tweet på under 280 tegn."),
        ]
    if passage_key == "vandkredsløb":
        return [
            ("rc-numeric-derive",   f"Læs teksten: hvis to tredjedele af nedbøren falder på havet, hvor stor en andel falder så på land?\n\n{passage}"),
            ("rc-ordering",         f"ARTIKEL: {passage}\n\nSæt følgende fire trin i vandkredsløbet i den rigtige rækkefølge: kondensation, nedbør, fordampning, overfladeafstrømning."),
            ("reason-causal",       f"Baseret på teksten:\n{passage}\n\nHvorfor fordamper der så meget mere vand fra havet end fra land? Forklar mekanismen."),
            ("reason-multi_step",   f"{passage}\n\nHvis solen er drivkraften bag fordampning og fordampning skaber skyer, hvad kan vi så konkludere om, hvad der ville ske med nedbøren hvis solindstrålingen halveredes? Vis dine trin."),
            ("reason-ranking",      f"Baseret på teksten:\n{passage}\n\nRangér de tre veje, som vand på land kan følge (infiltration, overfladeafstrømning, transpiration), efter hvor stor en rolle du tror de spiller for det samlede vandkredsløb. Kort begrundelse for hver."),
            ("textman-style_formal",f"Omskriv artiklens indledning i formel akademisk stil.\n\n{passage}"),
            ("textman-summary",     f"Skriv 3 bulletpoints der opsummerer teksten.\n\n{passage}"),
        ]
    return []


GENERIC_PROBES = [
    ("rc-multi_fact",       "Læs følgende artikel og besvar spørgsmålet.\n\nARTIKEL ({title}):\n{text}\n\nSpørgsmål: Sammenfat de tre vigtigste pointer artiklen gør."),
    ("rc-numeric-lookup",   "Baseret på nedenstående tekst, hvad er det største tal eller den seneste dato der nævnes, og hvad refererer det til?\n\nTekst:\n{text}"),
    ("rc-attribution",      "{text}\n\n---\nHvem eller hvad er hovedpersonen/emnet i artiklen, og hvilke andre navngivne personer, steder eller institutioner er nævnt?"),
    ("reason-causal",       "Læs følgende og forklar hvorfor.\n\n{text}\n\nHvad er den vigtigste årsag-virkning-sammenhæng artiklen beskriver? Forklar mekanismen."),
    ("reason-fact_check",   "Er følgende påstand SAND eller FALSK ifølge teksten? Begrund.\nPåstand: Artiklens hovedperson/emne blev grundlagt/født i det 20. århundrede.\n\nTekst:\n{text}"),
    ("reason-analogy",      "Baseret på artiklen om {title}:\n{text}\n\nLav en analogi der forklarer artiklens hovedemne ved at sammenligne med noget mere velkendt. Forklar hvor analogien holder og hvor den bryder sammen."),
    ("textman-summary",     "Opsummer artiklen i 3 bulletpoints.\n\nARTIKEL ({title}):\n{text}"),
    ("textman-extraction",  "Udtræk personer, steder, datoer og tal som JSON med nøglerne people, places, dates, numbers.\n\n{text}"),
    ("textman-style_casual",f"Omskriv artiklens indledning i afslappet talesprog.\n\n{{text}}"),
    ("textman-tweet",       "{text}\n\nOmskriv artiklens essens som en tweet på under 280 tegn."),
]


def wiki_article(idx, min_chars=2000, max_chars=4500):
    """Pull a real DA wiki article by index (skips ones outside length window)."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.da", split="train",
                      streaming=False)
    import random
    rng = random.Random(idx)
    idxs = list(range(len(ds))); rng.shuffle(idxs)
    for i in idxs:
        r = ds[i]
        if min_chars <= len(r["text"]) <= max_chars:
            return r["title"], r["text"]
    raise SystemExit(f"no wiki article found in [{min_chars},{max_chars}] chars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--max-new", type=int, default=350)
    ap.add_argument("--passages", default=None,
                    help="Comma-separated passage keys (default: all built-in)")
    ap.add_argument("--wiki", default=None,
                    help="Comma-separated wiki seeds, each fetches a real DA "
                         "wiki article via that RNG seed (skips built-in probes).")
    ap.add_argument("--only", default=None, help="Comma-separated probe names")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.float16).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id] + ([end_id] if end_id != tok.unk_token_id else [])

    want = set(args.only.split(",")) if args.only else None

    if args.wiki:
        # Real wiki article mode — use GENERIC_PROBES with the fetched article.
        for seed in args.wiki.split(","):
            title, text = wiki_article(int(seed))
            print(f"\n\n{'#'*80}\n#  WIKI (seed={seed}): {title}  ({len(text)} chars)\n{'#'*80}",
                  flush=True)
            print(f"[TEXT PREVIEW]\n{text[:400]}...\n", flush=True)
            for name, tpl in GENERIC_PROBES:
                if want and name not in want: continue
                prompt = tpl.format(title=title, text=text)
                print(f"\n{'='*80}\n▶ [{title}] {name}\n{'='*80}", flush=True)
                wrapped = f"{USER}{prompt}{END}{ASST}"
                enc = tok(wrapped, return_tensors="pt", add_special_tokens=False,
                          return_token_type_ids=False).to("cuda")
                streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
                with torch.no_grad():
                    model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                   pad_token_id=tok.pad_token_id, eos_token_id=eos_ids,
                                   streamer=streamer)
                sys.stdout.flush()
        return

    passages = args.passages.split(",") if args.passages else list(PASSAGES.keys())
    for pkey in passages:
        print(f"\n\n{'#'*80}\n#  PASSAGE: {pkey}\n{'#'*80}", flush=True)
        for name, prompt in probes_for(pkey):
            if want and name not in want: continue
            print(f"\n{'='*80}\n▶ [{pkey}] {name}\n{'='*80}", flush=True)
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
