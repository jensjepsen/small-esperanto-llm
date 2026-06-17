"""Hand-crafted Dolly-style probe in Esperanto.

Novel passages and questions the model has not seen during SFT
training. Covers all Dolly categories so we can measure real
generalization rather than memorization of the training set
(every example in data/sft/sft_dolly.jsonl is trained on, so the
random-sample probe is a memorization test, not a generalization
test).

Usage:
    CKPT=runs/large/checkpoint-44000-sft-v63/checkpoint-9450 \\
        uv run python scripts/dolly_handcraft_probe.py
"""
import os
import re
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from esperanto_lm.data import _morpheme_preprocess

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
SPECIAL = (USER, ASST, END)


def preprocess_chat(text):
    pat = "(" + "|".join(re.escape(t) for t in SPECIAL) + ")"
    return " ".join(
        p if p in SPECIAL else _morpheme_preprocess(p)
        for p in re.split(pat, text))


ckpt = os.environ["CKPT"]
print(f"Using {ckpt}\n")
tok = PreTrainedTokenizerFast.from_pretrained("tokenizer_morpheme")
tok.add_special_tokens({"additional_special_tokens": list(SPECIAL)})
model = AutoModelForCausalLM.from_pretrained(
    ckpt, torch_dtype=torch.float16).cuda().eval()
model.resize_token_embeddings(len(tok))
end_id = tok.convert_tokens_to_ids(END)


def ask(prompt, max_new_tokens=100):
    chat = preprocess_chat(f"{USER} {prompt} {ASST}")
    inputs = tok(chat, return_tensors="pt",
                 return_token_type_ids=False).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=end_id,
            pad_token_id=tok.pad_token_id or end_id)
    gen = out[0][inputs["input_ids"].shape[-1]:].tolist()
    toks = tok.convert_ids_to_tokens(gen)
    cleaned = []
    for t in toks:
        if t == END: break
        if t in ("<s>", "</s>", "<pad>", "<unk>", USER, ASST): continue
        cleaned.append(t)
    return "".join(" " if t == "<w>" else t for t in cleaned).strip()


# Format: (category, prompt, accepted_keywords)
# All hand-written: fictional or specific real facts NOT verbatim
# from sft_dolly.jsonl.
ITEMS = [
    # ---- closed_qa: fact extraction from short passage ----
    ("closed_qa",
     "Anjelika Kovaĉ naskiĝis en Slovakio en 1962 kaj fariĝis fama violonistino.\n"
     "Kio estas la profesio de Anjelika?",
     ["violonist", "violono", "muzikist"]),

    ("closed_qa",
     "La Bluaj Montoj situas en orienta Aŭstralio kaj kovras pli ol 11 000 km². "
     "Ili estas konataj pro siaj profundaj kanjonoj kaj antikvaj eŭkaliptaj arbaroj.\n"
     "En kiu lando situas la Bluaj Montoj?",
     ["aŭstralio", "aŭstrali"]),

    ("closed_qa",
     "La biciklo de Petro havas tri sonorilojn, du retrovidilojn, kaj brilan ruĝan kadron. "
     "Lia frato havas similan biciklon sed kun nigra kadro.\n"
     "Kian koloron havas la biciklo de Petro?",
     ["ruĝ"]),

    ("closed_qa",
     "La aviadilo de la firmao Aerolinjoj de la Nordo flugas trifoje semajne inter "
     "Rejkjaviko kaj Helsinko. La flugo daŭras kvar horojn.\n"
     "Kiom da horoj daŭras la flugo?",
     ["kvar", "4"]),

    ("closed_qa",
     "Maria havas du katojn, Tigro kaj Neĝo. Tigro estas ruĝbruna kaj havas longan voston. "
     "Neĝo estas tute blanka kaj timema.\n"
     "Kiu el la katoj de Maria estas blanka?",
     ["neĝo"]),

    ("closed_qa",
     "La libro 'Sub la Luno' de Olga Velo aperis en 2018 kaj ricevis la premion 'Verda Stelo' "
     "en 2019. La intrigo okazas en fora futuro sur la planedo Marso.\n"
     "Sur kiu planedo okazas la intrigo de la libro?",
     ["mars"]),

    # ---- classification: pick subset / assign label ----
    ("classification",
     "Klasifiku ĉiun el la sekvaj kiel fruktoj aŭ legomoj: pomo, karoto, banano, "
     "tomato, kukumo, ĉerizo, brokolo, persiko.",
     ["pomo", "banano", "ĉerizo", "persiko"]),

    ("classification",
     "Diru al mi, kiuj el la sekvaj estas muzikinstrumentoj: piano, hundo, fluto, "
     "biciklo, gitaro, libro, violono.",
     ["piano", "fluto", "gitaro", "violono"]),

    ("classification",
     "Klasifiku la sekvajn vortojn kiel verbojn aŭ substantivojn: kuri, tablo, "
     "kanti, ĝardeno, dormi, libro, lerni.",
     ["kuri", "kanti", "dormi", "lerni"]),

    ("classification",
     "Ĉu la sekva frazo esprimas pozitivan aŭ negativan opinion?\n"
     "'La filmo estis terure enuiga kaj mi bedaŭras la perditan tempon.'",
     ["negativ"]),

    ("classification",
     "Klasifiku ĉiun el la sekvaj bestoj kiel mamulojn aŭ birdojn: hundo, agleo, "
     "kato, paserto, hipopotamo, najtingalo.",
     ["hundo", "kato", "hipopotamo"]),

    # ---- information_extraction: pull specific fact(s) ----
    ("information_extraction",
     "El la suba teksto, eltiru la jaron, kiam la firmao estis fondita.\n\n"
     "Kvantum Komputiloj S.A. estis fondita en 1987 de tri inĝenieroj en Prago. "
     "La firmao kreskis rapide kaj havas hodiaŭ pli ol 500 dungitojn.",
     ["1987"]),

    ("information_extraction",
     "Eltiru la nomojn de la urboj el la teksto.\n\n"
     "Dum mia vojaĝo mi vizitis Berlinon, Prahon, Vienon kaj Budapeŝton antaŭ "
     "reveni hejmen.",
     ["berlin", "prah", "vien", "budapeŝt"]),

    ("information_extraction",
     "El la suba teksto, eltiru la nomon de la ĉefa rolulo.\n\n"
     "En la mateno, Lukaso ellitiĝis kaj iris al la kuirejo. Tie li trovis sian "
     "patrinon, kiu jam preparis matenmanĝon. Poste, li iris al la lernejo.",
     ["lukas"]),

    ("information_extraction",
     "Eltiru la prezojn el la teksto.\n\n"
     "La nova telefono kostas 250 eŭrojn, dum la malnova modelo kostas nur 120 eŭrojn. "
     "La protekta ujo estas vendita por 15 eŭroj.",
     ["250", "120", "15"]),

    ("information_extraction",
     "Eltiru la temperaturojn menciitajn en la teksto.\n\n"
     "Hieraŭ la temperaturo en Madrido atingis 38 gradojn, dum en Oslo ĝi estis nur "
     "12 gradoj. En Reykjaviko la termometro montris 6 gradojn.",
     ["38", "12", "6"]),

    # ---- open_qa: world knowledge ----
    ("open_qa",
     "Kio estas la valuto de Japanio?",
     ["jeno", "japana jeno"]),

    ("open_qa",
     "Kiu pentris la Monan Lizon?",
     ["leonardo", "da vinci", "davinci"]),

    ("open_qa",
     "Kio estas la plej longa rivero en Afriko?",
     ["nilo", "nile"]),

    ("open_qa",
     "Kiu skribis la verkon 'Hamleto'?",
     ["shakespeare", "ŝekspir", "shakspear"]),

    ("open_qa",
     "Kiu malkovris penicilinon?",
     ["fleming", "alexander fleming"]),

    ("open_qa",
     "Kio estas la kemia simbolo de oro?",
     ["au"]),

    # ---- general_qa: explain/advise/describe ----
    ("general_qa",
     "Kio okazas kiam akvo estas hejtita ĝis 100 gradoj Celsius?",
     ["bolas", "boli", "vapor", "vaporiĝ", "gas"]),

    ("general_qa",
     "Kiel mi povas mildigi mian streson?",
     ["spir", "meditad", "ekzerc", "promen", "dorm", "ripoz", "muzik"]),

    ("general_qa",
     "Kial folioj de arboj falas en aŭtuno?",
     ["mall", "lum", "varm", "frost", "sezon", "ŝparas", "akvo", "energi"]),

    ("general_qa",
     "Kio estas la diferenco inter rivero kaj lago?",
     ["fluas", "fluado", "fluo", "movi", "staras", "senmova"]),

    ("general_qa",
     "Kial homoj bezonas dormi?",
     ["ripoz", "energi", "cerb", "san", "memor", "resaniĝ"]),

    # ---- brainstorming: list generation ----
    ("brainstorming",
     "Donu al mi liston de kvin koloroj, kiuj aperas en ĉielarko.",
     ["ruĝa", "oranĝa", "flava", "verda", "blua", "violkolor", "violet", "indig"]),

    ("brainstorming",
     "Nomu kvar muzikinstrumentojn.",
     ["piano", "violono", "gitaro", "fluto", "tamburo", "trumpet", "saksofon", "violonĉel"]),

    ("brainstorming",
     "Donu al mi liston de tri sportoj, kiujn oni povas praktiki en akvo.",
     ["naĝ", "sub-naĝ", "akvo-pilk", "akvo pilk", "remad", "rem", "surfad", "surf", "veler", "vel"]),

    ("brainstorming",
     "Nomu kvin landojn en Eŭropo.",
     ["franci", "german", "hispan", "ital", "pollando", "portugali", "nederland",
      "belgio", "svedio", "norvegio", "finnlando", "danio", "greki", "ĉeĥio",
      "aŭstrio", "rumani", "hungari"]),

    ("brainstorming",
     "Sugestu kvar manĝaĵojn, kiuj estas tipaj por matenmanĝo.",
     ["pano", "ovo", "lakto", "kafo", "teo", "fromaĝ", "marmelad", "buter", "fruktaĵ",
      "fruktoj", "fruktokuk", "kuko", "jogurto", "cerealo", "muesli", "krespoj", "ŝinko"]),

    # ---- summarization: compress passage ----
    ("summarization",
     "Resumu en unu frazo:\n\n"
     "La Internacia Spaca Stacio (ISS) estas modula spacstacio en malalta tera "
     "orbito. Ĝi estas multnacia kunlabora projekto kun kvin partoprenantaj "
     "spacagentejoj: NASA (Usono), Roskosmos (Rusio), JAXA (Japanio), ESA "
     "(Eŭropo) kaj CSA (Kanado). La proprieto kaj uzo de la stacio estas "
     "establitaj per interregistraj traktatoj.",
     ["spacstacio", "stacio", "iss", "spaca"]),

    ("summarization",
     "Resumu mallonge:\n\n"
     "Olive Schreiner (24-a de marto 1855 – 11-a de decembro 1920) estis "
     "sudafrika verkistino, kontraŭmilita aktivulino kaj politikistino. Ŝi "
     "estas plej konata pro sia romano The Story of an African Farm (1883), "
     "kiu estis tre influa kaj iĝis klasikaĵo de la viktoria literaturo.",
     ["sudafrika", "verkist", "schreiner", "african farm", "1855"]),

    ("summarization",
     "Resumu en unu frazo:\n\n"
     "Tomas Vilkas naskiĝis en Vilno en 1934 kaj fariĝis konata vitra "
     "skulptisto. Liaj grandformataj instalaĵoj ornamas pli ol kvardek "
     "publikajn placojn en orienta Eŭropo. Li mortis en 2008, sed lia "
     "studio ankoraŭ funkcias hodiaŭ sub la gvido de lia plej aĝa filino.",
     ["vilkas", "vitra", "skulpt", "vilno"]),

    ("summarization",
     "Resumu mallonge:\n\n"
     "La giganta panda estas urso indiĝena al centra Ĉinio. Ĝia dieto konsistas "
     "preskaŭ tute el bambuo, kvankam ĝi estas teknike karnovorulo. Pro perdo "
     "de vivejo kaj malalta naskorato, la specio estis longe konsiderata "
     "endanĝerigita, sed ĝia statuso pliboniĝis lastatempe.",
     ["pand", "urs", "bambuo", "ĉini"]),

    ("summarization",
     "Resumu en unu frazo:\n\n"
     "Heinrich Schliemann (1822–1890) estis germana komercisto kaj amatora "
     "arkeologo, kiu fariĝis riĉa per komerco antaŭ dediĉi sian vivon al la "
     "serĉo de antikvaj urboj menciitaj en homera poezio. En 1871 li komencis "
     "fosadojn ĉe la monteto Hisarlik en nuna Turkio, kie li kredis trovi la "
     "legendan Trojon.",
     ["schliemann", "arkeolog", "troj", "hisarlik", "german"]),
]


total = 0
correct = 0
results_by_cat = {}
for cat, prompt, keys in ITEMS:
    pred = ask(prompt, max_new_tokens=80)
    ok = any(k.lower() in pred.lower() for k in keys)
    mark = "✓" if ok else "✗"
    print(f"  {mark} [{cat}] expected ∈ {keys[:3]}{'...' if len(keys)>3 else ''}")
    print(f"      Q: {prompt[:120]}{'...' if len(prompt)>120 else ''}")
    print(f"      pred: {pred[:160]}")
    correct += ok
    total += 1
    results_by_cat.setdefault(cat, [0, 0])
    results_by_cat[cat][0] += 1
    results_by_cat[cat][1] += ok

print(f"\n=== Total: {correct}/{total} ===")
for cat, (t, c) in results_by_cat.items():
    print(f"  {cat:25s}: {c}/{t}")
