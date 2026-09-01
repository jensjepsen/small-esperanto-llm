"""Hand-written Danish instruction shapes, to complement the generated bank.

The generated set (203 candidates) was lexically varied but structurally
uniform: 202 of 203 were `[verb] ordret [felterne]. [mangel-clause] {null}.`
Asking one prompt for "terse / neutral / polite / technical" produced synonym
substitution inside a single template, not different shapes. A model trained on
that still learns to expect instructions of that form.

These are grouped by SHAPE rather than wording, covering forms the generator
never produced: questions, telegraphic fragments, multi-sentence explanations,
numbered rules, second-person framing, and instructions where the null rule
comes first rather than last.

{null} is a placeholder filled at render time.
"""

SHAPES = {
    # ── telegraphic: barely a sentence ────────────────────────────────────
    "telegraphic": [
        "Felter. Ordret. Mangler: {null}.",
        "Ordret udtræk. Ikke nævnt = {null}.",
        "Kopier ordret. Intet fundet: {null}.",
        "Ordrette værdier. Fravær: {null}.",
        "Udtræk. Ordret. {null} ved mangel.",
        "Én linje per felt, ordret, {null} hvis intet.",
        "Ordret afskrift per felt. Ellers {null}.",
    ],
    # ── question forms ────────────────────────────────────────────────────
    "question": [
        "Hvad står der i teksten om hvert felt? Svar med ordret afskrift, "
        "og {null} hvis teksten ikke nævner feltet.",
        "Kan du finde hvert felts værdi i teksten? Gengiv den ordret. Er der "
        "intet at finde, så skriv {null}.",
        "Hvilken tekst hører til hvert felt? Kopier den præcis som den står, "
        "eller skriv {null} hvis den ikke er der.",
        "Hvor i teksten står oplysningen til hvert felt? Skriv den af ordret; "
        "står den ikke der, skriv {null}.",
    ],
    # ── polite / softened ─────────────────────────────────────────────────
    "polite": [
        "Vær venlig at udfylde felterne med ordret tekst fra passagen. Hvis "
        "et felt ikke omtales, så skriv {null}.",
        "Du bedes gengive hvert felts værdi ordret fra teksten. Skulle et "
        "felt mangle, angives {null}.",
        "Vil du udtrække felterne nedenfor? Værdierne skal være ordrette, og "
        "manglende felter markeres {null}.",
        "Hjælp med at finde felternes værdier i teksten. Kopier ordret, og "
        "brug {null} hvor teksten intet siger.",
    ],
    # ── null rule stated FIRST, not last ──────────────────────────────────
    "null_first": [
        "Brug {null} for ethvert felt teksten ikke nævner. Alle øvrige felter "
        "udfyldes med ordret tekst fra passagen.",
        "Står oplysningen ikke i teksten, er svaret {null}. Ellers gengives "
        "den ordret, uden omskrivning.",
        "{null} betyder 'ikke nævnt i teksten'. Brug det hvor det passer, og "
        "kopier ellers ordret.",
        "Manglende felter får værdien {null}. Resten udfyldes med nøjagtig "
        "afskrift fra teksten.",
    ],
    # ── numbered rules ────────────────────────────────────────────────────
    "numbered": [
        "1) Læs teksten. 2) Find hvert felts værdi. 3) Skriv den ordret af. "
        "4) Mangler feltet, skriv {null}.",
        "Regler: (a) værdier kopieres ordret, (b) én linje per felt, "
        "(c) felter uden dækning i teksten får {null}.",
        "Tre krav: ordret afskrift, én linje per felt, og {null} for felter "
        "teksten ikke omtaler.",
    ],
    # ── explanatory, multi-sentence ───────────────────────────────────────
    "explanatory": [
        "Nedenfor står en tekst og nogle feltnavne. Din opgave er at finde "
        "hvad teksten siger om hvert felt. Værdien skal være en ordret "
        "tekststump, ikke en omskrivning. Nævner teksten ikke feltet, "
        "skriver du {null}.",
        "Felterne beskriver oplysninger, der kan stå i teksten. Find dem, og "
        "gengiv dem præcis som de er formuleret. Det er vigtigt ikke at "
        "omformulere. Er en oplysning ikke til stede, angives {null}.",
        "Opgaven er udtræk, ikke opsummering. Hvert felt besvares med tekst "
        "kopieret ordret fra passagen. Hvis passagen ikke dækker feltet, "
        "bruges {null} i stedet for et gæt.",
    ],
    # ── negative framing: what NOT to do ──────────────────────────────────
    "negative": [
        "Omskriv ikke, forkort ikke, og gæt ikke. Kopier ordret fra teksten, "
        "og brug {null} når feltet ikke er dækket.",
        "Undgå at formulere dig selv: værdierne skal stå ordret i teksten. "
        "Findes en værdi ikke, er svaret {null}, ikke et skøn.",
        "Der må ikke opfindes værdier. Kun ordret tekst fra passagen, og "
        "{null} hvor passagen tier.",
    ],
    # ── role / framing ────────────────────────────────────────────────────
    "role": [
        "Du er registrator. Før felterne til protokols med ordret tekst fra "
        "kilden. Ikke-dækkede felter noteres {null}.",
        "Som korrekturlæser skal du hente felternes ordlyd uændret fra "
        "teksten. Mangler et felt, sættes {null}.",
        "Arbejd som arkivar: hvert felt får den ordrette tekst fra kilden, "
        "og {null} hvor kilden intet oplyser.",
    ],
    # ── conditional / if-then ─────────────────────────────────────────────
    "conditional": [
        "Hvis feltet står i teksten, gengives det ordret. Hvis ikke, skrives "
        "{null}.",
        "Findes oplysningen? Så skriv den af ordret. Findes den ikke? Så "
        "{null}.",
        "Når et felt kan belægges med tekst, kopieres teksten ordret; ellers "
        "anføres {null}.",
    ],
}

ALL = [t for v in SHAPES.values() for t in v]
