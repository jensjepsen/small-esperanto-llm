"""Generate ICL Q/A pairs from rich regression traces.

Input: a JSONL produced by `run_regression_parallel.py` with the
extended schema (events + entities + setup_relations alongside
prose).

Output: an SFT JSONL where each line is

    {"messages":
        [{"role": "user",    "content": "<prose>\\n\\nDemando: <Q>"},
         {"role": "assistant", "content": "<A>"}]}

Each trace yields multiple Q/A pairs covering several question
templates. Questions are grounded in the trace's actual events
and entity properties, so answers are always factually correct
relative to the rendered prose.

Templates:

  - intrinsic property      "Kia estis la koloro de la X?"
                            answered from entity.properties.

  - first/last action       "Kio okazis unue/laste?"
                            answered from events[0/-1].

  - action attribution      "Kiu prenis la Y?" / "Kion la X manĝis?"
                            answered from events[i].roles.

  - state change            "Post X-i la pordon, kia estis ĝia stato?"
                            answered from events[i].property_changes.

  - location at start       "Kie estis la Y komence?"
                            answered from setup_relations.

  - sequencing              "Kio okazis post la prenado?"
                            answered by walking events forward.

Usage:
    python scripts/generate_icl_from_traces.py \\
        --in  data/causal_corpus/sample_10000_rich_v27.jsonl \\
        --out data/causal_corpus/sample_10000_icl_v27.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# Esperanto question/answer phrasings — kept compact; the SFT
# trainer will tokenize these as ordinary text.

CARDINALS_EO = [
    "nul", "unu", "du", "tri", "kvar", "kvin",
    "ses", "sep", "ok", "naŭ", "dek",
]


def _load_unmarked() -> dict[str, str]:
    """Load unmarked (default) slot values from the lexicon. The
    realizer doesn't surface these in prose — Q/A whose answer is
    the unmarked value trains the model to guess, not read."""
    from esperanto_lm.ontology import load_lexicon
    lex = load_lexicon()
    return {
        name: slot.unmarked
        for name, slot in lex.slots.items()
        if getattr(slot, "unmarked", None) is not None
    }


UNMARKED: dict[str, str] = {}  # populated lazily on first use
SKIP_VERBS: set[str] = set()  # populated lazily on first use


def _load_skip_verbs() -> set[str]:
    """Verbs to skip in Q/A generation: cascade-only reactive events
    + agentless weather verbs (pluvi, neĝi) that dominate first/last
    event answers."""
    from esperanto_lm.ontology import load_lexicon
    lex = load_lexicon()
    skip = {
        a.lemma for a in lex.actions.values()
        if getattr(a, "cascade_only", False)
    }
    # Agentless weather verbs: no "agent" role → not useful for
    # who/what Q/A and overrepresented in first/last events.
    for a in lex.actions.values():
        if not any(r.name == "agent" for r in a.roles):
            skip.add(a.lemma)
    return skip


def _should_skip_verb(verb: str) -> bool:
    """True if verb is engine-internal or cascade-only."""
    global SKIP_VERBS, _ALL_ACTIONS
    if not SKIP_VERBS:
        SKIP_VERBS = _load_skip_verbs()
    if not _ALL_ACTIONS:
        from esperanto_lm.ontology import load_lexicon
        _ALL_ACTIONS = set(load_lexicon().actions.keys())
    if verb.startswith("_"):
        return True
    if verb not in _ALL_ACTIONS:
        return True
    return verb in SKIP_VERBS


_ALL_ACTIONS: set[str] = set()


def _past(verb: str) -> str:
    """Esperanto past-tense form. Strip infinitive -i, add -is.
    Naive — assumes regular verbs, which is all our action lemmas."""
    if verb.endswith("i"):
        return verb[:-1] + "is"
    return verb + "is"


def _noun_acc(noun: str) -> str:
    """Accusative ending: -o → -on, -oj → -ojn. Leaves names/
    pronouns alone (caller is responsible)."""
    if noun.endswith("oj"):
        return noun + "n"
    if noun.endswith("o"):
        return noun + "n"
    return noun


def _name(eid: str, entities: dict) -> str:
    """Surface form for an entity: concept lemma. Capitalize for
    person types (proper nouns). Falls back to eid."""
    ent = entities.get(eid)
    if ent is None:
        return eid
    lemma = ent["concept"]
    if ent["type"] == "person":
        return lemma.capitalize()
    return lemma


def _acc(noun: str) -> str:
    """Accusative ending for an Esperanto noun. Naive: -o → -on,
    -oj → -ojn. Leaves names alone."""
    if noun and noun[0].isupper():
        # Proper noun — apply -n directly
        return noun + "n" if not noun.endswith("n") else noun
    if noun.endswith("oj"):
        return noun + "n"
    if noun.endswith("o"):
        return noun + "n"
    return noun


def _q_intrinsic_property(rec: dict, rng: random.Random) -> list[dict]:
    """For each entity with a notable observable property (color,
    posture, openness), emit a Q/A. Skips body parts (eid contains
    '_'-suffix substrings) to keep questions about top-level items.
    """
    global UNMARKED
    if not UNMARKED:
        UNMARKED = _load_unmarked()
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    interesting_slots = ["koloro", "posture", "openness",
                         "fullness", "lock_state", "power_state",
                         "cleanliness"]
    for ent in rec["entities"]:
        if "_" in ent["eid"] and ent["eid"] != ent["concept"]:
            continue  # body part / sub-component
        if ent["type"] in ("location", "abstract"):
            continue
        for slot in interesting_slots:
            vals = ent["properties"].get(slot)
            if not vals:
                continue
            val = vals[0]
            # Skip unmarked (default) values — the realizer doesn't
            # surface them in prose, so the model can't extract them.
            if val == UNMARKED.get(slot):
                continue
            name = _name(ent["eid"], entities)
            if slot == "koloro":
                q = f"Kia estis la koloro de la {ent['concept']}?"
                # koloro slot already stores the adjective form
                # (ruĝa, blua, blanka …); do NOT add another -a.
                a = val
            elif slot == "posture":
                q = f"En kia pozicio estis la {ent['concept']}?"
                a = val
            elif slot == "openness":
                q = f"Ĉu la {ent['concept']} estis malfermita aŭ fermita?"
                a = val
            elif slot == "fullness":
                q = f"Ĉu la {ent['concept']} estis plena aŭ malplena?"
                a = val
            elif slot == "lock_state":
                q = f"Ĉu la {ent['concept']} estis ŝlosita aŭ malŝlosita?"
                a = val
            elif slot == "power_state":
                q = f"Ĉu la {ent['concept']} estis aktiva aŭ neaktiva?"
                a = val
            elif slot == "cleanliness":
                q = f"Ĉu la {ent['concept']} estis pura aŭ malpura?"
                a = val
            else:
                continue
            out.append({"q": q, "a": a})
    return out


def _q_first_last(rec: dict, rng: random.Random) -> list[dict]:
    """First and last verb in the event sequence."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []

    def describe(ev):
        a = ev["roles"].get("agent")
        if a is None:
            return f"{ev['action']}"
        agent_name = _name(a, entities)
        theme = ev["roles"].get("theme")
        if theme is None:
            return f"{agent_name} {_past(ev['action'])}"
        # theme may be a list (fari.parts) — collapse
        if isinstance(theme, list):
            theme_name = ", ".join(_name(t, entities) for t in theme)
        else:
            theme_name = _name(theme, entities)
        return f"{agent_name} {_past(ev['action'])} {theme_name}"

    first = next((e for e in events if not _should_skip_verb(e["action"])), None)
    last = next((e for e in reversed(events) if not _should_skip_verb(e["action"])), None)
    if first is None:
        return []
    out.append({
        "q": "Kio okazis unue en la rakonto?",
        "a": describe(first) + ".",
    })
    if last is not None and last is not first:
        out.append({
            "q": "Kio okazis laste en la rakonto?",
            "a": describe(last) + ".",
        })
    return out


def _q_action_attribution(rec: dict, rng: random.Random) -> list[dict]:
    """Per content event, ask "who did X to Y" and / or
    "what did Z do"."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []

    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        agent = ev["roles"].get("agent")
        theme = ev["roles"].get("theme")
        if agent is None or theme is None:
            continue
        if isinstance(theme, list):
            continue  # list themes (fari.parts) handled elsewhere
        agent_ent = entities.get(agent)
        theme_ent = entities.get(theme)
        if agent_ent is None or theme_ent is None:
            continue
        agent_name = _name(agent, entities)
        theme_name = _name(theme, entities)
        # "Who did X-i the Y?" (theme is accusative in Esperanto)
        out.append({
            "q": (f"Kiu {_past(ev['action'])} la "
                  f"{_noun_acc(theme_ent['concept'])}?"),
            "a": agent_name + ".",
        })
        # "What did Z X-i?"
        if agent_ent["type"] == "person":
            out.append({
                "q": f"Kion {agent_name} {_past(ev['action'])}?",
                "a": "la " + _noun_acc(theme_ent["concept"]) + ".",
            })
        # "What did Z do to the Y?" — verb extraction
        if agent_ent["type"] == "person":
            shapes = [
                f"Kion {agent_name} faris al la {theme_ent['concept']}?",
                f"Kion {agent_name} faris kun la {theme_ent['concept']}?",
            ]
            out.append({
                "q": shapes[rng.randrange(len(shapes))],
                "a": f"{_past(ev['action'])} ĝin.",
            })
    return out


def _q_state_change(rec: dict, rng: random.Random) -> list[dict]:
    """Property-change questions: "After X did Y, what state was Z in?"
    """
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        if not ev.get("property_changes"):
            continue
        for key, new_val in ev["property_changes"].items():
            if "|" not in key:
                continue
            eid, slot = key.split("|", 1)
            ent = entities.get(eid)
            if ent is None:
                continue
            # Skip slots we don't have natural Esperanto phrasings for
            if slot not in ("openness", "fullness", "lock_state",
                            "power_state", "cleanliness", "posture",
                            "wetness", "temperature", "presence"):
                continue
            verb = ev["action"]
            theme = ev["roles"].get("theme")
            if theme and isinstance(theme, str):
                theme_ent_pc = entities.get(theme)
                if theme_ent_pc is None:
                    continue
                theme_name = _noun_acc(theme_ent_pc["concept"])
                q = (f"Post kiam la aganto {_past(verb)} la "
                     f"{theme_name}, kia estis la stato de la "
                     f"{ent['concept']}?")
            else:
                q = (f"Kio okazis al la {ent['concept']} post la "
                     f"ago?")
            out.append({"q": q, "a": str(new_val)})
    return out


def _q_location_at_start(rec: dict, rng: random.Random) -> list[dict]:
    """"Where was X at the start?" — from setup_relations. Includes
    both `en` (in) and `sur` (on) placements so the model sees both
    prepositions; `apud` (next to) likewise. Picks the preposition
    based on the actual asserted relation."""
    setup = rec.get("setup_relations", [])
    if not setup:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    # Vary the question shape by preposition so the model learns
    # to answer "Sur kio...", "Apud kio...", "Ĉe kio..." as well
    # as the default "Kie estis...".
    prep_for_rel = {"en": "En", "sur": "Sur", "apud": "Apud"}
    q_shapes = {
        "en": [
            ("Kie estis la {x} komence?", "En la {y}."),
            ("En kio estis la {x}?", "En la {y}."),
        ],
        "sur": [
            ("Sur kio estis la {x}?", "Sur la {y}."),
            ("Kie estis la {x} komence?", "Sur la {y}."),
        ],
        "apud": [
            ("Apud kio estis la {x}?", "Apud la {y}."),
            ("Ĉe kio estis la {x}?", "Ĉe la {y}."),
            ("Kie estis la {x} komence?", "Apud la {y}."),
        ],
    }
    out = []
    for r in setup:
        if r["relation"] not in prep_for_rel:
            continue
        if len(r["args"]) != 2:
            continue
        contained, container = r["args"]
        c_ent = entities.get(contained)
        co_ent = entities.get(container)
        if c_ent is None or co_ent is None:
            continue
        if c_ent["type"] in ("location",):
            continue
        if "_" in contained and contained != c_ent["concept"]:
            continue  # body part
        shapes = q_shapes.get(r["relation"], [])
        shape = shapes[rng.randrange(len(shapes))] if shapes else None
        if shape is None:
            continue
        q_tmpl, a_tmpl = shape
        out.append({
            "q": q_tmpl.format(x=c_ent["concept"], y=co_ent["concept"]),
            "a": a_tmpl.format(x=c_ent["concept"], y=co_ent["concept"]),
        })
    return out


def _q_instrument_and_parts(rec: dict, rng: random.Random) -> list[dict]:
    """For events with an instrument and/or parts, ask about the
    tool used and/or the materials. Covers three shapes:

      - instrument only (kuiri per forno): "Per kio X kuiris?"
      - parts only (fari without crafted_with): "El kio oni faris la X?"
      - both (fari per najlilo el ligno+najlo): combined Q/A.

    Skips body-part instruments (mano, okulo) — only functional
    tools (forno, martelo, ŝlosilo) read naturally."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for ev in events:
        agent = ev["roles"].get("agent")
        if agent is None or isinstance(agent, list):
            continue
        agent_name = _name(agent, entities)

        # Resolve instrument (if any, and if it's a real tool)
        instr = ev["roles"].get("instrument")
        instr_name = None
        if instr and isinstance(instr, str):
            instr_ent = entities.get(instr)
            if instr_ent is not None:
                if "_" in instr and instr != instr_ent["concept"]:
                    instr = None  # body part
                elif instr_ent["type"] not in ("artifact", "substance"):
                    instr = None
                else:
                    instr_name = instr_ent["concept"]

        # Resolve parts list (if any)
        parts = ev["roles"].get("parts")
        part_names: list[str] = []
        if parts and isinstance(parts, list):
            for p in parts:
                p_ent = entities.get(p)
                if p_ent is not None:
                    part_names.append(p_ent["concept"])

        if not instr_name and not part_names:
            continue

        # Theme for the question phrasing
        theme = ev["roles"].get("theme")
        theme_phrase = ""
        if theme and isinstance(theme, str):
            theme_ent = entities.get(theme)
            if theme_ent is not None:
                theme_phrase = f" la {_noun_acc(theme_ent['concept'])}"

        verb = _past(ev["action"])

        # Format parts as "el X kaj Y"
        if part_names:
            if len(part_names) == 1:
                parts_phrase = part_names[0]
            elif len(part_names) == 2:
                parts_phrase = f"{part_names[0]} kaj {part_names[1]}"
            else:
                parts_phrase = (", ".join(part_names[:-1])
                               + f", kaj {part_names[-1]}")

        if instr_name and part_names:
            # Both tool and materials
            out.append({
                "q": (f"Per kio kaj el kio {agent_name} "
                      f"{verb}{theme_phrase}?"),
                "a": f"Per {instr_name}, el {parts_phrase}.",
            })
            # Also split into individual questions
            out.append({
                "q": f"Per kio {agent_name} {verb}{theme_phrase}?",
                "a": f"per {instr_name}",
            })
            out.append({
                "q": f"El kio {agent_name} {verb}{theme_phrase}?",
                "a": f"el {parts_phrase}",
            })
            # Count ingredients
            n = len(part_names)
            if n < len(CARDINALS_EO):
                out.append({
                    "q": (f"Kiom da ingrediencoj bezoniĝis por "
                          f"{ev['action']}{theme_phrase}?"),
                    "a": CARDINALS_EO[n],
                })
        elif instr_name:
            # Tool only
            out.append({
                "q": f"Per kio {agent_name} {verb}{theme_phrase}?",
                "a": f"per {instr_name}",
            })
        elif part_names:
            # Materials only (e.g. sandviĉo without crafted_with)
            out.append({
                "q": f"El kio {agent_name} {verb}{theme_phrase}?",
                "a": f"el {parts_phrase}",
            })
            n = len(part_names)
            if n < len(CARDINALS_EO):
                out.append({
                    "q": (f"Kiom da ingrediencoj bezoniĝis por "
                          f"{ev['action']}{theme_phrase}?"),
                    "a": CARDINALS_EO[n],
                })
    return out


def _q_count(rec: dict, rng: random.Random) -> list[dict]:
    """For entities with count > 1, ask "Kiom da X estis?"."""
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for ent in rec["entities"]:
        if "_" in ent["eid"] and ent["eid"] != ent["concept"]:
            continue
        if ent["type"] in ("location", "abstract"):
            continue
        count_vals = ent["properties"].get("count")
        if not count_vals:
            continue
        try:
            n = int(count_vals[0])
        except (ValueError, TypeError):
            continue
        if n <= 1 or n >= len(CARDINALS_EO):
            continue
        out.append({
            "q": f"Kiom da {ent['concept']}j estis?",
            "a": CARDINALS_EO[n],
        })
    return out


def _q_why(rec: dict, rng: random.Random) -> list[dict]:
    """Causal "Kial X-iĝis? Ĉar Y." from `event.caused_by`. Each
    event that lists a causing event id gets a Q/A whose answer
    points to the cause's verb (and optionally its theme)."""
    events = rec.get("events", [])
    if not events:
        return []
    by_id = {ev["id"]: ev for ev in events if "id" in ev}
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []

    def describe_short(ev):
        a = ev["roles"].get("agent") or ev["roles"].get("theme")
        if a is None:
            return _past(ev["action"])
        ent = entities.get(a) if isinstance(a, str) else None
        if ent is None:
            return _past(ev["action"])
        name = _name(a, entities)
        return f"{name} {_past(ev['action'])}"

    for ev in events:
        causes = ev.get("caused_by") or []
        if not causes:
            continue
        if _should_skip_verb(ev["action"]):
            continue
        cause_id = causes[0]
        cause = by_id.get(cause_id)
        if cause is None or _should_skip_verb(cause["action"]):
            continue
        # Build "Kial <effect>? Ĉar <cause>."
        effect_phrase = describe_short(ev)
        cause_phrase = describe_short(cause)
        if not effect_phrase or not cause_phrase:
            continue
        out.append({
            "q": f"Kial {effect_phrase}?",
            "a": f"Ĉar {cause_phrase}.",
        })
    return out


def _q_possession(rec: dict, rng: random.Random) -> list[dict]:
    """Who had what at scene start: "Kiu havis la X-on?" → "Y."
    and inverse "Kion Y havis?" → "la X-on." From havi in
    setup_relations."""
    setup = rec.get("setup_relations", [])
    if not setup:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for r in setup:
        if r["relation"] != "havi" or len(r["args"]) != 2:
            continue
        owner_eid, item_eid = r["args"]
        owner_ent = entities.get(owner_eid)
        item_ent = entities.get(item_eid)
        if owner_ent is None or item_ent is None:
            continue
        if "_" in item_eid and item_eid != item_ent["concept"]:
            continue
        owner_name = _name(owner_eid, entities)
        item_name = item_ent["concept"]
        # "Kiu havis la X-on?"
        out.append({
            "q": f"Kiu havis la {_noun_acc(item_name)}?",
            "a": f"{owner_name}.",
        })
        # "Kion Y havis?"
        if owner_ent["type"] == "person":
            out.append({
                "q": f"Kion {owner_name} havis?",
                "a": f"la {_noun_acc(item_name)}.",
            })
    return out


def _q_container_contents(rec: dict, rng: random.Random) -> list[dict]:
    """What was inside a container: "Kio estis en la glaso?" → "akvo."
    From en(content, container) in setup_relations where the
    container is a non-location artifact (glaso, korbo, botelo)."""
    setup = rec.get("setup_relations", [])
    if not setup:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for r in setup:
        if r["relation"] != "en" or len(r["args"]) != 2:
            continue
        content_eid, container_eid = r["args"]
        content_ent = entities.get(content_eid)
        container_ent = entities.get(container_eid)
        if content_ent is None or container_ent is None:
            continue
        # Only non-location containers (glaso, korbo, botelo, …)
        if container_ent["type"] == "location":
            continue
        if "_" in content_eid and content_eid != content_ent["concept"]:
            continue
        out.append({
            "q": f"Kio estis en la {container_ent['concept']}?",
            "a": f"{content_ent['concept']}.",
        })
    return out


def _q_existence(rec: dict, rng: random.Random, *,
                  all_concepts: frozenset[str] | None = None,
                  ) -> list[dict]:
    """Boolean existence: "Ĉu estis X en la sceno?" → "Jes." / "Ne."
    Generates both positive (entity IS in scene) and negative
    (entity concept NOT in scene) so the model learns both answers.
    Negative candidates drawn from `all_concepts` (the full corpus
    concept pool) minus what's present in this trace — no hardcoded
    list."""
    part_eids = {
        r["args"][1] for r in rec.get("setup_relations", [])
        if r["relation"] == "havas_parton" and len(r["args"]) == 2
    }
    present: set[str] = set()
    out = []
    for ent in rec["entities"]:
        if ent["eid"] in part_eids or ent["eid"] == "mondo":
            continue
        if ent["type"] in ("location", "abstract"):
            continue
        if "_" in ent["eid"] and ent["eid"] != ent["concept"]:
            continue
        present.add(ent["concept"])
    def _yes(concept: str) -> str:
        return rng.choice([
            "Jes.",
            "Jes, estis.",
            f"Jes, estis {concept} en la sceno.",
            f"Jes, {concept} estis en la sceno.",
        ])

    def _no(concept: str) -> str:
        return rng.choice([
            "Ne.",
            "Ne, ne estis.",
            f"Ne, ne estis {concept} en la sceno.",
            f"Ne, {concept} ne estis en la sceno.",
        ])

    # Sample up to 2 positive
    pos_list = list(present)
    rng.shuffle(pos_list)
    for concept in pos_list[:2]:
        out.append({
            "q": f"Ĉu estis {concept} en la sceno?",
            "a": _yes(concept),
        })
    # Negative: concepts that exist in the corpus but not this trace
    if all_concepts:
        absent = list(all_concepts - present)
        rng.shuffle(absent)
        for concept in absent[:2]:
            out.append({
                "q": f"Ĉu estis {concept} en la sceno?",
                "a": _no(concept),
            })
    return out


def _passive_participle(verb: str) -> str:
    """Esperanto passive past participle: stem + -ita.
    manĝi → manĝita, preni → prenita, fermi → fermita."""
    if verb.endswith("i"):
        return verb[:-1] + "ita"
    return verb + "ita"


def _q_location_contents(rec: dict, rng: random.Random) -> list[dict]:
    """What's at a location: "Kio troviĝas en la kuirejo?" → list of
    entities. Trains the "Kio estas/troviĝas en X?" pattern that
    maps to the wiki "Kio estas la ĉefurbo de X?" shape."""
    setup = rec.get("setup_relations", [])
    if not setup:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    part_eids = {
        r["args"][1] for r in setup
        if r["relation"] == "havas_parton" and len(r["args"]) == 2
    }
    # Group non-part, non-location entities by their container location
    by_loc: dict[str, list[str]] = {}
    for r in setup:
        if r["relation"] != "en" or len(r["args"]) != 2:
            continue
        contained, container = r["args"]
        if contained in part_eids:
            continue
        c_ent = entities.get(contained)
        co_ent = entities.get(container)
        if c_ent is None or co_ent is None:
            continue
        if c_ent["type"] in ("location", "abstract"):
            continue
        if co_ent["type"] != "location":
            continue
        if "_" in contained and contained != c_ent["concept"]:
            continue
        by_loc.setdefault(container, []).append(c_ent["concept"])
    out = []
    for loc_eid, concepts in by_loc.items():
        if len(concepts) < 2:
            continue
        loc_ent = entities.get(loc_eid)
        if loc_ent is None:
            continue
        items = sorted(set(concepts))[:5]
        if len(items) == 1:
            listing = items[0]
        elif len(items) == 2:
            listing = f"{items[0]} kaj {items[1]}"
        else:
            listing = ", ".join(items[:-1]) + f", kaj {items[-1]}"
        q_shapes = [
            f"Kio troviĝas en la {loc_ent['concept']}?",
            f"Kio estas en la {loc_ent['concept']}?",
        ]
        out.append({
            "q": q_shapes[rng.randrange(len(q_shapes))],
            "a": listing + ".",
        })
    return out


def _q_container_identity(rec: dict, rng: random.Random) -> list[dict]:
    """Inverse of container_contents: "En kio estis la akvo?" → "en
    la glaso." Trains extraction of the container from a content
    entity."""
    setup = rec.get("setup_relations", [])
    if not setup:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for r in setup:
        if r["relation"] != "en" or len(r["args"]) != 2:
            continue
        contained, container = r["args"]
        c_ent = entities.get(contained)
        co_ent = entities.get(container)
        if c_ent is None or co_ent is None:
            continue
        # Only non-location containers (glaso, korbo, botelo)
        if co_ent["type"] == "location":
            continue
        if "_" in contained and contained != c_ent["concept"]:
            continue
        out.append({
            "q": f"En kio estis la {c_ent['concept']}?",
            "a": f"En la {co_ent['concept']}.",
        })
    return out


def _q_consequence(rec: dict, rng: random.Random) -> list[dict]:
    """What happened to a theme entity — active or passive voice.
    Purely grammatical: derives answer from the verb, no hardcoded
    consequence mappings.
      active:  "Dentisto manĝis ĝin."
      passive: "Ĝi estis manĝita."
    """
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        agent = ev["roles"].get("agent")
        theme = ev["roles"].get("theme")
        if not theme or isinstance(theme, list):
            continue
        theme_ent = entities.get(theme)
        if theme_ent is None:
            continue
        if "_" in theme and theme != theme_ent["concept"]:
            continue
        theme_name = theme_ent["concept"]
        verb = ev["action"]

        answers = []
        if agent:
            answers.append(f"{_name(agent, entities)} {_past(verb)} ĝin.")
        answers.append(f"Ĝi estis {_passive_participle(verb)}.")

        q = rng.choice([
            f"Kio okazis al la {theme_name}?",
            f"Kio okazis kun la {theme_name}?",
        ])
        out.append({"q": q, "a": rng.choice(answers)})
    return out


def _q_movement(rec: dict, rng: random.Random) -> list[dict]:
    """For movement events (iri, kuri, veni, eniri, flugi), ask
    "Kien X iris?" → "Al la Y." Varies question shape: Kien,
    Al kiu loko, Ĉe kiu loko."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    move_verbs = {"iri", "kuri", "veni", "eniri", "flugi"}
    out = []
    for ev in events:
        if ev["action"] not in move_verbs:
            continue
        agent = ev["roles"].get("agent")
        dest = ev["roles"].get("destination") or ev["roles"].get("theme")
        if not agent or not dest or isinstance(dest, list):
            continue
        agent_ent = entities.get(agent)
        dest_ent = entities.get(dest)
        if agent_ent is None or dest_ent is None:
            continue
        if dest_ent["type"] != "location":
            continue
        agent_name = _name(agent, entities)
        shapes = [
            (f"Kien {agent_name} {_past(ev['action'])}?",
             f"Al la {dest_ent['concept']}."),
            (f"Al kiu loko {agent_name} {_past(ev['action'])}?",
             f"Al la {dest_ent['concept']}."),
        ]
        q, a = shapes[rng.randrange(len(shapes))]
        out.append({"q": q, "a": a})
    return out


def _q_recipient(rec: dict, rng: random.Random) -> list[dict]:
    """For transfer events (doni, montri, instrui, rakonti), ask
    "Al kiu X donis la Y?" → "Al Z." """
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    transfer_verbs = {"doni", "montri", "instrui", "rakonti",
                      "demandi", "respondi"}
    out = []
    for ev in events:
        if ev["action"] not in transfer_verbs:
            continue
        agent = ev["roles"].get("agent")
        recipient = ev["roles"].get("recipient")
        if not agent or not recipient:
            continue
        if isinstance(agent, list) or isinstance(recipient, list):
            continue
        agent_ent = entities.get(agent)
        recip_ent = entities.get(recipient)
        if agent_ent is None or recip_ent is None:
            continue
        agent_name = _name(agent, entities)
        recip_name = _name(recipient, entities)
        theme = ev["roles"].get("theme")
        if theme and isinstance(theme, str):
            theme_ent = entities.get(theme)
            if theme_ent is not None:
                out.append({
                    "q": (f"Al kiu {agent_name} {_past(ev['action'])} "
                          f"la {_noun_acc(theme_ent['concept'])}?"),
                    "a": f"Al {recip_name}.",
                })
        else:
            out.append({
                "q": f"Al kiu {agent_name} {_past(ev['action'])}?",
                "a": f"Al {recip_name}.",
            })
    return out


def _q_category_count(rec: dict, rng: random.Random) -> list[dict]:
    """Count entities by category: "Kiom da mebloj estis?" → "tri".
    Groups non-part, non-location entities by their entity type,
    emits a count Q/A for types with ≥2 members."""
    part_eids = {
        r["args"][1] for r in rec.get("setup_relations", [])
        if r["relation"] == "havas_parton" and len(r["args"]) == 2
    }
    type_labels = {
        "person": "personoj",
        "animal": "bestoj",
        "artifact": "mebloj",
        "substance": "substancoj",
    }
    by_type: dict[str, int] = {}
    for ent in rec["entities"]:
        if ent["eid"] in part_eids or ent["eid"] == "mondo":
            continue
        if ent["type"] in ("location", "abstract"):
            continue
        if "_" in ent["eid"] and ent["eid"] != ent["concept"]:
            continue
        label = type_labels.get(ent["type"])
        if label is None:
            continue
        count_vals = ent["properties"].get("count")
        try:
            n = int(count_vals[0]) if count_vals else 1
        except (ValueError, TypeError):
            n = 1
        by_type[label] = by_type.get(label, 0) + n
    out = []
    for label, total in by_type.items():
        if total < 2 or total >= len(CARDINALS_EO):
            continue
        out.append({
            "q": f"Kiom da {label} estis en la sceno?",
            "a": CARDINALS_EO[total],
        })
    return out


def _q_verb_count(rec: dict, rng: random.Random) -> list[dict]:
    """Count how many times specific verbs fire in the chain.
    "Kiom da fojoj la aganto prenis ion?" → "tri".
    Also total event count and distinct-concept counts.
    Produces higher numbers (5-15) than entity counts, balancing
    the du/tri bias in the training distribution."""
    events = rec.get("events", [])
    if len(events) < 3:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []

    # Total meaningful events
    meaningful = [e for e in events if not _should_skip_verb(e["action"])]
    n_total = len(meaningful)
    if 2 <= n_total < len(CARDINALS_EO):
        shapes = [
            f"Kiom da agoj okazis en la rakonto?",
            f"Kiom da eventoj okazis entute?",
        ]
        out.append({
            "q": shapes[rng.randrange(len(shapes))],
            "a": CARDINALS_EO[n_total],
        })

    # Per-verb counts — any verb appearing ≥2 times is countable.
    from collections import Counter
    verb_counts = Counter(
        e["action"] for e in events if not _should_skip_verb(e["action"]))
    for verb, n in verb_counts.items():
        if n < 2 or n >= len(CARDINALS_EO):
            continue
        shapes = [
            f"Kiom da fojoj iu {_past(verb)} en la rakonto?",
            f"Kiom da fojoj okazis {verb}?",
        ]
        out.append({
            "q": shapes[rng.randrange(len(shapes))],
            "a": CARDINALS_EO[n],
        })

    # Distinct concepts by type
    part_eids = {
        r["args"][1] for r in rec.get("setup_relations", [])
        if r["relation"] == "havas_parton" and len(r["args"]) == 2
    }
    by_type: dict[str, set] = {}
    for ent in rec["entities"]:
        if ent["eid"] in part_eids or ent["eid"] == "mondo":
            continue
        if ent["type"] in ("location", "abstract"):
            continue
        if "_" in ent["eid"] and ent["eid"] != ent["concept"]:
            continue
        by_type.setdefault(ent["type"], set()).add(ent["concept"])
    type_labels = {
        "person": "malsamaj personoj",
        "animal": "malsamaj bestoj",
        "artifact": "malsamaj objektoj",
        "substance": "malsamaj substancoj",
    }
    for etype, concepts in by_type.items():
        n = len(concepts)
        label = type_labels.get(etype)
        if label is None or n < 2 or n >= len(CARDINALS_EO):
            continue
        out.append({
            "q": f"Kiom da {label} estis en la sceno?",
            "a": CARDINALS_EO[n],
        })

    return out


def _q_multi_hop(rec: dict, rng: random.Random) -> list[dict]:
    """Two-step reasoning: find an event's agent, then look up a
    property of that agent from the scene.

    "Kie estis la persono kiu prenis la glason?" — find preni(agent,
    glaso) → agent=Dentisto → find en(Dentisto, kuirejo) → "En la
    kuirejo."

    Only fires when the agent's location is unambiguous in
    setup_relations."""
    events = rec.get("events", [])
    setup = rec.get("setup_relations", [])
    if not events or not setup:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}

    # Build agent→location map from setup en-relations
    agent_loc: dict[str, str] = {}
    for r in setup:
        if r["relation"] != "en" or len(r["args"]) != 2:
            continue
        contained, container = r["args"]
        c_ent = entities.get(contained)
        co_ent = entities.get(container)
        if c_ent is None or co_ent is None:
            continue
        if c_ent["type"] not in ("person", "animal"):
            continue
        if co_ent["type"] != "location":
            continue
        agent_loc[contained] = container

    out = []
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        agent = ev["roles"].get("agent")
        theme = ev["roles"].get("theme")
        if not agent or not theme or isinstance(theme, list):
            continue
        if agent not in agent_loc:
            continue
        agent_ent = entities.get(agent)
        theme_ent = entities.get(theme)
        loc_ent = entities.get(agent_loc[agent])
        if agent_ent is None or theme_ent is None or loc_ent is None:
            continue
        if "_" in theme and theme != theme_ent["concept"]:
            continue

        q_shapes = [
            (f"Kie estis la persono kiu {_past(ev['action'])} "
             f"la {_noun_acc(theme_ent['concept'])}?",
             f"En la {loc_ent['concept']}."),
            (f"En kiu loko estis tiu kiu {_past(ev['action'])} "
             f"la {_noun_acc(theme_ent['concept'])}?",
             f"En la {loc_ent['concept']}."),
        ]
        q, a = q_shapes[rng.randrange(len(q_shapes))]
        out.append({"q": q, "a": a})
    return out


_CONCEPT_CATEGORIES: dict[str, list[str]] = {}


def _load_concept_categories() -> dict[str, list[str]]:
    """Load concept → category list from the lex. Used by coreference
    to map "la besto" → the concept that has category=besto."""
    from esperanto_lm.ontology import load_lexicon
    lex = load_lexicon()
    return {
        name: list(c.category)
        for name, c in lex.concepts.items()
        if c.category
    }


def _q_coreference(rec: dict, rng: random.Random) -> list[dict]:
    """Alias resolution: the realizer refers to entities by category
    aliases ("la besto" for ĉimpanzo, "la ujo" for glaso, "la
    trinkaĵo" for kafo). Ask what the alias refers to.

    "Kio estas 'la besto' en la rakonto?" → "ĉimpanzo."
    "Kio estas 'la ujo' en la rakonto?" → "glaso."
    """
    global _CONCEPT_CATEGORIES
    if not _CONCEPT_CATEGORIES:
        _CONCEPT_CATEGORIES = _load_concept_categories()

    entities = {e["eid"]: e for e in rec["entities"]}
    part_eids = {
        r["args"][1] for r in rec.get("setup_relations", [])
        if r["relation"] == "havas_parton" and len(r["args"]) == 2
    }
    out = []
    for ent in rec["entities"]:
        if ent["eid"] in part_eids or ent["eid"] == "mondo":
            continue
        if ent["type"] in ("location", "abstract"):
            continue
        if "_" in ent["eid"] and ent["eid"] != ent["concept"]:
            continue
        concept = ent["concept"]
        cats = _CONCEPT_CATEGORIES.get(concept, [])
        if not cats:
            continue
        # Pick one category as the alias
        alias = cats[0]
        # Skip if alias == concept (no aliasing)
        if alias == concept:
            continue
        out.append({
            "q": f"Kio estas 'la {alias}' en la rakonto?",
            "a": f"{concept}.",
        })
    return out


def _q_ordering(rec: dict, rng: random.Random) -> list[dict]:
    """Adjacent-pair "Kio okazis post X-ado?" questions over the
    event chain. Limited to verbs whose past form reads naturally
    (skipping intransitive _wet/pluvi/aperi cascade markers)."""
    events = rec.get("events", [])
    if len(events) < 2:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []

    def describe(ev):
        a = ev["roles"].get("agent")
        if a is None:
            return _past(ev["action"])
        agent_name = _name(a, entities)
        theme = ev["roles"].get("theme")
        if theme and isinstance(theme, str):
            theme_ent = entities.get(theme)
            if theme_ent is not None:
                return (f"{agent_name} {_past(ev['action'])} "
                        f"la {_noun_acc(theme_ent['concept'])}")
        return f"{agent_name} {_past(ev['action'])}"

    for i in range(len(events) - 1):
        prev, nxt = events[i], events[i + 1]
        if _should_skip_verb(prev["action"]) or _should_skip_verb(nxt["action"]):
            continue
        # Build "Kio okazis post la X-ado de la Y?" — use the
        # verb's noun form (action+o) so the question reads
        # idiomatically.
        prev_verb = prev["action"]
        if prev_verb.endswith("i"):
            prev_noun = prev_verb[:-1] + "ado"
        else:
            prev_noun = prev_verb + "ado"
        out.append({
            "q": f"Kio okazis post la {prev_noun}?",
            "a": describe(nxt) + ".",
        })
    return out


# Registry of question generators.
GENERATORS = [
    _q_intrinsic_property,
    _q_first_last,
    _q_action_attribution,
    _q_state_change,
    _q_location_at_start,
    _q_instrument_and_parts,
    _q_count,
    # _q_category_count — requires counting by type; not in prose.
    _q_location_contents,
    _q_container_identity,
    _q_consequence,
    _q_possession,
    _q_container_contents,
    _q_movement,
    _q_recipient,
    # _q_verb_count — requires counting verb occurrences; not in prose.
    _q_multi_hop,
    _q_coreference,
    _q_ordering,
    # _q_why — skipped: 95% of causal chains are pluvi→_wet,
    # producing "Ĉar pluvis." mode collapse. Needs richer causal
    # annotations in the engine before this template is useful.
]


def generate_qas_for_trace(
    rec: dict, rng: random.Random, max_per_trace: int = 4,
    all_concepts: frozenset[str] | None = None,
) -> list[dict]:
    """Yield up to max_per_trace Q/A pairs sampled across generators.
    Skipping empty generators; sampled uniformly so question types
    stay balanced.

    `all_concepts`: the full set of concept lemmas across ALL traces
    in the input file. Passed to generators that need a negative-
    sampling pool (e.g. _q_existence picks concepts NOT in this
    trace but known to exist in the corpus)."""
    candidates: list[dict] = []
    for gen in GENERATORS:
        if gen == _q_existence:
            candidates.extend(gen(rec, rng, all_concepts=all_concepts))
        else:
            candidates.extend(gen(rec, rng))
    if not candidates:
        return []
    rng.shuffle(candidates)
    seen_qs: set = set()
    picked: list[dict] = []
    for qa in candidates:
        if qa["q"] in seen_qs:
            continue
        seen_qs.add(qa["q"])
        picked.append(qa)
        if len(picked) >= max_per_trace:
            break
    return picked


def format_sft_record(prose: str, qa: dict) -> dict:
    """Wrap a (prose, Q, A) triple into the SFT conversation format
    `train_sft.py` expects."""
    return {
        "messages": [
            {"role": "user",
             "content": f"{prose}\n\nDemando: {qa['q']}"},
            {"role": "assistant", "content": qa["a"]},
        ]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-per-trace", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)

    # First pass: collect all concept lemmas across the corpus for
    # negative-sampling in existence questions. Excludes body parts
    # and locations — same filter as the per-trace generators.
    all_concepts: set[str] = set()
    with open(args.inp) as fin:
        for line in fin:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            for ent in rec.get("entities", []):
                if ent["type"] in ("location", "abstract"):
                    continue
                if "_" in ent["eid"] and ent["eid"] != ent["concept"]:
                    continue
                all_concepts.add(ent["concept"])
    all_concepts_frozen = frozenset(all_concepts)

    # Second pass: generate Q/A.
    n_traces = 0
    n_qas = 0
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            prose = rec.get("prose")
            if not prose:
                continue
            qas = generate_qas_for_trace(
                rec, rng, max_per_trace=args.max_per_trace,
                all_concepts=all_concepts_frozen)
            for qa in qas:
                fout.write(json.dumps(
                    format_sft_record(prose, qa),
                    ensure_ascii=False) + "\n")
                n_qas += 1
            n_traces += 1
    print(f"Wrote {n_qas} Q/A pairs from {n_traces} traces to {args.out}")


if __name__ == "__main__":
    main()
