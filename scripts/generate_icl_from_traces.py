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
from collections import Counter
from typing import Optional
from pathlib import Path


# Esperanto question/answer phrasings — kept compact; the SFT
# trainer will tokenize these as ordinary text.

def _cardinal_eo(n: int) -> str:
    if n == 0:
        return "nul"
    ones = ["", "unu", "du", "tri", "kvar", "kvin",
            "ses", "sep", "ok", "naŭ"]
    parts = []
    if n >= 100:
        h = n // 100
        parts.append("cent" if h == 1 else ones[h] + "cent")
        n %= 100
    if n >= 10:
        d = n // 10
        parts.append("dek" if d == 1 else ones[d] + "dek")
        n %= 10
    if n > 0:
        parts.append(ones[n])
    return " ".join(parts)


CARDINALS_EO = [_cardinal_eo(i) for i in range(101)]


def _load_unmarked() -> dict[str, str]:
    """Load unmarked (default) slot values from the lexicon. The
    realizer doesn't surface these in prose — Q/A whose answer is
    the unmarked value trains the model to guess, not read."""
    lex = _get_lex()
    return {
        name: slot.unmarked
        for name, slot in lex.slots.items()
        if getattr(slot, "unmarked", None) is not None
    }


UNMARKED: dict[str, str] = {}  # populated lazily on first use
SKIP_VERBS: set[str] = set()  # populated lazily on first use
_LEX = None


def _get_lex():
    global _LEX
    if _LEX is None:
        from esperanto_lm.ontology import load_lexicon
        _LEX = load_lexicon()
    return _LEX


def _load_skip_verbs() -> set[str]:
    """Verbs to skip in Q/A generation: cascade-only reactive events
    + agentless weather verbs (pluvi, neĝi) that dominate first/last
    event answers."""
    lex = _get_lex()
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
        _ALL_ACTIONS = set(_get_lex().actions.keys())
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
    """Surface form for an entity used in Q/A questions and answers.

    When the entity has a proper name (eid distinct from the concept
    lemma, e.g., a Wikidata-spawned "petro_silva"), the realizer's
    `_render_person_name` shared helper produces "Petro Silva". The
    model then has to resolve aliases / pronouns the prose uses
    ("la kuracisto", "li") back to the named entity — coref training
    built into the answer choice.

    Falls back to the capitalized concept for persons without proper
    names ("kuracisto" → "Kuracisto"), and to the bare lemma for
    non-persons."""
    from esperanto_lm.ontology.realize.render import _render_person_name
    ent = entities.get(eid)
    if ent is None:
        return eid
    concept = ent["concept"]
    if ent["type"] == "person":
        if eid != concept:
            return _render_person_name(eid)
        return concept.capitalize()
    return concept


def _subject_phrase(ev: dict, entities: dict) -> str | None:
    """Render the subject of an event as it should appear in Q/A
    answers — joining agent + companion with "kaj" when both are
    bound, otherwise just the agent's name. Returns None when the
    agent isn't resolvable. Mirrors what the renderer does for
    companion-bearing events ("Petro kaj Maria dormis.")."""
    agent = ev["roles"].get("agent")
    if agent is None or agent not in entities:
        return None
    agent_name = _name(agent, entities)
    companion = ev["roles"].get("companion")
    if companion and isinstance(companion, str) and companion in entities:
        companion_name = _name(companion, entities)
        return f"{agent_name} kaj {companion_name}"
    return agent_name


_CARDINALS_ACC_LOOKUP = None


def _disclosed_count(eid: str, sentence_facts: list) -> int | None:
    """Return the count value the renderer surfaced for `eid`, taken
    from disclosed `count` facts in `sentence_facts`. None when the
    prose never said how many. Picks the LAST disclosed count so
    consumption-verb quantity (e.g. manĝi event qty) shadows the
    entity's natural count, mirroring what the prose actually said."""
    found: int | None = None
    for _idx, facts in sentence_facts or []:
        for f in facts:
            if f.get("kind") != "count":
                continue
            if f.get("entity") != eid:
                continue
            try:
                found = int(f.get("value"))
            except (TypeError, ValueError):
                pass
    return found


def _disclosed_adjective(eid: str, sentence_facts: list) -> str | None:
    """Return an attributive adjective the renderer surfaced for
    `eid`. Pulls the most-recent `state` fact with phase=attributive.
    None when no adjective was rendered."""
    found: str | None = None
    for _idx, facts in sentence_facts or []:
        for f in facts:
            if f.get("kind") != "state":
                continue
            if f.get("entity") != eid:
                continue
            if f.get("phase") != "attributive":
                continue
            v = f.get("value")
            if isinstance(v, str):
                found = v
    return found


_SPECIFIER_REL_CACHE: dict | None = None


def _specifier_rel_schema() -> dict:
    """Build a per-relation-name index of (head_pos, anchor_pos, prep)
    for relations the lexicon marks `specifier_of=[h, a]`. Cached per
    lexicon load — same source the renderer uses (Relation schema).
    Used by `_specifier_noun_phrase` to surface "la libro sur la
    tablo" / "la glaso de Petro" style anchored phrases."""
    global _SPECIFIER_REL_CACHE
    if _SPECIFIER_REL_CACHE is not None:
        return _SPECIFIER_REL_CACHE
    lex = _get_lex()
    out: dict = {}
    for name, rel_def in lex.relations.items():
        if not getattr(rel_def, "specifier_of", None):
            continue
        h, a = rel_def.specifier_of[0], rel_def.specifier_of[1]
        prep = getattr(rel_def, "specifier_preposition", None) or name
        out[name] = (h, a, prep)
    _SPECIFIER_REL_CACHE = out
    return out


def _disclosed_specifier(
    eid: str, sentence_facts: list, entities: dict,
) -> tuple[str, str] | None:
    """Return (preposition, anchor_phrase) for a specifier the
    renderer attached to `eid`, or None when no specifier was
    surfaced. Reads `relation` facts and checks each against the
    lexicon's `specifier_of` schema — same source the renderer's
    `_pick_specifier` consults. Picks the LAST disclosed specifier
    relation so back-references (later mentions) override earlier
    ones. Anchor name is built via `_name` (proper-noun aware)."""
    schema = _specifier_rel_schema()
    found: tuple[str, str] | None = None
    for _idx, facts in sentence_facts or []:
        for f in facts:
            if f.get("kind") != "relation":
                continue
            rel = f.get("rel")
            entry = schema.get(rel)
            if entry is None:
                continue
            h, a, prep = entry
            args = f.get("args") or ()
            if len(args) <= max(h, a):
                continue
            if args[h] != eid:
                continue
            anchor_eid = args[a]
            anchor_ent = entities.get(anchor_eid)
            if anchor_ent is None:
                continue
            anchor_name = _name(anchor_eid, entities)
            if anchor_ent["type"] == "person":
                found = (prep, anchor_name)
            else:
                found = (prep, f"la {anchor_name}")
    return found


def _specifier_noun_phrase(
    eid: str, entities: dict, sentence_facts: list,
    case: str = "acc", *, force_definite: bool = True,
) -> str | None:
    """Build a noun-phrase matching what the prose actually said for
    `eid` — pulling count and attributive adjective from disclosed
    facts. Returns `la {count?} {adj?} {noun}` (or accusative variant)
    or None when the entity isn't resolvable. Skips count/adj when
    they weren't disclosed (so the question doesn't claim modifiers
    the model can't extract). The bare-noun fallback matches what
    legacy generators emit.

    Examples (case=acc):
      prose: "Petro vidis tri grandajn glasojn"
        → "la tri grandajn glasojn"
      prose: "Petro manĝis la pomon"
        → "la pomon"
      prose: "Petro plenigis la ruĝan glason"
        → "la ruĝan glason"
    """
    ent = entities.get(eid)
    if ent is None:
        return None
    concept = ent["concept"]
    count = _disclosed_count(eid, sentence_facts)
    adj = _disclosed_adjective(eid, sentence_facts)
    # Partial-quantity check: when the consumption/transfer-verb
    # rendered count is LESS than the entity's natural count, the
    # phrase is indefinite — "unu pomon" not "la unu pomon". Mirrors
    # the renderer's `is_partial` flag (render.py:594-596).
    is_partial = False
    if count is not None:
        natural_vals = ent["properties"].get("count")
        if natural_vals:
            try:
                natural = int(natural_vals[0])
                if count < natural:
                    is_partial = True
            except (TypeError, ValueError):
                pass
    pl = (count is not None and count > 1)
    if case == "acc":
        head = (f"{concept}jn" if pl
                else f"{concept}n" if concept.endswith("o") else concept)
        adj_part = _inflect_adj_acc(adj, plural=pl) if adj else None
    else:
        head = (f"{concept}j" if pl else concept)
        adj_part = _inflect_adj_nom(adj, plural=pl) if adj else None
    pieces: list[str] = []
    if force_definite and not is_partial:
        pieces.append("la")
    if count is not None and count >= 1:
        pieces.append(_int_to_eo(count))
    if adj_part:
        pieces.append(adj_part)
    pieces.append(head)
    spec = _disclosed_specifier(eid, sentence_facts, entities)
    if spec is not None:
        prep, anchor_phrase = spec
        pieces.append(prep)
        pieces.append(anchor_phrase)
    return " ".join(pieces)


def _inflect_adj_nom(adj: str, *, plural: bool) -> str:
    if not adj:
        return adj
    base = adj if adj.endswith("a") else adj + "a"
    return base + "j" if plural else base


def _inflect_adj_acc(adj: str, *, plural: bool) -> str:
    nom = _inflect_adj_nom(adj, plural=plural)
    return nom + "n"


_INT_TO_EO_CACHE: dict[int, str] = {}


def _math_phrase(
    addends: list[tuple[int, str]],
    op_kind: str,
    result_count: int,
    result_noun: str,
    rng,
) -> str:
    """Render a chain-of-thought arithmetic phrase, picking word vs
    symbol op style ONCE so the answer stays consistent. Used by
    the subtraction and tallying generators so both share the same
    shape.

    `addends`: list of (count, noun) tuples. For subtraction, pass
      [(N, noun), (M, noun)] — the noun usually repeats. For tallying,
      pass [(N, noun_a), (M, noun_b)] — distinct nouns sharing a
      category.
    `op_kind`: 'add' or 'sub'.
    `result_count`, `result_noun`: the K and category/noun string.

    Example outputs:
      "tri pomoj plus du oranĝoj egalas kvin fruktoj."
      "tri pomoj + du oranĝoj = kvin fruktoj."
      "sep ovoj minus du ovoj egalas kvin ovoj."
      "sep ovoj - du ovoj = kvin ovoj."
    """
    if op_kind == "add":
        op_word, op_sym = "plus", "+"
    else:
        op_word, op_sym = "minus", "-"
    op, equals = rng.choice([(op_word, "egalas"), (op_sym, "=")])
    addend_phrases = [f"{_int_to_eo(c)} {n}" for c, n in addends]
    chain = f" {op} ".join(addend_phrases)
    result = f"{_int_to_eo(result_count)} {result_noun}"
    return f"{chain} {equals} {result}."


def _int_to_eo(n: int) -> str:
    """Local cardinal-renderer matching the realizer's int_to_esperanto
    output ("unu", "du", "tri", ...). Cached for the small int range
    we ever ask about."""
    cached = _INT_TO_EO_CACHE.get(n)
    if cached is not None:
        return cached
    from esperanto_lm.ontology.realize.render import int_to_esperanto
    out = int_to_esperanto(n)
    _INT_TO_EO_CACHE[n] = out
    return out


def _theme_phrase(ev: dict, entities: dict, case: str = "acc") -> str | None:
    """Render the theme of an event as it should appear in Q/A
    answers — joining all `theme` + `additional_theme_*` siblings
    with commas + "kaj". `case` is "acc" (accusative, default) or
    "nom". Returns None when no theme is resolvable.

    Example: ev with theme=pano, additional_theme_1=fromaĝo →
    "la panon kaj la fromaĝon" (case=acc) or "la pano kaj la fromaĝo"
    (case=nom). Single-theme events return "la panon" /
    "la pano". Mirrors the renderer's joined-list surface form."""
    eids = _theme_eids(ev)
    parts: list[str] = []
    for eid in eids:
        ent = entities.get(eid)
        if ent is None:
            continue
        c = ent["concept"]
        parts.append("la " + (_noun_acc(c) if case == "acc" else c))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} kaj {parts[1]}"
    return ", ".join(parts[:-1]) + f" kaj {parts[-1]}"


def _theme_eids(ev: dict) -> list[str]:
    """Return all theme-class entity ids for an event: the primary
    `theme` plus any `additional_theme_*` siblings (vidi binds
    additional_theme_1 / _2 under semantic_role=theme — the renderer
    joins them as "la X-on, la Y-on kaj la Z-on" in prose). Returns
    eid strings only; list themes (fari.parts) and unbound roles
    are filtered out."""
    out: list[str] = []
    theme = ev["roles"].get("theme")
    if isinstance(theme, str):
        out.append(theme)
    for k, v in ev["roles"].items():
        if k.startswith("additional_theme") and isinstance(v, str):
            out.append(v)
    return out


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

    For `koloro` we emit MULTIPLE phrasings per entity so the model
    sees the attribute-to-entity binding in several surface forms.
    The salience filter already prunes which entities show up in
    prose, so questions only fire for entities the model actually
    saw."""
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
            concept = ent["concept"]
            if slot == "koloro":
                q = f"Kia estis la koloro de la {concept}?"
                a = val
            elif slot == "posture":
                q = f"En kia pozicio estis la {concept}?"
                a = val
            elif slot == "openness":
                q = f"Ĉu la {concept} estis malfermita aŭ fermita?"
                a = val
            elif slot == "fullness":
                q = f"Ĉu la {concept} estis plena aŭ malplena?"
                a = val
            elif slot == "lock_state":
                q = f"Ĉu la {concept} estis ŝlosita aŭ malŝlosita?"
                a = val
            elif slot == "power_state":
                q = f"Ĉu la {concept} estis aktiva aŭ neaktiva?"
                a = val
            elif slot == "cleanliness":
                q = f"Ĉu la {concept} estis pura aŭ malpura?"
                a = val
            else:
                continue
            out.append({
                "q": q, "a": a,
                "requires": [{
                    "kind": "state", "entity": ent["eid"],
                    "slot": slot, "value": val,
                }],
            })

    # Reverse direction: when a (slot, value) is held by exactly one
    # in-trace entity, emit "what had value V?" → entity. Lets the
    # model answer 'Kio estis bruna?' → 'tablo' after seeing the
    # forward 'Kia koloro de tablo?' → 'bruna'. Skip when multiple
    # entities share the value — the answer would be ambiguous.
    by_slot_val: dict[tuple[str, str], list[str]] = {}
    for ent in rec["entities"]:
        if "_" in ent["eid"] and ent["eid"] != ent["concept"]:
            continue
        if ent["type"] in ("location", "abstract"):
            continue
        for slot in interesting_slots:
            vals = ent["properties"].get(slot)
            if not vals:
                continue
            val = vals[0]
            if val == UNMARKED.get(slot):
                continue
            by_slot_val.setdefault((slot, val), []).append(ent["eid"])
    for (slot, val), eids in by_slot_val.items():
        if len(eids) != 1:
            continue  # ambiguous reverse
        eid = eids[0]
        ent = entities[eid]
        concept = ent["concept"]
        if slot == "koloro":
            q = f"Kio estis {val}?"
        elif slot == "openness":
            # `val` is malfermita/fermita — adjectival form already
            q = f"Kio estis {val}?"
        elif slot == "fullness":
            q = f"Kio estis {val}?"
        elif slot == "lock_state":
            q = f"Kio estis {val}?"
        elif slot == "power_state":
            q = f"Kio estis {val}?"
        elif slot == "cleanliness":
            q = f"Kio estis {val}?"
        elif slot == "posture":
            q = f"Kio estis en pozicio {val}?"
        else:
            continue
        out.append({
            "q": q, "a": f"la {concept}.",
            "requires": [{
                "kind": "state", "entity": eid,
                "slot": slot, "value": val,
            }],
        })
    return out


def _q_first_last(rec: dict, rng: random.Random) -> list[dict]:
    """First and last verb in the event sequence."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []

    def describe(ev):
        subj = _subject_phrase(ev, entities)
        if subj is None:
            return f"{ev['action']}"
        verb_past = _past(ev["action"])
        theme = ev["roles"].get("theme")
        if theme is None:
            return f"{subj} {verb_past}"
        # theme may be a list (fari.parts) — collapse
        if isinstance(theme, list):
            theme_name = ", ".join(_name(t, entities) for t in theme)
            return f"{subj} {verb_past} {theme_name}"
        theme_phrase = _theme_phrase(ev, entities, case="nom")
        if theme_phrase is None:
            theme_phrase = _name(theme, entities)
        return f"{subj} {verb_past} {theme_phrase}"

    first = next((e for e in events if not _should_skip_verb(e["action"])), None)
    last = next((e for e in reversed(events) if not _should_skip_verb(e["action"])), None)
    if first is None:
        return []
    out.append({
        "q": "Kio okazis unue en la rakonto?",
        "a": describe(first) + ".",
        "requires": [{"kind": "event", "event_id": first["id"]}],
    })
    if last is not None and last is not first:
        out.append({
            "q": "Kio okazis laste en la rakonto?",
            "a": describe(last) + ".",
            "requires": [{"kind": "event", "event_id": last["id"]}],
        })
    return out


def _q_action_attribution(rec: dict, rng: random.Random) -> list[dict]:
    """Per content event, ask "who did X to Y" and / or
    "what did Z do"."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    sentence_facts = rec.get("sentence_facts") or []
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
        # Collect additional themes (semantic_role="theme" siblings:
        # vidi.additional_theme_1, _2). The renderer joins them as
        # "la X-on, la Y-on kaj la Z-on" in prose; Q/A asks
        # "Kion vidis X?" should answer with the same joint list.
        addl_themes = []
        for rk, rv in ev["roles"].items():
            if rk == "theme" or not rk.startswith("additional_theme"):
                continue
            rv_ent = entities.get(rv) if isinstance(rv, str) else None
            if rv_ent is not None:
                addl_themes.append(rv)
        all_theme_eids = [theme] + addl_themes
        agent_name = _name(agent, entities)
        ev_pattern = {"kind": "event", "event_id": ev["id"]}
        companion = ev["roles"].get("companion")
        companion_ent = entities.get(companion) if companion else None
        if companion_ent is not None:
            companion_name = _name(companion, entities)
            subj_phrase = f"{agent_name} kaj {companion_name}"
        else:
            subj_phrase = agent_name
        verb_past = _past(ev["action"])
        # Per-theme attribution: "Kiu V-is la theme_i?" → SUBJ
        # Use specifier phrase ("la du grandajn glasojn") so the Q
        # matches what the prose actually said. Bare noun fallback
        # for legacy compatibility when no count/adj was disclosed.
        for th_eid in all_theme_eids:
            th_ent = entities.get(th_eid)
            if th_ent is None:
                continue
            th_phrase = _specifier_noun_phrase(
                th_eid, entities, sentence_facts, case="acc")
            if th_phrase is None:
                th_phrase = "la " + _noun_acc(th_ent["concept"])
            out.append({
                "q": f"Kiu {verb_past} {th_phrase}?",
                "a": rng.choice([
                    f"{subj_phrase}.",
                    f"{subj_phrase} {verb_past} {th_phrase}.",
                    f"Estis {subj_phrase}.",
                ]),
                "requires": [ev_pattern],
            })
        # "What did Z X-i?" — joint listing of all themes if multiple
        if agent_ent["type"] == "person":
            theme_concepts = []
            for e in all_theme_eids:
                if e not in entities:
                    continue
                p = _specifier_noun_phrase(
                    e, entities, sentence_facts, case="acc")
                if p is None:
                    p = "la " + _noun_acc(entities[e]["concept"])
                theme_concepts.append(p)
            if len(theme_concepts) == 1:
                themes_phrase = theme_concepts[0]
            elif len(theme_concepts) == 2:
                themes_phrase = f"{theme_concepts[0]} kaj {theme_concepts[1]}"
            else:
                themes_phrase = (", ".join(theme_concepts[:-1])
                                 + f" kaj {theme_concepts[-1]}")
            out.append({
                "q": f"Kion {agent_name} {verb_past}?",
                "a": rng.choice([
                    f"{themes_phrase}.",
                    f"{agent_name} {verb_past} {themes_phrase}.",
                ]),
                "requires": [ev_pattern],
            })
        # "What did Z do to the Y?" — verb extraction (primary theme)
        if agent_ent["type"] == "person":
            shapes = [
                f"Kion {agent_name} faris al la {theme_ent['concept']}?",
                f"Kion {agent_name} faris kun la {theme_ent['concept']}?",
            ]
            out.append({
                "q": shapes[rng.randrange(len(shapes))],
                "a": rng.choice([
                    f"{verb_past} ĝin.",
                    f"{agent_name} {verb_past} ĝin.",
                    f"{verb_past} la {theme_ent['concept']}n.",
                ]),
                "requires": [ev_pattern],
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
            out.append({
                "q": q, "a": str(new_val),
                "requires": [{"kind": "event", "event_id": ev["id"]}],
            })
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
    # Derive the spatial relation set from the lexicon (containment
    # relations declared in containment.jsonl) plus `apud` for peer
    # adjacency. Build Q/A shapes dynamically: every supported
    # relation gets both the bare "Kie?" form and a relation-specific
    # "Prep kio?" form. Capitalize the relation name for sentence-
    # start use; the Esperanto preposition equals the relation name.
    from esperanto_lm.ontology.containment import (
        containment_relation_names, resolve_containment)
    lex = _get_lex()
    containment_rels = containment_relation_names(
        resolve_containment(lex), lex)
    # `apud` is a peer (non-containment) but still a spatial location;
    # it's also `specifier_of`-eligible in the schema.
    spatial_rels = sorted(containment_rels | {"apud"})
    prep_for_rel = {r: r.capitalize() for r in spatial_rels}
    q_shapes = {}
    for r in spatial_rels:
        Cap = r.capitalize()
        q_shapes[r] = [
            (f"Kie estis la {{x}} komence?", f"{Cap} la {{y}}."),
            (f"{Cap} kio estis la {{x}}?", f"{Cap} la {{y}}."),
            # Full-sentence echo: "La X estis prep la Y." Matches
            # the natural prose form and gives the model a complete-
            # answer template the eval sometimes expects.
            (f"Kie estis la {{x}} komence?",
             f"La {{x}} estis {r} la {{y}}."),
        ]
    # `apud` admits the synonymous "Ĉe" form for question/answer.
    if "apud" in q_shapes:
        q_shapes["apud"].extend([
            ("Ĉe kio estis la {x}?", "Ĉe la {y}."),
            ("Ĉe kio estis la {x}?", "La {x} estis ĉe la {y}."),
        ])
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
            "requires": [{
                "kind": "relation", "rel": r["relation"],
                "args[0]": contained, "args[1]": container,
            }],
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

        ev_pattern = {"kind": "event", "event_id": ev["id"]}
        if instr_name and part_names:
            # Both tool and materials
            out.append({
                "q": (f"Per kio kaj el kio {agent_name} "
                      f"{verb}{theme_phrase}?"),
                "a": f"Per {instr_name}, el {parts_phrase}.",
                "requires": [ev_pattern],
            })
            # Also split into individual questions
            out.append({
                "q": f"Per kio {agent_name} {verb}{theme_phrase}?",
                "a": f"per {instr_name}",
                "requires": [ev_pattern],
            })
            out.append({
                "q": f"El kio {agent_name} {verb}{theme_phrase}?",
                "a": f"el {parts_phrase}",
                "requires": [ev_pattern],
            })
        elif instr_name:
            # Tool only
            out.append({
                "q": f"Per kio {agent_name} {verb}{theme_phrase}?",
                "a": f"per {instr_name}",
                "requires": [ev_pattern],
            })
        elif part_names:
            # Materials only (e.g. sandviĉo without crafted_with)
            out.append({
                "q": f"El kio {agent_name} {verb}{theme_phrase}?",
                "a": f"el {parts_phrase}",
                "requires": [ev_pattern],
            })
    return out


def _q_count(rec: dict, rng: random.Random) -> list[dict]:
    """For entities with count > 1, ask how many — plain, in a
    location, or owned by someone.

    When multiple stacks of the same concept exist in the scene
    (pomo_1 count=3 + pomo_2 count=2), aggregate and emit a single
    CoT answer showing the addition (3 + 2 = 5). Single-stack cases
    stay bare retrieval — there's nothing to compute."""
    entities = {e["eid"]: e for e in rec["entities"]}
    setup_rels = rec.get("setup_relations", [])
    loc_of = {}
    owner_of = {}
    for rel in setup_rels:
        if rel["relation"] in ("en", "sur", "sub"):
            loc_of.setdefault(rel["args"][0],
                              (rel["relation"], rel["args"][1]))
        elif rel["relation"] == "havi":
            owner_of.setdefault(rel["args"][1], rel["args"][0])
    # Group countable entities by concept. Each bucket holds
    # (eid, n) tuples for every stack of that concept — including
    # singletons (count=1), because they contribute to the total
    # when multi-stack splits land "5 → 3+1+1" and we still want
    # the aggregated answer to be 5.
    by_concept: dict[str, list[tuple[str, int]]] = {}
    for ent in rec["entities"]:
        # Skip body-parts (`petro_mano`) but allow split-stack siblings
        # tagged with `sibling_of` by the spawner.
        is_part = (
            "_" in ent["eid"]
            and ent["eid"] != ent["concept"]
            and ent.get("sibling_of") is None)
        if is_part:
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
        if n < 1 or n >= len(CARDINALS_EO):
            continue
        by_concept.setdefault(ent["concept"], []).append((ent["eid"], n))
    out = []
    for concept, stacks in by_concept.items():
        total = sum(n for _e, n in stacks)
        # Skip total=1 to avoid "Kiom da X? Unu." mode collapse — the
        # bare-singleton case isn't quiz-worthy. Eval never asks
        # count=1 questions.
        if total < 2 or total >= len(CARDINALS_EO):
            continue
        # Per-stack count facts must all be disclosed in prose for the
        # sum to be derivable.
        count_pats = [
            {"kind": "count", "entity": eid, "value": n}
            for eid, n in stacks
        ]
        if len(stacks) >= 2:
            # Multi-stack: CoT showing the addition. All shapes walk
            # the sum explicitly.
            addends_str = " plus ".join(
                f"{_int_to_eo(n)} {concept}j" if n > 1 else f"unu {concept}"
                for _e, n in stacks)
            tot_eo = _int_to_eo(total)
            cot_shapes = [
                f"{addends_str} egalas {tot_eo} {concept}j.",
                f"{addends_str} = {tot_eo} {concept}j.",
                f"Sume {tot_eo} {concept}j ({addends_str}).",
            ]
            out.append({
                "q": f"Kiom da {concept}j estis?",
                "a": rng.choice(cot_shapes),
                "requires": count_pats,
            })
        else:
            # Single-stack: bare retrieval is honest — nothing to
            # compute, just read off the prose.
            eid, n = stacks[0]
            out.append({
                "q": f"Kiom da {concept}j estis?",
                "a": _count_answer(n, concept, rng, "estis"),
                "requires": count_pats,
            })
            loc_info = loc_of.get(eid)
            if loc_info:
                prep, loc_eid = loc_info
                if loc_eid in entities:
                    loc_name = entities[loc_eid]["concept"]
                    out.append({
                        "q": f"Kiom da {concept}j estis {prep} la {loc_name}?",
                        "a": _count_answer(n, concept, rng, "estis"),
                        "requires": [
                            count_pats[0],
                            {"kind": "relation", "rel": prep,
                             "args[0]": eid, "args[1]": loc_eid},
                        ],
                    })
            owner_eid = owner_of.get(eid)
            if owner_eid and owner_eid in entities:
                owner_name = _entity_name(owner_eid, entities)
                out.append({
                    "q": f"Kiom da {concept}j havis {owner_name}?",
                    "a": _count_answer(n, concept, rng, "estis"),
                    "requires": [
                        count_pats[0],
                        {"kind": "relation", "rel": "havi",
                         "args[0]": owner_eid, "args[1]": eid},
                    ],
                })
    return out


def _count_answer(n: int, concept: str, rng: random.Random,
                  verb: str = "estis") -> str:
    """Vary count answer format: bare number, with noun, or full sentence."""
    num = CARDINALS_EO[n]
    pl = f"{concept}j" if n > 1 else concept
    choice = rng.randint(0, 2)
    if choice == 0:
        return num.capitalize()
    elif choice == 1:
        return f"{num.capitalize()} {pl}."
    else:
        return f"{verb.capitalize()} {num} {pl}."


def _q_event_theme_count(rec: dict, rng: random.Random) -> list[dict]:
    """Event-bound count: "Kiom da X-oj agent V-is?" → "N X-oj."
    Composes the event role-binding with the entity's count fact —
    teaches the model to join "Petro vidis du glasojn..." (from
    prose) with the count answer. Iterates every theme-class entity
    bound to the event (theme + additional_theme_*); each themed
    entity with count > 1 gets a Q/A bound to that event. Multi-theme
    events (vidi.theme + .additional_theme_*) thus get one Q/A per
    counted theme."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        agent = ev["roles"].get("agent")
        if agent is None or agent not in entities:
            continue
        agent_name = _name(agent, entities)
        verb_past = _past(ev["action"])
        for th_eid in _theme_eids(ev):
            th_ent = entities.get(th_eid)
            if th_ent is None:
                continue
            if "_" in th_eid and th_eid != th_ent["concept"]:
                continue
            count_vals = th_ent["properties"].get("count")
            if not count_vals:
                continue
            try:
                n = int(count_vals[0])
            except (ValueError, TypeError):
                continue
            if n < 2 or n >= len(CARDINALS_EO):
                continue
            concept = th_ent["concept"]
            num = CARDINALS_EO[n]
            pl_acc = f"{concept}jn"
            q = f"Kiom da {concept}j {agent_name} {verb_past}?"
            a = rng.choice([
                f"{num.capitalize()}.",
                f"{num.capitalize()} {pl_acc}.",
                f"{agent_name} {verb_past} {num} {pl_acc}.",
            ])
            out.append({
                "q": q, "a": a,
                "requires": [
                    {"kind": "event", "event_id": ev["id"]},
                    {"kind": "count", "entity": th_eid, "value": n},
                ],
            })
    return out


def _q_count_delta(rec: dict, rng: random.Random) -> list[dict]:
    """Arithmetic Q/A from count changes: 'there were N, someone
    ate/gave/drank M, how many remain?'"""
    entities = {e["eid"]: e for e in rec["entities"]}
    events = rec.get("events", [])
    out = []
    count_deltas: dict[str, list[tuple[str, str, int, int]]] = {}
    for ev in events:
        for k, v in ev.get("property_changes", {}).items():
            if "|count" not in k:
                continue
            eid = k.split("|")[0]
            try:
                new_val = int(v)
            except (ValueError, TypeError):
                continue
            count_deltas.setdefault(eid, []).append(
                (ev["action"], eid, new_val))
    for eid, changes in count_deltas.items():
        ent = entities.get(eid)
        if ent is None:
            continue
        concept = ent["concept"]
        count_vals = ent["properties"].get("count")
        if not count_vals:
            continue
        try:
            initial = int(count_vals[0])
        except (ValueError, TypeError):
            continue
        if initial <= 1:
            continue
        final = changes[-1][2]
        delta = initial - final
        if delta <= 0 or delta >= initial:
            continue
        if initial >= len(CARDINALS_EO) or final >= len(CARDINALS_EO):
            continue
        if delta >= len(CARDINALS_EO):
            continue
        lex = _get_lex()
        agent_eid = None
        verb = None
        # `restas` (remains) only fits consumption — verbs where the
        # theme disappears from the world, not just changes owner.
        # `lex.state_verbs[("presence", "manĝita")]` is the lexicon-
        # derived set: manĝi/bruli/trinki etc. doni/vendi/aĉeti only
        # set count=0 (transfer) and are absent here, so their
        # "Kiom da X restas?" misframing is naturally avoided.
        lex = _get_lex()
        consumption_verbs = lex.state_verbs.get(
            ("presence", "manĝita"), ())
        for ev in events:
            if ev["action"] in consumption_verbs:
                theme = ev["roles"].get("theme")
                if theme == eid:
                    agent_eid = ev["roles"].get("agent")
                    verb = ev["action"]
                    break
        if agent_eid is None or verb is None:
            continue
        agent_ent = entities.get(agent_eid)
        if agent_ent and agent_ent.get("type") == "person":
            agent_name = _name(agent_eid, entities)
        else:
            agent_name = "la " + (
                agent_ent["concept"] if agent_ent else agent_eid)
        verb_past = {
            "manĝi": "manĝis", "trinki": "trinkis",
            "doni": "donis", "vendi": "vendis",
            "aĉeti": "aĉetis",
        }.get(verb, verb)
        ev_id = None
        for ev in events:
            if (ev["action"] == verb
                    and ev["roles"].get("theme") == eid
                    and ev["roles"].get("agent") == agent_eid):
                ev_id = ev["id"]
                break
        # Arithmetic: needs the initial count disclosed (for the model
        # to subtract from) AND the change event narrated.
        requires = [{"kind": "count", "entity": eid, "value": initial}]
        if ev_id is not None:
            requires.append({"kind": "event", "event_id": ev_id})
        # Chain-of-thought arithmetic answer: show initial - delta =
        # final so the model learns the subtraction operation, not
        # just the answer. Three shape variants:
        #   "N minus M = K."
        #   "Komence estis N. V-is M. Restas K."
        #   "K X-oj" (bare answer — kept for variation)
        n_eo = _int_to_eo(initial)
        m_eo = _int_to_eo(delta)
        k_eo = _int_to_eo(final)
        pl = f"{concept}j" if final > 1 else concept
        pl_acc = f"{concept}jn" if delta > 1 else f"{concept}n"
        pl_full = f"{concept}j"
        sing = concept
        # All shapes walk the subtraction explicitly; only the surface
        # form varies (equation / narrated / reasoned). Bare-answer
        # forms were removed — they don't expose the operation and
        # let the model bypass arithmetic.
        cot_shapes = [
            # Equation: "N X minus M X egalas K X."
            _math_phrase(
                [(initial, pl_full if initial > 1 else sing),
                 (delta, pl_full if delta > 1 else sing)],
                "sub", final, pl, rng),
            # Narrated trace: setup → action → result.
            (f"Komence estis {n_eo} {pl_full}. {agent_name} "
             f"{verb_past} {m_eo} {pl_acc}. Restas {k_eo} {pl}."),
            # Reasoned with explicit subtraction.
            (f"Estis {n_eo} {pl_full}. {agent_name} {verb_past} "
             f"{m_eo} {pl_acc}, do restas {n_eo} minus {m_eo}, "
             f"egalas {k_eo} {pl}."),
            # Equation + narrative gloss.
            (f"{n_eo} minus {m_eo} egalas {k_eo}. {agent_name} "
             f"{verb_past} {m_eo} {pl_acc}, restas {k_eo} {pl}."),
        ]
        out.append({
            "q": (f"Kiom da {concept}j restas post kiam"
                  f" {agent_name} {verb_past}?"),
            "a": rng.choice(cot_shapes),
            "requires": requires,
        })
    return out


def _entity_name(eid, entities):
    """Like _name but prefixes non-persons with 'la' for use in
    sentence positions where the article is needed."""
    ent = entities.get(eid)
    if not ent:
        return eid
    if ent.get("type") == "person":
        return _name(eid, entities)
    return "la " + ent["concept"]


def _trace_state_at(position, entities, events, setup_relations):
    """Snapshot of (havi, counts) after the first `position` events
    fire. Replays setup + each event's property_changes (for counts)
    and relation_changes (for havi). Mirrors what the engine does
    but only tracks the slices the donor/recipient Q-types need.

    `havi`: dict[eid → set of owned entity ids] — havi(owner, item)
            edges that hold at this position.
    `counts`: dict[eid → int] — current count of each entity.

    Entities created mid-trace (TransferN partial-split sub-stacks)
    appear only after their `created_at_event` index has passed."""
    havi: dict = {}
    for r in setup_relations or ():
        if not isinstance(r, dict):
            continue
        if r.get("relation") != "havi":
            continue
        args = r.get("args", [])
        if len(args) == 2:
            havi.setdefault(args[0], set()).add(args[1])
    counts: dict = {}
    for eid, ent in entities.items():
        created = ent.get("created_at_event")
        if created is not None and created >= position:
            continue
        c = ent.get("properties", {}).get("count")
        if c:
            try:
                counts[eid] = int(c[0])
            except (ValueError, TypeError):
                pass
    for i, ev in enumerate(events):
        if i >= position:
            break
        for k, v in ev.get("property_changes", {}).items():
            if "|count" not in k:
                continue
            eid_k = k.split("|", 1)[0]
            try:
                counts[eid_k] = int(v)
            except (ValueError, TypeError):
                pass
        for r in ev.get("relation_changes", ()):
            if not isinstance(r, dict):
                continue
            rel = r.get("rel") or r.get("relation")
            if rel != "havi":
                continue
            args = r.get("args", [])
            if len(args) != 2:
                continue
            owner, item = args
            if r.get("added"):
                havi.setdefault(owner, set()).add(item)
            else:
                havi.get(owner, set()).discard(item)
    return havi, counts


def _count_owned_at(eid, concept, position, entities, events, setup_relations):
    """Sum of counts across all `concept`-typed entities `eid` currently
    `havi`s, after `position` events have fired. The same aggregate
    semantic the planner uses (`agent.planner._count_owned`), but
    walking the trace from setup forward to the requested position."""
    havi, counts = _trace_state_at(
        position, entities, events, setup_relations)
    total = 0
    for owned_eid in havi.get(eid, ()):
        owned_ent = entities.get(owned_eid)
        if owned_ent is None:
            continue
        if owned_ent.get("concept") != concept:
            continue
        total += counts.get(owned_eid, 1)
    return total


def _q_count_transfer(rec: dict, rng: random.Random) -> list[dict]:
    """Transfer Q/A: after a verb that establishes havi(non-agent,
    theme), ask who has how many. The arithmetic is grounded in the
    donor's and recipient's ACTUAL `count_owned(concept)` snapshots
    before and after the event — NOT in the source stash's initial
    count, which in multi-step chains diverges from either party's
    holding (see notes at the donor-Q gate below)."""
    from esperanto_lm.ontology.agent.forward_planner import (
        _build_rule_effects_index, _RULE_EFFECTS_CACHE,
    )
    lex = _get_lex()
    from esperanto_lm.ontology.dsl.rules import DEFAULT_DSL_RULES
    re = _RULE_EFFECTS_CACHE.get(id(lex))
    if re is None:
        re = _build_rule_effects_index(DEFAULT_DSL_RULES, lex)
        _RULE_EFFECTS_CACHE[id(lex)] = re
    transfer_verbs = {}
    for verb, entry in re.items():
        for rule in entry.get("rules", []):
            for rel, role_names in rule.get("adds", []):
                if rel == "havi" and role_names[0] != "agent":
                    transfer_verbs[verb] = role_names[0]
    entities = {e["eid"]: e for e in rec["entities"]}
    events = rec.get("events", [])
    setup_rels = rec.get("setup_relations") or []
    out = []
    for i, ev in enumerate(events):
        if ev["action"] not in transfer_verbs:
            continue
        recip_role = transfer_verbs[ev["action"]]
        agent = ev["roles"].get("agent")
        recip = ev["roles"].get(recip_role)
        theme = ev["roles"].get("theme")
        if not all((agent, recip, theme)):
            continue
        ent = entities.get(theme)
        if ent is None:
            continue
        concept = ent["concept"]
        donor_before = _count_owned_at(
            agent, concept, i, entities, events, setup_rels)
        donor_after = _count_owned_at(
            agent, concept, i + 1, entities, events, setup_rels)
        recip_before = _count_owned_at(
            recip, concept, i, entities, events, setup_rels)
        recip_after = _count_owned_at(
            recip, concept, i + 1, entities, events, setup_rels)
        donor_delta = donor_before - donor_after
        recip_delta = recip_after - recip_before
        donor = _entity_name(agent, entities)
        recipient = _entity_name(recip, entities)
        verb_past = ev["action"].replace("i", "is", 1)
        req = [{"kind": "event", "event_id": ev["id"]}]

        # Recipient side: only emit when recipient's count_owned of
        # `concept` genuinely went up across the event. Covers
        # straight transfers (0 → N) and chained chains where
        # recipient already held something (M → M+N).
        if (recip_delta > 0
                and recip_before < len(CARDINALS_EO)
                and recip_after < len(CARDINALS_EO)
                and recip_delta < len(CARDINALS_EO)):
            recip_pl = f"{concept}j" if recip_after != 1 else concept
            n_before = _int_to_eo(recip_before)
            m_delta = _int_to_eo(recip_delta)
            k_after = _int_to_eo(recip_after)
            if recip_before == 0:
                ans = (f"Nulo plus {m_delta} egalas {m_delta} "
                       f"{recip_pl}. {recipient} havas {m_delta} "
                       f"{recip_pl}.")
            else:
                ans = (f"{n_before} plus {m_delta} egalas {k_after} "
                       f"{recip_pl}. {recipient} havas {k_after} "
                       f"{recip_pl}.")
            out.append({
                "q": (f"Kiom da {concept}j havas {recipient}"
                      f" post kiam {donor} {verb_past}?"),
                "a": ans,
                "requires": req,
            })

        # Donor side: only emit when donor's count_owned of `concept`
        # genuinely DROPPED across the event. In chains where the
        # agent didn't really hold any of `concept` at scene start
        # but is the agent of a transfer from a floating stash, the
        # donor-side narrative "donor had X, gave M, has X-M" maps
        # to no real entity. This delta-based gate filters those
        # cases without per-verb logic: if the agent's actual
        # holding doesn't change, the Q is meaningless and skipped.
        if (donor_delta > 0
                and donor_before < len(CARDINALS_EO)
                and donor_after < len(CARDINALS_EO)
                and donor_delta < len(CARDINALS_EO)):
            donor_pl = f"{concept}j" if donor_after != 1 else concept
            pl_full = f"{concept}j"
            sing = concept
            initial_acc = (
                f"{concept}jn" if donor_before > 1 else f"{concept}n")
            pl_acc = (
                f"{concept}jn" if donor_delta > 1 else f"{concept}n")
            n_eo = _int_to_eo(donor_before)
            m_eo = _int_to_eo(donor_delta)
            k_eo = _int_to_eo(donor_after)
            cot_donor = [
                _math_phrase(
                    [(donor_before, pl_full if donor_before > 1 else sing),
                     (donor_delta, pl_full if donor_delta > 1 else sing)],
                    "sub", donor_after, donor_pl, rng),
                (f"Komence {donor} havis {n_eo} {initial_acc}. "
                 f"Donis {m_eo} {pl_acc}. Restas {k_eo} {donor_pl}."),
                (f"{donor} havis {n_eo} {initial_acc}, donis {m_eo} "
                 f"{pl_acc}, do {n_eo} minus {m_eo} egalas {k_eo} "
                 f"{donor_pl}."),
                (f"{n_eo} minus {m_eo} egalas {k_eo}. {donor} restas "
                 f"kun {k_eo} {donor_pl}."),
            ]
            out.append({
                "q": (f"Kiom da {concept}j ankoraŭ havas {donor}"
                      f" post kiam {donor} {verb_past}"
                      f" al {recipient}?"),
                "a": rng.choice(cot_donor),
                "requires": req,
            })
    return out


def _q_count_sum(rec: dict, rng: random.Random) -> list[dict]:
    """Cross-entity tallying: when two or more entities in the same
    scene share a category (frukto, meblo, gefrato, etc.), emit a
    sum question with a CoT answer showing the addition. Eval has
    questions like "Kiom da gefratoj?" given "kvar fratojn kaj du
    fratinojn" — needs A+B=K, not subtraction.

    Only fires when each entity's count is disclosed in
    sentence_facts (so the model can recover N and M from prose)."""
    global _CONCEPT_CATEGORIES
    if not _CONCEPT_CATEGORIES:
        _CONCEPT_CATEGORIES = _load_concept_categories()
    entities = {e["eid"]: e for e in rec["entities"]}
    sentence_facts = rec.get("sentence_facts") or []
    # Map each entity to (concept, disclosed_count) when count is
    # disclosed in prose. Skip body parts / locations / abstracts.
    by_category: dict[str, list[tuple[str, str, int]]] = {}
    for ent in rec["entities"]:
        if ent["eid"] == "mondo" or ent["type"] in ("location", "abstract"):
            continue
        if "_" in ent["eid"] and ent["eid"] != ent["concept"]:
            continue
        count = _disclosed_count(ent["eid"], sentence_facts)
        if count is None or count < 1:
            count = 1  # implicit singleton — count=1 disclosure rare
        cats = _CONCEPT_CATEGORIES.get(ent["concept"], [])
        if not cats:
            continue
        # Use the most-specific (first) category to keep aliases coherent
        cat = cats[0]
        if cat == ent["concept"]:
            continue
        by_category.setdefault(cat, []).append(
            (ent["eid"], ent["concept"], count))
    out = []
    for cat, members in by_category.items():
        if len(members) < 2:
            continue
        total = sum(c for _e, _c, c in members)
        if total >= len(CARDINALS_EO):
            continue
        # Skip when any addend is too large for the cardinal lookup
        if any(c >= len(CARDINALS_EO) for _e, _c, c in members):
            continue
        addends = [
            (c, f"{concept}j" if c > 1 else concept)
            for _eid, concept, c in members
        ]
        cat_pl = f"{cat}j"
        n_eo = _int_to_eo(total)
        # All-CoT: equation / narrated tally / step-through. The
        # bare cardinal answer was removed so every emit shows the
        # addition explicitly.
        addends_str = " kaj ".join(
            f"{_int_to_eo(c)} {n}" for c, n in addends)
        a_shapes = [
            # Equation: "N X plus M Y egalas K cat."
            _math_phrase(addends, "add", total, cat_pl, rng),
            # Narrated tally with category aggregation.
            f"Estas {addends_str}. Sume {n_eo} {cat_pl}.",
            # Step + answer.
            (f"{addends_str} — tio sumas al {n_eo} {cat_pl}."),
        ]
        # Require each entity's count fact for grounding
        requires = [
            {"kind": "count", "entity": eid, "value": c}
            for eid, _c, c in members
        ]
        out.append({
            "q": f"Kiom da {cat_pl} estis?",
            "a": rng.choice(a_shapes),
            "requires": requires,
        })
    return out


def _q_count_chain_funcall(rec: dict, rng: random.Random) -> list[dict]:
    """Multi-event count chain as a funcall sequence.

    For each entity with ≥2 count-affecting events, emit a Q/A where
    the answer is the chain `[[N OP1 d1]] [[#1 OP2 d2]] ...`. Mirrors
    `_q_count_delta` (single-event case) but extends to longer chains
    and emits the funcall format instead of Eo CoT prose.

    The model learns to identify operands from the scene narrative AND
    the operator polarity (consumption/transfer → minus; receive/pick
    up → plus) from the trace events that the prose surfaces."""
    entities = {e["eid"]: e for e in rec["entities"]}
    events = rec.get("events", [])
    # Walk events in order; for each count change, attribute it to
    # the most recent USER-VISIBLE event (not the synthetic `_change`
    # that carries the property_change). The realizer's disclosure
    # log records the source verb's event_id (preni/vendi/doni etc.),
    # never the `_change` synthetic — without this attribution, the
    # chain's `requires=event(_change.id)` never matches any
    # disclosed fact and every chain Q gets filtered as undisclosed.
    by_eid: dict[str, list[tuple[str, str, int]]] = {}
    last_user_event: dict | None = None
    for ev in events:
        if ev.get("action") != "_change":
            last_user_event = ev
        for k, v in ev.get("property_changes", {}).items():
            if "|count" not in k:
                continue
            eid = k.split("|")[0]
            try:
                new_val = int(v)
            except (ValueError, TypeError):
                continue
            src_ev = (last_user_event
                      if ev.get("action") == "_change"
                      and last_user_event is not None
                      else ev)
            by_eid.setdefault(eid, []).append(
                (src_ev["id"], src_ev["action"], new_val))

    out = []
    for eid, changes in by_eid.items():
        if len(changes) < 2:
            continue
        ent = entities.get(eid)
        if ent is None:
            continue
        concept = ent["concept"]
        count_vals = ent["properties"].get("count")
        if not count_vals:
            continue
        try:
            initial = int(count_vals[0])
        except (ValueError, TypeError):
            continue
        if initial < 1:
            continue
        calls = []
        prev = initial
        requires: list = [{"kind": "count", "entity": eid, "value": initial}]
        ok = True
        for i, (ev_id, _action, new_val) in enumerate(changes):
            delta = new_val - prev
            if delta == 0:
                ok = False
                break
            op = "+" if delta > 0 else "-"
            mag = abs(delta)
            if i == 0:
                calls.append(f"[[{prev}{op}{mag}]]")
            else:
                calls.append(f"[[#{i}{op}{mag}]]")
            prev = new_val
            requires.append({"kind": "event", "event_id": ev_id})
        if not ok or len(calls) < 2:
            continue
        pl = f"{concept}j" if prev != 1 else concept
        q = f"Kiom da {pl} restas?"
        out.append({"q": q, "a": " ".join(calls), "requires": requires})
    return out


def _q_multi_lemma_sum_funcall(rec: dict, rng: random.Random) -> list[dict]:
    """Category-level sum across two or more entities, rendered as a
    funcall call.

    Mirrors `_q_count_sum` (which renders the answer as Eo prose like
    `tri pomoj kaj du oranĝoj egalas kvin fruktoj.`) but emits the
    answer as a single chain call `[[a+b+c]]`. This trains the model
    to surface aggregation in the structured-call format the executor
    can run."""
    global _CONCEPT_CATEGORIES
    if not _CONCEPT_CATEGORIES:
        _CONCEPT_CATEGORIES = _load_concept_categories()
    sentence_facts = rec.get("sentence_facts") or []
    by_category: dict[str, list[tuple[str, str, int]]] = {}
    for ent in rec["entities"]:
        if ent["eid"] == "mondo" or ent["type"] in ("location", "abstract"):
            continue
        if "_" in ent["eid"] and ent["eid"] != ent["concept"]:
            continue
        count = _disclosed_count(ent["eid"], sentence_facts)
        if count is None or count < 1:
            count = 1
        cats = _CONCEPT_CATEGORIES.get(ent["concept"], [])
        if not cats:
            continue
        cat = cats[0]
        if cat == ent["concept"]:
            continue
        by_category.setdefault(cat, []).append(
            (ent["eid"], ent["concept"], count))
    out = []
    for cat, members in by_category.items():
        if len(members) < 2:
            continue
        total = sum(c for _, _, c in members)
        if total < 2 or total >= 1000:
            continue
        counts = [str(c) for _, _, c in members]
        expr = "+".join(counts)
        requires = [
            {"kind": "count", "entity": eid, "value": c}
            for eid, _c, c in members
        ]
        out.append({
            "q": f"Kiom da {cat}j estis?",
            "a": f"[[{expr}]]",
            "requires": requires,
        })
    return out


def _q_count_before(rec: dict, rng: random.Random) -> list[dict]:
    """Temporal: 'how many X were there before Y happened?'
    Answer is the initial count from the entity properties."""
    entities = {e["eid"]: e for e in rec["entities"]}
    events = rec.get("events", [])
    out = []
    seen = set()
    for i, ev in enumerate(events):
        for k, v in ev.get("property_changes", {}).items():
            if "|count" not in k:
                continue
            eid = k.split("|")[0]
            if eid in seen:
                continue
            seen.add(eid)
            ent = entities.get(eid)
            if ent is None:
                continue
            count_vals = ent["properties"].get("count")
            if not count_vals:
                continue
            try:
                initial = int(count_vals[0])
            except (ValueError, TypeError):
                continue
            if initial <= 1 or initial >= len(CARDINALS_EO):
                continue
            concept = ent["concept"]
            agent_eid = ev["roles"].get("agent")
            verb = ev["action"]
            cause_ev = ev
            if not agent_eid or verb == "_change":
                for prev in reversed(events[:i]):
                    if prev["roles"].get("theme") == eid and prev["action"] != "_change":
                        agent_eid = prev["roles"].get("agent")
                        verb = prev["action"]
                        cause_ev = prev
                        break
            if not agent_eid:
                continue
            agent = _entity_name(agent_eid, entities)
            verb_past = verb.replace("i", "is", 1)
            out.append({
                "q": (f"Kiom da {concept}j estis antaŭ ol"
                      f" {agent} {verb_past}?"),
                "a": _count_answer(initial, concept, rng, "estis"),
                "requires": [
                    {"kind": "count", "entity": eid, "value": initial},
                    {"kind": "event", "event_id": cause_ev["id"]},
                ],
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
            "requires": [
                {"kind": "event", "event_id": ev["id"]},
                {"kind": "event", "event_id": cause_id},
            ],
        })
    return out


_WHY_PROP_SKIP_SLOTS = frozenset({"count", "weather", "tempo_de_tago"})


def _q_why_property(rec: dict, rng: random.Random) -> list[dict]:
    """Property-change attribution: "Kial la X estas Y?" → "Ĉar Z V-is ĝin."
    Every non-skip event with property_changes yields a question linking
    the resulting state back to the action that caused it."""
    global UNMARKED
    if not UNMARKED:
        UNMARKED = _load_unmarked()
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
        agent = ev["roles"].get("agent")
        agent_name = _name(agent, entities) if agent else None
        verb = ev["action"]
        for key, new_val in ev["property_changes"].items():
            if "|" not in key:
                continue
            eid, slot = key.split("|", 1)
            if slot in _WHY_PROP_SKIP_SLOTS:
                continue
            if new_val == UNMARKED.get(slot):
                continue
            ent = entities.get(eid)
            if ent is None:
                continue
            q = f"Kial la {ent['concept']} estas {new_val}?"
            verb_past = _past(verb)
            if agent_name:
                theme = ev["roles"].get("theme")
                if theme and isinstance(theme, str) and theme in entities:
                    obj_acc = _noun_acc(entities[theme]['concept'])
                    a_shapes = [
                        f"Ĉar {agent_name} {verb_past} la {obj_acc}.",
                        f"Pro tio, ke {agent_name} {verb_past} la {obj_acc}.",
                        f"{agent_name} {verb_past} la {obj_acc}.",
                    ]
                elif eid == ev["roles"].get("agent"):
                    a_shapes = [
                        f"Ĉar {agent_name} {verb_past}.",
                        f"Pro tio, ke {agent_name} {verb_past}.",
                        f"{agent_name} {verb_past}.",
                    ]
                else:
                    a_shapes = [
                        f"Ĉar {agent_name} {verb_past} ĝin.",
                        f"Pro tio, ke {agent_name} {verb_past} ĝin.",
                        f"{agent_name} {verb_past} ĝin.",
                    ]
                a = rng.choice(a_shapes)
            else:
                a = rng.choice([
                    f"Ĉar ĝi {verb_past}.",
                    f"Pro tio, ke ĝi {verb_past}.",
                ])
            out.append({
                "q": q, "a": a,
                "requires": [{"kind": "event", "event_id": ev["id"]}],
            })
    return out


_PERCEPTION_VERBS: set[str] | None = None


def _load_perception_verbs() -> set[str]:
    lex = _get_lex()
    sensory = frozenset({"see_capable", "hear_capable", "smell_capable"})
    return {
        name for name, a in lex.actions.items()
        if any(r.name == "instrument" and sensory & (
            getattr(r, "properties", {}) or {}).keys()
            for r in a.roles)
    }


def _q_enablement(rec: dict, rng: random.Random) -> list[dict]:
    """Purpose questions: "Kial X V-is?" → "Por V2-i la Z-on."
    Pairs each non-perception action with a random later action by
    the same agent. Randomized lookahead gives varying abstraction."""
    global _PERCEPTION_VERBS
    if _PERCEPTION_VERBS is None:
        _PERCEPTION_VERBS = _load_perception_verbs()
    events = rec.get("events", [])
    if not events:
        return []
    raw_ents = rec.get("entities", [])
    if not raw_ents or not isinstance(raw_ents[0], dict):
        return []
    entities = {e["eid"]: e for e in raw_ents}
    agentful = [
        e for e in events
        if e.get("roles", {}).get("agent")
        and not _should_skip_verb(e["action"])
        and e["action"] not in _PERCEPTION_VERBS
    ]
    if len(agentful) < 2:
        return []

    # Pre-compute per-event: source keys (theme/dest) and target
    # keys (theme/dest/instrument/parts) + agent, so the inner
    # loop is set intersection instead of per-call dict walks.
    src_keys = []
    tgt_keys = []
    agents = []
    for ev in agentful:
        sk = set()
        t = ev["roles"].get("theme")
        if isinstance(t, str): sk.add(t)
        d = ev["roles"].get("destination")
        if isinstance(d, str): sk.add(d)
        src_keys.append(sk)
        tk = set(sk)
        inst = ev["roles"].get("instrument")
        if isinstance(inst, str): tk.add(inst)
        parts = ev["roles"].get("parts")
        if isinstance(parts, list):
            tk.update(p for p in parts if isinstance(p, str))
        tgt_keys.append(tk)
        agents.append(ev["roles"]["agent"])

    out = []
    for i, ev in enumerate(agentful[:-1]):
        if not src_keys[i]:
            continue
        agent = agents[i]
        later = [
            agentful[j] for j in range(i + 1, len(agentful))
            if agents[j] == agent
            and src_keys[i] & tgt_keys[j]
            and agentful[j]["action"] != ev["action"]
        ]
        if not later:
            continue
        target = rng.choice(later)
        agent_name = _name(agent, entities)

        verb_past = _past(ev["action"])
        theme = ev["roles"].get("theme")
        dest = ev["roles"].get("destination")
        if theme and isinstance(theme, str) and theme in entities:
            theme_acc = _acc(entities[theme]["concept"])
            q_shapes = [
                f"Kial {agent_name} {verb_past} la {theme_acc}?",
                f"Por kio {agent_name} {verb_past} la {theme_acc}?",
            ]
        elif dest and dest in entities:
            dest_l = entities[dest]["concept"]
            q_shapes = [
                f"Kial {agent_name} {verb_past} al la {dest_l}?",
                f"Por kio {agent_name} {verb_past} al la {dest_l}?",
            ]
        else:
            q_shapes = [
                f"Kial {agent_name} {verb_past}?",
                f"Por kio {agent_name} {verb_past}?",
            ]
        q = rng.choice(q_shapes)

        tgt_verb = target["action"]
        tgt_theme = target["roles"].get("theme")
        tgt_dest = target["roles"].get("destination")
        if tgt_theme and isinstance(tgt_theme, str) and tgt_theme in entities:
            tgt_obj = f"la {_acc(entities[tgt_theme]['concept'])}"
            tgt_obj_short = f"{tgt_obj} {target.get('_short', '')}".strip()
            a_shapes = [
                f"Por {tgt_verb} {tgt_obj}.",
                f"Ĉar {agent_name} volis {tgt_verb} {tgt_obj}.",
                f"Por ke {agent_name} povu {tgt_verb} {tgt_obj}.",
                f"Por {tgt_verb} ĝin.",
            ]
        elif tgt_dest and tgt_dest in entities:
            tgt_obj = f"al la {entities[tgt_dest]['concept']}"
            a_shapes = [
                f"Por {tgt_verb} {tgt_obj}.",
                f"Ĉar {agent_name} volis {tgt_verb} {tgt_obj}.",
                f"Por ke {agent_name} povu {tgt_verb} {tgt_obj}.",
            ]
        else:
            a_shapes = [
                f"Por {tgt_verb}.",
                f"Ĉar {agent_name} volis {tgt_verb}.",
                f"Por ke {agent_name} povu {tgt_verb}.",
            ]
        a = rng.choice(a_shapes)
        out.append({
            "q": q, "a": a,
            "requires": [
                {"kind": "event", "event_id": ev["id"]},
                {"kind": "event", "event_id": target["id"]},
            ],
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
        havi_pat = {"kind": "relation", "rel": "havi",
                    "args[0]": owner_eid, "args[1]": item_eid}
        # "Kiu havis la X-on?"
        out.append({
            "q": f"Kiu havis la {_noun_acc(item_name)}?",
            "a": f"{owner_name}.",
            "requires": [havi_pat],
        })
        # "Kion Y havis?"
        if owner_ent["type"] == "person":
            out.append({
                "q": f"Kion {owner_name} havis?",
                "a": f"la {_noun_acc(item_name)}.",
                "requires": [havi_pat],
            })
        # "Kies estas la X?" — possessive question.
        # Only fires when owner is a person; "kies" with non-person
        # owner reads awkwardly.
        if owner_ent["type"] == "person":
            out.append({
                "q": f"Kies estas la {item_name}?",
                "a": f"de {owner_name}.",
                "requires": [havi_pat],
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
            "requires": [{
                "kind": "relation", "rel": "en",
                "args[0]": content_eid, "args[1]": container_eid,
            }],
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
            "requires": [{"kind": "intro", "concept": concept}],
        })
    # Negative: concepts that exist in the corpus but not this trace.
    # No requires pattern — the model defaults to "Ne" when it doesn't
    # see the concept, which is the correct behavior. Always disclosed
    # by absence (we're asserting an absence, which the prose's full
    # set of intros lets the model verify).
    if all_concepts:
        absent = list(all_concepts - present)
        rng.shuffle(absent)
        for concept in absent[:2]:
            out.append({
                "q": f"Ĉu estis {concept} en la sceno?",
                "a": _no(concept),
                "requires": [],
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
        by_loc.setdefault(container, []).append((contained, c_ent["concept"]))
    out = []
    for loc_eid, items in by_loc.items():
        if len(items) < 2:
            continue
        loc_ent = entities.get(loc_eid)
        if loc_ent is None:
            continue
        deduped = []
        seen_c = set()
        for eid, concept in items:
            if concept in seen_c:
                continue
            seen_c.add(concept)
            deduped.append((eid, concept))
            if len(deduped) >= 5:
                break
        concepts_only = [c for _, c in deduped]
        if len(concepts_only) == 1:
            listing = concepts_only[0]
        elif len(concepts_only) == 2:
            listing = f"{concepts_only[0]} kaj {concepts_only[1]}"
        else:
            listing = ", ".join(concepts_only[:-1]) + f", kaj {concepts_only[-1]}"
        q_shapes = [
            f"Kio troviĝas en la {loc_ent['concept']}?",
            f"Kio estas en la {loc_ent['concept']}?",
        ]
        out.append({
            "q": q_shapes[rng.randrange(len(q_shapes))],
            "a": listing + ".",
            "requires": [
                {"kind": "relation", "rel": "en",
                 "args[0]": eid, "args[1]": loc_eid}
                for eid, _ in deduped
            ],
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
            "requires": [{
                "kind": "relation", "rel": "en",
                "args[0]": contained, "args[1]": container,
            }],
        })
    return out


_MOVEMENT_VERBS: set[str] | None = None


def _load_movement_verbs() -> set[str]:
    lex = _get_lex()
    return {
        name for name, a in lex.actions.items()
        if any(r.name == "destination" for r in a.roles)
    }


def _q_location_at_end(rec: dict, rng: random.Random) -> list[dict]:
    """Where is entity X at the end of the trace? Tracks movement
    events (verbs with a destination role) and asks about the agent's
    final location when it differs from their starting position.
    Also asks about static location relations (en/sur/apud) from
    setup to get preposition diversity."""
    global _MOVEMENT_VERBS
    if _MOVEMENT_VERBS is None:
        _MOVEMENT_VERBS = _load_movement_verbs()
    events = rec.get("events", [])
    raw_ents = rec.get("entities", [])
    if not raw_ents or not isinstance(raw_ents[0], dict):
        return []
    entities = {e["eid"]: e for e in raw_ents}
    setup = rec.get("setup_relations", [])
    if not setup:
        return []

    setup_loc: dict[str, tuple[str, str]] = {}
    for r in setup:
        from esperanto_lm.ontology.containment import (
            containment_relation_names, resolve_containment)
        _lex = _get_lex()
        _spatial = containment_relation_names(
            resolve_containment(_lex), _lex) | {"apud"}
        if r["relation"] in _spatial and len(r["args"]) == 2:
            setup_loc[r["args"][0]] = (r["relation"], r["args"][1])

    final_loc: dict[str, tuple[str, str]] = dict(setup_loc)

    for ev in events:
        agent = ev.get("roles", {}).get("agent")
        dest = ev.get("roles", {}).get("destination")
        if agent and dest and ev["action"] in _MOVEMENT_VERBS:
            final_loc[agent] = ("en", dest)

    out = []
    seen = set()
    for eid, (prep, dest_eid) in final_loc.items():
        ent = entities.get(eid)
        dest_ent = entities.get(dest_eid)
        if ent is None or dest_ent is None:
            continue
        if ent["type"] == "abstract":
            continue
        if "_" in eid and eid != ent["concept"]:
            continue
        concept = ent["concept"]
        if concept in seen:
            continue
        seen.add(concept)
        moved = setup_loc.get(eid) != (prep, dest_eid)
        name = _name(eid, entities) if ent["type"] == "person" else concept
        if moved:
            q = rng.choice([
                f"Kie estas {name} fine de la rakonto?",
                f"Kie troviĝas {name} fine?",
            ])
        else:
            q = rng.choice([
                f"Kie estas la {concept}?",
                f"Kie troviĝas la {concept}?",
            ])
        surface_prep = prep
        if prep == "apud":
            surface_prep = rng.choice(["apud", "ĉe"])
        a = f"{surface_prep.capitalize()} la {dest_ent['concept']}."
        # Last position: from setup (literal relation in prose) OR
        # established via a movement event (which is disclosed).
        requires: list[dict] = []
        if moved:
            for ev in events:
                if (ev.get("roles", {}).get("agent") == eid
                        and ev.get("roles", {}).get("destination") == dest_eid
                        and ev["action"] in _MOVEMENT_VERBS):
                    requires.append({"kind": "event", "event_id": ev["id"]})
                    break
        else:
            requires.append({"kind": "relation", "rel": prep,
                             "args[0]": eid, "args[1]": dest_eid})
        out.append({"q": q, "a": a, "requires": requires})
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
        verb = ev["action"]
        sentence_facts = rec.get("sentence_facts") or []
        theme_phrase_nom = _specifier_noun_phrase(
            theme, entities, sentence_facts, case="nom")
        if theme_phrase_nom is None:
            theme_phrase_nom = "la " + theme_ent["concept"]

        answers = []
        subj = _subject_phrase(ev, entities)
        if subj:
            answers.append(f"{subj} {_past(verb)} ĝin.")
        answers.append(f"Ĝi estis {_passive_participle(verb)}.")

        q = rng.choice([
            f"Kio okazis al {theme_phrase_nom}?",
            f"Kio okazis kun {theme_phrase_nom}?",
        ])
        out.append({
            "q": q, "a": rng.choice(answers),
            "requires": [{"kind": "event", "event_id": ev["id"]}],
        })
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
        ev_pat = {"kind": "event", "event_id": ev["id"]}
        shapes = [
            (f"Kien {agent_name} {_past(ev['action'])}?",
             f"Al la {dest_ent['concept']}."),
            (f"Al kiu loko {agent_name} {_past(ev['action'])}?",
             f"Al la {dest_ent['concept']}."),
        ]
        q, a = shapes[rng.randrange(len(shapes))]
        out.append({"q": q, "a": a, "requires": [ev_pat]})
        # Reverse: "Kiu iris al la X?" → agent
        out.append({
            "q": f"Kiu {_past(ev['action'])} al la {dest_ent['concept']}?",
            "a": f"{agent_name}.",
            "requires": [ev_pat],
        })
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
        ev_pat = {"kind": "event", "event_id": ev["id"]}
        theme = ev["roles"].get("theme")
        if theme and isinstance(theme, str):
            theme_ent = entities.get(theme)
            if theme_ent is not None:
                sentence_facts = rec.get("sentence_facts") or []
                theme_phrase = _specifier_noun_phrase(
                    theme, entities, sentence_facts, case="acc")
                if theme_phrase is None:
                    theme_phrase = "la " + _noun_acc(theme_ent["concept"])
                out.append({
                    "q": (f"Al kiu {agent_name} {_past(ev['action'])} "
                          f"{theme_phrase}?"),
                    "a": f"Al {recip_name}.",
                    "requires": [ev_pat],
                })
                # Reverse: who gave/showed/told to this recipient?
                out.append({
                    "q": (f"De kiu {recip_name} ricevis "
                          f"{theme_phrase}?"),
                    "a": f"De {agent_name}.",
                    "requires": [ev_pat],
                })
        else:
            out.append({
                "q": f"Al kiu {agent_name} {_past(ev['action'])}?",
                "a": f"Al {recip_name}.",
                "requires": [ev_pat],
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
        out.append({
            "q": q, "a": a,
            "requires": [
                {"kind": "event", "event_id": ev["id"]},
                {"kind": "relation", "rel": "en",
                 "args[0]": agent, "args[1]": agent_loc[agent]},
            ],
        })
    return out


_CONCEPT_CATEGORIES: dict[str, list[str]] = {}


def _load_concept_categories() -> dict[str, list[str]]:
    """Load concept → category list from the lex. Used by coreference
    to map "la besto" → the concept that has category=besto."""
    lex = _get_lex()
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
            "requires": [{
                "kind": "category", "entity": ent["eid"], "value": alias,
            }],
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
        subj = _subject_phrase(ev, entities)
        if subj is None:
            return _past(ev["action"])
        theme_phrase = _theme_phrase(ev, entities, case="acc")
        if theme_phrase is not None:
            return f"{subj} {_past(ev['action'])} {theme_phrase}"
        return f"{subj} {_past(ev['action'])}"

    for i in range(len(events) - 1):
        prev, nxt = events[i], events[i + 1]
        if _should_skip_verb(prev["action"]) or _should_skip_verb(nxt["action"]):
            continue
        # Build "Kio okazis post la X-ado de la Y?" — use the
        # verb's noun form (action+o) so the question reads
        # idiomatically.
        prev_verb = prev["action"]
        nxt_verb = nxt["action"]
        prev_noun = (prev_verb[:-1] if prev_verb.endswith("i")
                     else prev_verb) + "ado"
        nxt_noun = (nxt_verb[:-1] if nxt_verb.endswith("i")
                    else nxt_verb) + "ado"
        req = [
            {"kind": "event", "event_id": prev["id"]},
            {"kind": "event", "event_id": nxt["id"]},
        ]
        out.append({
            "q": f"Kio okazis post la {prev_noun}?",
            "a": describe(nxt) + ".",
            "requires": req,
        })
        # Reverse: "Kio okazis antaŭ la X-ado?" → previous event.
        out.append({
            "q": f"Kio okazis antaŭ la {nxt_noun}?",
            "a": describe(prev) + ".",
            "requires": req,
        })
    return out


def _q_cloze_event(rec: dict, rng: random.Random) -> list[dict]:
    """Cloze fill-in-the-blank from event sentences. For each event,
    pick one of {agent, verb, theme, destination} to blank and ask
    the model to complete the sentence. Trains token-level prediction
    against the trace facts as a complement to question-shaped Q/A —
    same disclosure, different surface task.

    The answer is always a single content word (a name, a verb-past
    form, or a noun in accusative). Matcher tolerates trailing
    punctuation via the substring rules."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        agent_id = ev["roles"].get("agent")
        if not agent_id or isinstance(agent_id, list):
            continue
        agent_ent = entities.get(agent_id)
        if agent_ent is None or agent_ent["type"] != "person":
            continue
        agent_name = _name(agent_id, entities)
        if not agent_name or agent_name.lower() == "mondo":
            continue
        verb_past = _past(ev["action"])
        req = [{"kind": "event", "event_id": ev["id"]}]

        theme_id = ev["roles"].get("theme")
        theme_phrase = None
        theme_word_acc = None
        if theme_id and not isinstance(theme_id, list):
            theme_ent = entities.get(theme_id)
            if theme_ent is not None and "_" not in theme_id:
                theme_word_acc = _noun_acc(theme_ent["concept"])
                theme_phrase = f"la {theme_word_acc}"

        dest_id = ev["roles"].get("destination") or ev["roles"].get("location")
        dest_word = None
        if dest_id and not isinstance(dest_id, list):
            dest_ent = entities.get(dest_id)
            if dest_ent is not None and dest_ent["type"] == "location":
                dest_word = dest_ent["concept"]

        variants = []
        # Variant: blank agent (only when sentence has a theme so the
        # context is informative).
        if theme_phrase:
            variants.append({
                "q": (f"Kompletigu: ___ {verb_past} {theme_phrase}."),
                "a": agent_name,
            })
        # Variant: blank verb.
        if theme_phrase:
            variants.append({
                "q": (f"Kompletigu: {agent_name} ___ {theme_phrase}."),
                "a": verb_past,
            })
        # Variant: blank theme noun (the accusative form).
        if theme_phrase:
            variants.append({
                "q": (f"Kompletigu: {agent_name} {verb_past} la ___."),
                "a": theme_word_acc,
            })
        # Variant: blank destination on motion verbs ("Kien X iris?").
        if dest_word and ev["action"] in {"iri", "kuri", "veni",
                                            "eniri", "flugi"}:
            variants.append({
                "q": (f"Kompletigu: {agent_name} {verb_past} "
                      f"al la ___."),
                "a": dest_word,
            })

        if not variants:
            continue
        # Emit one variant per event to keep volume balanced — picker
        # adds variety across the corpus without inflating any single
        # event's contribution.
        pick = variants[rng.randrange(len(variants))]
        out.append({"q": pick["q"], "a": pick["a"], "requires": req})
    return out


def _q_active_passive(rec: dict, rng: random.Random) -> list[dict]:
    """Active ↔ passive transformation. Given an active event
    sentence ("Anna prenis la pomon"), rewrite as passive
    ("La pomo estis prenita de Anna"), and vice versa. Trains
    voice manipulation directly — distinct from `_q_de_agent_from_passive`
    which extracts the agent from a passive question."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        agent_id = ev["roles"].get("agent")
        theme_id = ev["roles"].get("theme")
        if (not agent_id or not theme_id
                or isinstance(agent_id, list)
                or isinstance(theme_id, list)):
            continue
        agent_ent = entities.get(agent_id)
        theme_ent = entities.get(theme_id)
        if agent_ent is None or theme_ent is None:
            continue
        if agent_ent["type"] != "person":
            continue
        agent_name = _name(agent_id, entities)
        if not agent_name or agent_name.lower() == "mondo":
            continue
        # Skip multi-word agent/theme inflections we can't reliably
        # decline (e.g. compound names with parentheses).
        if "_" in theme_id and theme_id != theme_ent["concept"]:
            continue
        theme_concept = theme_ent["concept"]
        verb_past = _past(ev["action"])
        participle = _passive_participle(ev["action"])
        active = f"{agent_name} {verb_past} la {_noun_acc(theme_concept)}"
        passive = f"La {theme_concept} estis {participle} de {agent_name}"
        req = [{"kind": "event", "event_id": ev["id"]}]
        # Pick direction.
        if rng.random() < 0.5:
            out.append({
                "q": f"Skribu pasive: {active}.",
                "a": f"{passive}.",
                "requires": req,
            })
        else:
            out.append({
                "q": f"Skribu aktive: {passive}.",
                "a": f"{active}.",
                "requires": req,
            })
    return out


def _q_sentence_construction(rec: dict, rng: random.Random) -> list[dict]:
    """Kit → prose: given (agent, verb, theme), write the sentence.
    Inverse of Q/A — teaches generation from structured input rather
    than extraction from text. Reuses the same trace facts as
    `_q_cloze_event` but flips the input/output direction."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        agent_id = ev["roles"].get("agent")
        theme_id = ev["roles"].get("theme")
        if (not agent_id or not theme_id
                or isinstance(agent_id, list)
                or isinstance(theme_id, list)):
            continue
        agent_ent = entities.get(agent_id)
        theme_ent = entities.get(theme_id)
        if agent_ent is None or theme_ent is None:
            continue
        if agent_ent["type"] != "person":
            continue
        agent_name = _name(agent_id, entities)
        if not agent_name or agent_name.lower() == "mondo":
            continue
        if "_" in theme_id and theme_id != theme_ent["concept"]:
            continue
        theme_concept = theme_ent["concept"]
        verb_lemma = ev["action"]
        verb_past = _past(verb_lemma)
        kit = (f"aganto={agent_name}, verbo={verb_lemma}, "
               f"objekto={theme_concept}")
        sentence = (f"{agent_name} {verb_past} la "
                    f"{_noun_acc(theme_concept)}.")
        out.append({
            "q": f"Skribu frazon: {kit}.",
            "a": sentence,
            "requires": [{"kind": "event", "event_id": ev["id"]}],
        })
    return out


def _q_true_false(rec: dict, rng: random.Random) -> list[dict]:
    """Discriminative true/false on event claims. For each event,
    generate one truthful claim (✓ → 'Jes') and one minimally
    perturbed false claim (✗ → 'Ne'). Perturbations swap the
    theme to another entity in the scene, or swap the agent to
    another person. Trains the model to read a claim and verify
    it against the prose — adversarial complement to the
    extractive Q/A shapes."""
    events = rec.get("events", [])
    if not events:
        return []
    raw_ents = rec.get("entities", [])
    entities = {e["eid"]: e for e in raw_ents}
    persons = [e["eid"] for e in raw_ents if e.get("type") == "person"]
    artifacts = [e["eid"] for e in raw_ents
                 if e.get("type") in ("artifact", "substance",
                                       "animal", "food")]
    out = []
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        agent_id = ev["roles"].get("agent")
        theme_id = ev["roles"].get("theme")
        if (not agent_id or not theme_id
                or isinstance(agent_id, list)
                or isinstance(theme_id, list)):
            continue
        agent_ent = entities.get(agent_id)
        theme_ent = entities.get(theme_id)
        if agent_ent is None or theme_ent is None:
            continue
        if agent_ent["type"] != "person":
            continue
        if "_" in theme_id and theme_id != theme_ent["concept"]:
            continue
        agent_name = _name(agent_id, entities)
        if not agent_name or agent_name.lower() == "mondo":
            continue
        verb_past = _past(ev["action"])
        theme_concept = theme_ent["concept"]
        true_claim = (f"{agent_name} {verb_past} la "
                      f"{_noun_acc(theme_concept)}")
        req = [{"kind": "event", "event_id": ev["id"]}]
        out.append({
            "q": f"Ĉu vere?: '{true_claim}.'",
            "a": "Jes.",
            "requires": req,
        })
        # Build a false claim by perturbing either the theme or the
        # agent. Pick a swap candidate from in-scene entities so the
        # false claim still references the right "world" (the model
        # has to disprove it from the events themselves, not by
        # noticing an out-of-scene name).
        swap_kind = None
        if len(artifacts) >= 2:
            alt_themes = [t for t in artifacts if t != theme_id
                          and "_" not in t]
            if alt_themes:
                swap_kind = "theme"
        if swap_kind is None and len(persons) >= 2:
            alt_agents = [p for p in persons if p != agent_id]
            if alt_agents:
                swap_kind = "agent"
        if swap_kind is None:
            continue
        if swap_kind == "theme":
            alt_id = rng.choice(alt_themes)
            alt_concept = entities[alt_id]["concept"]
            false_claim = (f"{agent_name} {verb_past} la "
                           f"{_noun_acc(alt_concept)}")
        else:
            alt_id = rng.choice(alt_agents)
            alt_name = _name(alt_id, entities)
            if not alt_name or alt_name.lower() == "mondo":
                continue
            false_claim = (f"{alt_name} {verb_past} la "
                           f"{_noun_acc(theme_concept)}")
        out.append({
            "q": f"Ĉu vere?: '{false_claim}.'",
            "a": "Ne.",
            "requires": req,
        })
    return out


def _q_pronoun_resolution(rec: dict, rng: random.Random) -> list[dict]:
    """Generative pronoun resolution: render an event sentence with
    pronouns substituted for the agent ('li'/'ŝi') and the theme
    ('ĝin'), then ask the model to bind each pronoun back to its
    referent. Forces explicit coref against the trace facts.

    Gender comes from `entity.properties.gender` (matches the
    realizer's `_pronoun_for_person`); 'li' is the default.
    Theme pronoun is always 'ĝin' (accusative, inanimate)."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        agent_id = ev["roles"].get("agent")
        theme_id = ev["roles"].get("theme")
        if (not agent_id or not theme_id
                or isinstance(agent_id, list)
                or isinstance(theme_id, list)):
            continue
        agent_ent = entities.get(agent_id)
        theme_ent = entities.get(theme_id)
        if agent_ent is None or theme_ent is None:
            continue
        if agent_ent["type"] != "person":
            continue
        if theme_ent["type"] == "person":
            continue  # "ĝin" doesn't fit a person theme
        if "_" in theme_id and theme_id != theme_ent["concept"]:
            continue
        agent_name = _name(agent_id, entities)
        if not agent_name or agent_name.lower() == "mondo":
            continue
        gender = agent_ent.get("properties", {}).get("gender")
        if isinstance(gender, list):
            gender = gender[0] if gender else None
        agent_pronoun = "ŝi" if gender == "virino" else "li"
        verb_past = _past(ev["action"])
        theme_concept = theme_ent["concept"]
        pronoun_sentence = f"{agent_pronoun.capitalize()} {verb_past} ĝin"
        q = (f"En la frazo \"{pronoun_sentence}\", "
             f"kiu estas '{agent_pronoun}' kaj kio estas 'ĝin'?")
        a = f"{agent_pronoun}={agent_name}, ĝin={theme_concept}"
        out.append({
            "q": q,
            "a": a,
            "requires": [{"kind": "event", "event_id": ev["id"]}],
        })
    return out


def _q_agent_summary(rec: dict, rng: random.Random) -> list[dict]:
    """What did X do? → multi-action summary listing the agent's
    actions. Trains longer extractive answers."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    by_agent: dict[str, list[dict]] = {}
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        agent = ev.get("roles", {}).get("agent")
        if agent:
            by_agent.setdefault(agent, []).append(ev)
    out = []
    for agent, acts in by_agent.items():
        if len(acts) < 2:
            continue
        agent_name = _name(agent, entities)
        # Subject phrase across the agent's events: if any of the
        # acts had a companion bound to this agent's events, surface
        # them in the answer ("Petro kaj Maria V-is..."). For
        # multi-event summaries we pick the most-common companion.
        from collections import Counter
        comp_counter: Counter = Counter()
        for ev in acts:
            c = ev["roles"].get("companion")
            if isinstance(c, str) and c in entities:
                comp_counter[c] += 1
        if comp_counter:
            top_comp_eid, top_comp_n = comp_counter.most_common(1)[0]
            if top_comp_n >= max(1, len(acts) // 2):
                subj = f"{agent_name} kaj {_name(top_comp_eid, entities)}"
            else:
                subj = agent_name
        else:
            subj = agent_name
        descs = []
        for ev in acts[:4]:
            theme_phrase = _theme_phrase(ev, entities, case="acc")
            if theme_phrase is not None:
                descs.append(
                    f"{_past(ev['action'])} {theme_phrase}")
            else:
                dest = ev["roles"].get("destination")
                if dest and dest in entities:
                    descs.append(
                        f"{_past(ev['action'])} al la "
                        f"{entities[dest]['concept']}")
                else:
                    descs.append(_past(ev["action"]))
        if len(descs) == 1:
            listing = descs[0]
        elif len(descs) == 2:
            listing = f"{descs[0]} kaj {descs[1]}"
        else:
            listing = ", ".join(descs[:-1]) + f", kaj {descs[-1]}"
        q = rng.choice([
            f"Kion faris {agent_name}?",
            f"Kion {agent_name} faris en la rakonto?",
        ])
        a = f"{subj} {listing}."
        out.append({
            "q": q, "a": a,
            "requires": [
                {"kind": "event", "event_id": e["id"]}
                for e in acts[:4]
            ],
        })
    return out


def _q_negation(rec: dict, rng: random.Random) -> list[dict]:
    """Did X do Y to Z? → No, X did W to Z. Picks an action that
    DID happen to an entity and asks about a different verb."""
    events = rec.get("events", [])
    if not events:
        return []
    raw_ents = rec.get("entities", [])
    if not raw_ents or not isinstance(raw_ents[0], dict):
        return []
    entities = {e["eid"]: e for e in raw_ents}
    agentful = [
        e for e in events
        if e.get("roles", {}).get("agent")
        and e.get("roles", {}).get("theme")
        and not _should_skip_verb(e["action"])
        and isinstance(e["roles"].get("theme"), str)
    ]
    if not agentful:
        return []
    verbs_used = {e["action"] for e in agentful}
    global _ALL_ACTIONS
    if not _ALL_ACTIONS:
        _ALL_ACTIONS = set(_get_lex().actions.keys())
    # Group actions by transitivity shape so the wrong verb takes the
    # same argument structure as the real one.
    _NEG_BY_SHAPE: dict[str, list[str]] = getattr(
        _q_negation, "_by_shape", {})
    if not _NEG_BY_SHAPE:
        lex = _get_lex()
        for name, a in lex.actions.items():
            if _should_skip_verb(name):
                continue
            has_theme = any(r.name == "theme" for r in a.roles)
            shape = "transitive" if has_theme else "intransitive"
            _NEG_BY_SHAPE.setdefault(shape, []).append(name)
        _q_negation._by_shape = _NEG_BY_SHAPE
    out = []
    for ev in agentful[:3]:
        shape = "transitive"
        candidates = [
            v for v in _NEG_BY_SHAPE.get(shape, [])
            if v not in verbs_used]
        if not candidates:
            continue
        wrong_verb = rng.choice(candidates)
        agent = ev["roles"]["agent"]
        theme = ev["roles"]["theme"]
        agent_name = _name(agent, entities)
        subj = _subject_phrase(ev, entities) or agent_name
        sentence_facts = rec.get("sentence_facts") or []
        theme_phrase = _specifier_noun_phrase(
            theme, entities, sentence_facts, case="acc")
        if theme_phrase is None:
            theme_phrase = _theme_phrase(ev, entities, case="acc")
        if theme_phrase is None:
            continue
        q = f"Ĉu {agent_name} {_past(wrong_verb)} {theme_phrase}?"
        a = f"Ne, {subj} {_past(ev['action'])} {theme_phrase}."
        out.append({
            "q": q, "a": a,
            "requires": [{"kind": "event", "event_id": ev["id"]}],
        })
    return out


def _q_multiple_choice(rec: dict, rng: random.Random) -> list[dict]:
    """Did X verb the A, B, or C? → X verbed the B. Distractors are
    other concepts from the same scene; count varies 1-3."""
    events = rec.get("events", [])
    if not events:
        return []
    raw_ents = rec.get("entities", [])
    if not raw_ents or not isinstance(raw_ents[0], dict):
        return []
    entities = {e["eid"]: e for e in raw_ents}
    scene_concepts = [
        e["concept"] for e in raw_ents
        if e["type"] not in ("location", "abstract", "person")
        and "_" not in e["eid"]]
    if len(scene_concepts) < 2:
        return []
    agentful = [
        e for e in events
        if e.get("roles", {}).get("agent")
        and isinstance(e["roles"].get("theme"), str)
        and not _should_skip_verb(e["action"])
    ]
    out = []
    for ev in agentful[:3]:
        theme = ev["roles"]["theme"]
        theme_ent = entities.get(theme)
        if theme_ent is None:
            continue
        correct = theme_ent["concept"]
        distractors = [c for c in scene_concepts if c != correct]
        if not distractors:
            continue
        n_dist = min(rng.randint(1, 3), len(distractors))
        picked = rng.sample(distractors, n_dist)
        options = [correct] + picked
        rng.shuffle(options)
        agent_name = _name(ev["roles"]["agent"], entities)
        subj = _subject_phrase(ev, entities) or agent_name
        acc_options = [_noun_acc(o) for o in options]
        if len(acc_options) == 2:
            option_str = f"{acc_options[0]} aŭ {acc_options[1]}"
        else:
            option_str = (", ".join(acc_options[:-1])
                          + f", aŭ {acc_options[-1]}")
        q = f"Ĉu {agent_name} {_past(ev['action'])} la {option_str}?"
        a = f"{subj} {_past(ev['action'])} la {_noun_acc(correct)}."
        out.append({
            "q": q, "a": a,
            "requires": [{"kind": "event", "event_id": ev["id"]}],
        })
    return out


def _q_entity_journey(rec: dict, rng: random.Random) -> list[dict]:
    """What did the agent do with/to entity X? Lists the sequence of
    actions involving the entity across different roles (theme, then
    instrument, etc.). Trains multi-step extraction."""
    events = rec.get("events", [])
    if not events:
        return []
    raw_ents = rec.get("entities", [])
    if not raw_ents or not isinstance(raw_ents[0], dict):
        return []
    entities = {e["eid"]: e for e in raw_ents}

    global _PERCEPTION_VERBS
    if _PERCEPTION_VERBS is None:
        _PERCEPTION_VERBS = _load_perception_verbs()
    by_agent_entity: dict[tuple[str, str], list[dict]] = {}
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        if ev["action"] in _PERCEPTION_VERBS:
            continue
        agent = ev.get("roles", {}).get("agent")
        if not agent:
            continue
        for role, val in ev["roles"].items():
            if role == "agent":
                continue
            if isinstance(val, str) and val in entities:
                ent = entities[val]
                if ent["type"] in ("location", "abstract", "person"):
                    continue
                if "_" in val and val != ent["concept"]:
                    continue
                by_agent_entity.setdefault((agent, val), []).append(ev)
            elif isinstance(val, list):
                for v in val:
                    if isinstance(v, str) and v in entities:
                        by_agent_entity.setdefault(
                            (agent, v), []).append(ev)

    lex = _get_lex()
    out = []
    for (agent, eid), evts in by_agent_entity.items():
        unique_actions = []
        seen = set()
        for ev in evts:
            if ev["action"] not in seen:
                unique_actions.append(ev)
                seen.add(ev["action"])
        if len(unique_actions) < 2:
            continue
        agent_name = _name(agent, entities)
        ent = entities[eid]
        concept = ent["concept"]

        descs = []
        for ev in unique_actions[:4]:
            role_name = None
            for rn, rv in ev["roles"].items():
                if rn == "agent":
                    continue
                if isinstance(rv, str) and rv == eid:
                    role_name = rn
                    break
                if isinstance(rv, list) and eid in rv:
                    role_name = rn
                    break
            if role_name == "theme":
                descs.append(f"{_past(ev['action'])} ĝin")
            elif role_name is not None:
                action_def = lex.actions.get(ev["action"])
                prep = None
                if action_def:
                    rd = next((r for r in action_def.roles
                               if r.name == role_name), None)
                    if rd:
                        prep = getattr(rd, "preposition", None)
                theme_eid = ev["roles"].get("theme")
                if (theme_eid and isinstance(theme_eid, str)
                        and theme_eid in entities):
                    theme_form = _noun_acc(entities[theme_eid]["concept"])
                    if prep:
                        descs.append(
                            f"{_past(ev['action'])} la {theme_form} "
                            f"{prep} ĝi")
                    else:
                        descs.append(
                            f"{_past(ev['action'])} la {theme_form}")
                else:
                    descs.append(f"{_past(ev['action'])} ĝin")

        if len(descs) == 2:
            listing = f"{descs[0]} kaj {descs[1]}"
        else:
            listing = ", ".join(descs[:-1]) + f", kaj {descs[-1]}"

        q = rng.choice([
            f"Kion faris {agent_name} kun la {concept}?",
            f"Kion {agent_name} faris al la {concept}?",
        ])
        # Most-common companion across this agent×entity series — if
        # consistent enough, surface joint subject in the answer.
        from collections import Counter
        comp_counter: Counter = Counter()
        for ev in unique_actions:
            c = ev["roles"].get("companion")
            if isinstance(c, str) and c in entities:
                comp_counter[c] += 1
        subj = agent_name
        if comp_counter:
            top_comp_eid, top_comp_n = comp_counter.most_common(1)[0]
            if top_comp_n >= max(1, len(unique_actions) // 2):
                subj = f"{agent_name} kaj {_name(top_comp_eid, entities)}"
        a = f"{subj} {listing}."
        out.append({
            "q": q, "a": a,
            "requires": [
                {"kind": "event", "event_id": e["id"]}
                for e in unique_actions[:4]
            ],
        })
    return out


def _q_definition(rec: dict, rng: random.Random) -> list[dict]:
    """What is X? → X estas Y. Only for entities that got a definition
    sentence in the prose (tracked by defined_entities on the trace)."""
    defined = rec.get("defined_entities", [])
    if not defined:
        return []
    raw_ents = rec.get("entities", [])
    if not raw_ents or not isinstance(raw_ents[0], dict):
        return []
    entities = {e["eid"]: e for e in raw_ents}
    from esperanto_lm.ontology.realize.plan import _build_definition
    lex = _get_lex()
    out = []
    for eid in defined:
        ent = entities.get(eid)
        if ent is None:
            continue
        defn = _build_definition(ent["concept"], lex)
        if defn is None:
            continue
        concept = ent["concept"]
        q = rng.choice([
            f"Kio estas {concept}?",
            f"Kio estas la {concept}?",
        ])
        a = defn
        out.append({
            "q": q, "a": a,
            "requires": [{"kind": "definition", "entity": eid}],
        })
    return out


def _q_companion(rec: dict, rng: random.Random) -> list[dict]:
    """For events with a bound companion role: "Kun kiu X verbis?"
    → "Kun Y." Trains kun-companion extraction; pairs with the joint
    "X kaj Y" answer that _q_action_attribution now emits for the
    same events."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        agent = ev["roles"].get("agent")
        companion = ev["roles"].get("companion")
        if not agent or not companion:
            continue
        agent_ent = entities.get(agent)
        companion_ent = entities.get(companion)
        if agent_ent is None or companion_ent is None:
            continue
        agent_name = _name(agent, entities)
        companion_name = _name(companion, entities)
        verb_past = _past(ev["action"])
        q = rng.choice([
            f"Kun kiu {agent_name} {verb_past}?",
            f"Kun kiu {verb_past} {agent_name}?",
        ])
        a = rng.choice([
            f"Kun {companion_name}.",
            f"{companion_name}.",
            f"{agent_name} {verb_past} kun {companion_name}.",
        ])
        out.append({
            "q": q, "a": a,
            "requires": [{"kind": "event", "event_id": ev["id"]}],
        })
    return out


def _q_intransitive_subject(rec: dict, rng: random.Random) -> list[dict]:
    """Subject-attribution for intransitive events (no theme bound).
    "Kiu V-is (en LOC)?" → "X." or "X kaj Y." (joint, when companion
    bound). Covers the dormi/ripozi/iri/naĝi/veni class — companion-
    bearing verbs whose prose often renders "X kaj Y dormis" and the
    model otherwise never sees a subject-extraction Q for these."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    scene = rec.get("scene")
    out = []
    for ev in events:
        if _should_skip_verb(ev["action"]):
            continue
        agent = ev["roles"].get("agent")
        if agent is None:
            continue
        theme = ev["roles"].get("theme")
        if theme is not None:
            continue
        agent_ent = entities.get(agent)
        if agent_ent is None:
            continue
        agent_name = _name(agent, entities)
        companion = ev["roles"].get("companion")
        companion_ent = (entities.get(companion)
                         if companion else None)
        if companion_ent is not None:
            companion_name = _name(companion, entities)
            subj = f"{agent_name} kaj {companion_name}"
        else:
            subj = agent_name
        verb_past = _past(ev["action"])
        ans = rng.choice([
            f"{subj}.",
            f"{subj} {verb_past}.",
            f"Estis {subj}.",
        ])
        q_shapes = [f"Kiu {verb_past}?"]
        if scene:
            q_shapes.append(f"Kiu {verb_past} en la {scene}?")
        q = rng.choice(q_shapes)
        out.append({
            "q": q, "a": ans,
            "requires": [{"kind": "event", "event_id": ev["id"]}],
        })
    return out


# Per-biography-prop Q/A templates. Each entry is
#   prop → (question_template, short_answer, long_answer)
# `{name}` = entity display name; `{value}` = resolved KB value.
# Short answer is the bare value; long answer mirrors the prose
# template ("en {value}", etc.) — both go into accepted_answers so
# the eval matcher accepts either form.
_BIO_Q_TEMPLATES: dict[str, tuple[str, str, str]] = {
    # Person — biography
    "naskiĝloko":       ("Kie naskiĝis {name}?",                "{value}", "en {value}"),
    "mortejo":          ("Kie mortis {name}?",                  "{value}", "en {value}"),
    "ŝtataneco":        ("Kies civitano estis {name}?",         "{value}", "civitano de {value}"),
    "parolas_lingvon":  ("Kiun lingvon parolis {name}?",        "{value}", "{value}"),
    "grava_verko":      ("Kion verkis {name}?",                 "{value}", "{value}"),
    "geedzo":           ("Kiu estis la geedzo de {name}?",      "{value}", "{value}"),
    "naskiĝdato":       ("En kiu jaro naskiĝis {name}?",        "{value}", "en {value}"),
    "mortdato":         ("En kiu jaro mortis {name}?",          "{value}", "en {value}"),
    # Places — geography
    "ĉefurbo":          ("Kio estas la ĉefurbo de {name}?",     "{value}", "{value}"),
    "najbara":          ("Al kiu lando estas najbara {name}?",  "{value}", "{value}"),
    "loko":             ("Kie situas {name}?",                  "{value}", "en {value}"),
    "oficiala_lingvo":  ("Kio estas la oficiala lingvo de {name}?", "{value}", "{value}"),
    "loĝantaro":        ("Kio estas la loĝantaro de {name}?",   "{value}", "{value}"),
    # Organizations / works
    "fondinto":         ("Kiu fondis {name}?",                  "{value}", "{value}"),
    "verkinto":         ("Kiu verkis {name}?",                  "{value}", "{value}"),
    "reĝisoro":         ("Kiu reĝisoris {name}?",               "{value}", "{value}"),
}


def _q_biography(rec: dict, rng: random.Random) -> list[dict]:
    """Q/A from `biography` Facts emitted by KB-grounded entity
    definitions. Each fact disclosed in prose ("Mozart naskiĝis en
    Salzburg.") generates a corresponding Q/A ("Kie naskiĝis Mozart?"
    → "Salzburg" / "en Salzburg"). Falls out of the existing disclosure
    framework — no new infra, just a new question kind."""
    from esperanto_lm.ontology.realize.render import _render_person_name
    sentence_facts = rec.get("sentence_facts") or []
    entities = {e["eid"]: e for e in rec["entities"]}
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()  # dedupe by (entity, prop)
    for _idx, facts in sentence_facts:
        for f in facts:
            if f.get("kind") != "biography":
                continue
            eid = f.get("entity")
            prop = f.get("prop")
            value = f.get("value")
            if not (eid and prop and value):
                continue
            if (eid, prop) in seen:
                continue
            seen.add((eid, prop))
            template = _BIO_Q_TEMPLATES.get(prop)
            if template is None:
                continue
            q_t, a_short_t, a_long_t = template
            ent = entities.get(eid)
            if ent is None:
                continue
            # Display name: grounded entities have eid as a slug of
            # the canonical EO label. Slug → display via the
            # render helper used elsewhere ("marie_curie" → "Marie
            # Curie"). Fabricated entities don't carry biography
            # facts, so no fallback needed.
            display_name = _render_person_name(eid)
            q = q_t.format(name=display_name)
            a_short = a_short_t.format(value=value)
            a_long = a_long_t.format(value=value)
            ans = rng.choice([a_short, a_long])
            out.append({
                "q": q,
                "a": ans,
                "requires": [{
                    "kind": "biography", "entity": eid,
                    "prop": prop, "value": value,
                }],
            })
    return out


def _q_de_agent_from_passive(rec: dict, rng: random.Random) -> list[dict]:
    """De-question for agent extraction: "De kiu estis V-ita la X?"
    → "De AGENT." Works regardless of whether the prose rendered the
    event actively or passively — the event fact + agent role is
    enough."""
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
        if not agent or not theme or isinstance(theme, list):
            continue
        agent_ent = entities.get(agent)
        theme_ent = entities.get(theme)
        if agent_ent is None or theme_ent is None:
            continue
        agent_name = _name(agent, entities)
        participle = _passive_participle(ev["action"])
        theme_concept = theme_ent["concept"]
        q_shapes = [
            f"De kiu estis {participle} la {theme_concept}?",
            f"De kiu la {theme_concept} estis {participle}?",
        ]
        q = q_shapes[rng.randrange(len(q_shapes))]
        a = rng.choice([
            f"De {agent_name}.",
            f"{agent_name}.",
            f"La {theme_concept} estis {participle} de {agent_name}.",
        ])
        out.append({
            "q": q, "a": a,
            "requires": [{"kind": "event", "event_id": ev["id"]}],
        })
    return out


def _q_de_parts_from_definition(rec: dict, rng: random.Random) -> list[dict]:
    """De-question for parts extraction: "De kio estas farita la X?"
    → "El A kaj B." Requires the entity to have been defined with a
    parts tail ("farita el A kaj B"); the havas_parton facts emitted
    by the definition renderer are the disclosure signal."""
    defined = rec.get("defined_entities", [])
    if not defined:
        return []
    raw_ents = rec.get("entities", [])
    if not raw_ents or not isinstance(raw_ents[0], dict):
        return []
    entities = {e["eid"]: e for e in raw_ents}
    parts_by_eid: dict[str, list[str]] = {}
    for sidx, facts in rec.get("sentence_facts", []) or []:
        for f in facts:
            if (f.get("kind") == "relation"
                    and f.get("rel") == "havas_parton"
                    and len(f.get("args", [])) == 2):
                parts_by_eid.setdefault(f["args"][0], []).append(f["args"][1])
    out = []
    for eid in defined:
        if eid not in parts_by_eid:
            continue
        ent = entities.get(eid)
        if ent is None:
            continue
        concept = ent["concept"]
        parts = parts_by_eid[eid]
        if len(parts) == 1:
            parts_str = parts[0]
        elif len(parts) == 2:
            parts_str = f"{parts[0]} kaj {parts[1]}"
        else:
            parts_str = ", ".join(parts[:-1]) + f", kaj {parts[-1]}"
        q = rng.choice([
            f"De kio estas farita la {concept}?",
            f"El kio estas farita la {concept}?",
        ])
        a = rng.choice([
            f"El {parts_str}.",
            f"De {parts_str}.",
            f"La {concept} estas farita el {parts_str}.",
        ])
        out.append({
            "q": q, "a": a,
            "requires": [
                {"kind": "relation", "rel": "havas_parton",
                 "args[0]": eid, "args[1]": p}
                for p in parts
            ],
        })
    return out


def _q_de_owner_from_havi(rec: dict, rng: random.Random) -> list[dict]:
    """De-question for possession: "De kiu estas la X?" / "Kies estas
    la X?" → "De OWNER." Requires a havi relation fact for the entity
    (either explicit "Y havis X" or specifier rendering "la X de Y")."""
    raw_ents = rec.get("entities", [])
    if not raw_ents or not isinstance(raw_ents[0], dict):
        return []
    entities = {e["eid"]: e for e in raw_ents}
    havi_pairs: dict[str, str] = {}
    for sidx, facts in rec.get("sentence_facts", []) or []:
        for f in facts:
            if (f.get("kind") == "relation"
                    and f.get("rel") == "havi"
                    and len(f.get("args", [])) == 2):
                havi_pairs[f["args"][1]] = f["args"][0]
    out = []
    for item_eid, owner_eid in havi_pairs.items():
        item_ent = entities.get(item_eid)
        owner_ent = entities.get(owner_eid)
        if item_ent is None or owner_ent is None:
            continue
        if item_ent.get("type") == "person":
            continue
        if "_" in item_eid and item_eid != item_ent["concept"]:
            continue  # body part
        concept = item_ent["concept"]
        owner_name = _name(owner_eid, entities)
        q = rng.choice([
            f"De kiu estas la {concept}?",
            f"Kies estas la {concept}?",
        ])
        a = rng.choice([
            f"De {owner_name}.",
            f"{owner_name}.",
            f"La {concept} estas de {owner_name}.",
        ])
        out.append({
            "q": q, "a": a,
            "requires": [{
                "kind": "relation", "rel": "havi",
                "args[0]": owner_eid, "args[1]": item_eid,
            }],
        })
    return out


def _q_subject_from_copula(rec: dict, rng: random.Random) -> list[dict]:
    """Inverse of definition: given prose "X estas Y", ask
    "Kio estas Y?" → "X". Trains subject-from-copula extraction
    needed for wiki-style "Parizo estas la ĉefurbo de Francio.
    Kio estas la ĉefurbo de Francio? → Parizo."

    Only fires when the category is uniquely held by one entity in
    the trace (no other entity has the same category fact), so the
    answer is unambiguous."""
    defined = rec.get("defined_entities", [])
    if not defined:
        return []
    raw_ents = rec.get("entities", [])
    if not raw_ents or not isinstance(raw_ents[0], dict):
        return []
    entities = {e["eid"]: e for e in raw_ents}
    lex = _get_lex()
    # Build map: category -> set of eids the category was claimed for
    # by any rendered definition in this trace.
    cat_to_eids: dict[str, set[str]] = {}
    for sidx, facts in rec.get("sentence_facts", []) or []:
        for f in facts:
            if f.get("kind") == "category":
                cat_to_eids.setdefault(f["value"], set()).add(f["entity"])
    out = []
    for eid in defined:
        ent = entities.get(eid)
        if ent is None:
            continue
        if ent.get("type") == "person":
            # Named person: "Petro estas kuracisto" → "Kiu estas kuracisto?" → "Petro"
            cat = ent["concept"]
            if cat not in cat_to_eids or len(cat_to_eids[cat]) != 1:
                continue
            # Skip when the person has no proper name — the eid is
            # just the concept lemma, so "Kiu estas gasto? Gasto." is
            # tautological. Only fire for named individuals (eid
            # distinct from concept lemma).
            if eid == ent["concept"]:
                continue
            name = _name(eid, entities)
            q = rng.choice([
                f"Kiu estas {cat}?",
                f"Kiu estas la {cat}?",
                f"Kiu en la rakonto estas {cat}?",
            ])
            a = rng.choice([
                f"{name}.",
                f"{name} estas {cat}.",
                f"La {cat} estas {name}.",
            ])
            out.append({
                "q": q, "a": a,
                "requires": [{"kind": "category", "entity": eid, "value": cat}],
            })
            continue
        concept_def = lex.concepts.get(ent["concept"])
        if concept_def is None:
            continue
        cats = getattr(concept_def, "category", None) or []
        if not cats:
            continue
        cat = cats[0]
        if cat not in cat_to_eids or len(cat_to_eids[cat]) != 1:
            continue
        # "Tablo estas meblo." → "Kio estas meblo?" → "Tablo"
        concept = ent["concept"]
        q = rng.choice([
            f"Kio estas {cat}?",
            f"Kio estas la {cat} en la rakonto?",
            f"Kiu {cat} estas en la rakonto?",
        ])
        a = rng.choice([
            f"{concept}.",
            f"{concept} estas {cat}.",
            f"La {cat} estas {concept}.",
        ])
        out.append({
            "q": q, "a": a,
            "requires": [{"kind": "category", "entity": eid, "value": cat}],
        })
    return out


# Registry of question generators.
GENERATORS = [
    _q_intrinsic_property,
    _q_first_last,
    _q_action_attribution,
    # _q_state_change — disabled: "Post kiam la aganto V-is la X,
    # kia estis la stato?" → "<state>" conflated with the eval's
    # "Kio okazis post la X-ado?" → next-event extraction. Model
    # learns "post X → state value" and answers state-words to
    # next-event questions. Also uses abstract "la aganto" instead
    # of a named subject — unusual surface form. State-after info
    # is already disclosed via _q_why_property and the disclosure
    # facts in the prose itself.
    # _q_state_change,
    _q_location_at_start,
    _q_instrument_and_parts,
    _q_count,
    _q_event_theme_count,
    _q_count_sum,
    _q_count_delta,
    _q_count_transfer,
    _q_count_chain_funcall,
    _q_multi_lemma_sum_funcall,
    _q_count_before,
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
    _q_why_property,
    _q_location_at_end,
    _q_enablement,
    _q_cloze_event,
    _q_active_passive,
    _q_sentence_construction,
    _q_true_false,
    _q_pronoun_resolution,
    _q_agent_summary,
    _q_negation,
    _q_multiple_choice,
    _q_entity_journey,
    _q_definition,
    _q_subject_from_copula,
    _q_de_agent_from_passive,
    _q_de_parts_from_definition,
    _q_de_owner_from_havi,
    _q_companion,
    _q_intransitive_subject,
    _q_biography,
    # _q_why — skipped: 95% of causal chains are pluvi→_wet,
    # producing "Ĉar pluvis." mode collapse. Needs richer causal
    # annotations in the engine before this template is useful.
]


def _fact_matches(pattern: dict, fact: dict) -> bool:
    """Pattern matches a fact when every key in pattern equals the
    corresponding fact field. `args[N]` keys check positional args."""
    for k, v in pattern.items():
        if k.startswith("args[") and k.endswith("]"):
            idx = int(k[5:-1])
            args = fact.get("args", [])
            if idx >= len(args) or args[idx] != v:
                return False
        elif k.startswith("roles."):
            role = k[len("roles."):]
            roles = dict(fact.get("roles", []))
            if roles.get(role) != v:
                return False
        elif fact.get(k) != v:
            return False
    return True


def _qa_disclosed(requires: list[dict], all_facts: list[dict]) -> bool:
    """A Q/A is disclosed when every required-fact pattern is matched
    by at least one fact in the trace's disclosure log."""
    if not requires:
        return True
    for pat in requires:
        if not any(_fact_matches(pat, f) for f in all_facts):
            return False
    return True


def _flatten_facts(sentence_facts) -> list[dict]:
    out = []
    for entry in sentence_facts or ():
        if isinstance(entry, list) and len(entry) == 2:
            _, facts = entry
        else:
            facts = entry
        for f in facts:
            out.append(f)
    return out


import re as _re

_KIE_PREP_RE = _re.compile(
    r'^\s*(en|sur|sub|apud|ĉe|al|tra)\s+',
    _re.IGNORECASE)


def _count_answer_form(a: str) -> str:
    """Sub-tag count answers by surface form so the picker can balance
    bare-cardinal vs CoT-math vs narrated. Otherwise the bare form
    (which dominates) crowds out the math forms in inverse-frequency
    sampling."""
    if not a:
        return "bare"
    if " minus " in a or " plus " in a or " - " in a or " + " in a:
        return "cot"
    if "Komence" in a:
        return "narr"
    if a.startswith("Sume "):
        return "sume"
    return "bare"


_COLOR_RE = _re.compile(
    r'^Kio\s+(?:estas\s+|estis\s+)?'
    r'(ruĝa|blua|verda|flava|blanka|nigra|bruna|griza|oranĝa)\??$',
    _re.IGNORECASE)
_MOVE_REV_RE = _re.compile(
    r'^Kiu\s+(iris|kuris|venis|eniris|flugis|naĝis)\s+al\b',
    _re.IGNORECASE)
_STATE_REV_RE = _re.compile(
    r'^Kio\s+(?:estas\s+|estis\s+)'
    r'(malfermita|fermita|plena|malplena|ŝlosita|malŝlosita|'
    r'aktiva|neaktiva|pura|malpura)\??$',
    _re.IGNORECASE)


def _qa_type_key(q: str, a: str | None = None) -> str:
    """Classify a question for balancing purposes. When `a` is given:
      - Kie/Kien split by answer preposition (kie:en, kie:sur, ...)
        so each preposition gets its own inverse-frequency bucket.
      - Count Q/A split by answer form (count:bare, count:cot,
        count:narr, count:sume) so math-CoT forms aren't crowded out
        by bare cardinals.
      - Reverse-direction Q/A (Kio estis bruna?, Kiu iris al X?)
        get their own sub-bucket so they aren't drowned by the
        dominant forward forms (`_q_action_attribution`'s many
        'Kiu V-is X?' candidates, etc.).
    """
    if "Kiom da" in q:
        if "restas" in q:
            base = "count_delta"
        elif "havas" in q:
            base = "count_transfer"
        elif "antaŭ" in q:
            base = "count_before"
        else:
            base = "count"
        if a:
            return f"{base}:{_count_answer_form(a)}"
        return base
    # "Por kio X V-is?" is a purpose-asking variant of Kial — tag
    # both under "kial" so the inverse-frequency picker sees them
    # as the same family.
    if q.startswith("Por kio "):
        return "kial"
    # Reverse-direction Q/A — distinct buckets so they don't drown.
    if _COLOR_RE.match(q) or _STATE_REV_RE.match(q):
        return "kio:rev_prop"
    if _MOVE_REV_RE.match(q):
        return "kiu:rev_move"
    # Kio sub-buckets — split by semantic shape so each gets its
    # own inverse-frequency share rather than competing under one
    # umbrella `kio`.
    if q.startswith("Kio okazis unue") or q.startswith("Kio okazis laste"):
        return "kio:first_last"
    if q.startswith("Kio okazis post "):
        return "kio:after"
    if q.startswith("Kio okazis antaŭ "):
        return "kio:before"
    if q.startswith("Kompletigu:"):
        return "cloze"
    if q.startswith("Skribu pasive:"):
        return "transform:passive"
    if q.startswith("Skribu aktive:"):
        return "transform:active"
    if q.startswith("Skribu frazon:"):
        return "construct"
    if q.startswith("Ĉu vere?:"):
        return "verify"
    if q.startswith("En la frazo") and "kiu estas" in q:
        return "pronoun_resolve"
    if q.startswith("Kio okazis al ") or q.startswith("Kio okazis kun "):
        return "kio:cons"
    if q.startswith("Kio estas en la "):
        return "kio:contents"
    if q.startswith("Kio estas 'la "):
        return "kio:alias"
    if q.startswith("Kio estas "):
        return "kio:ident"
    # Prepositional sub-buckets — passive voice and compound-question
    # variants get their own slots so small-volume forms like
    # "Al kiu lando" or "Per kio kaj el kio" don't drown.
    if q.startswith("En kiu loko "):
        return "en:loko"
    if q.startswith("En kio estis la "):
        return "en:contents"
    if q.startswith("En kiu jaro "):
        return "en:jaro"
    if q.startswith("En kia pozicio"):
        return "en:pozicio"
    if q.startswith("De kiu estis "):
        return "de:passive"
    if q.startswith("De kio estas farita") or q.startswith("De kio "):
        return "de:material"
    if q.startswith("De kiu ") and " ricevis " in q:
        return "de:giver"
    if q.startswith("Al kiu lando "):
        return "al:lando"
    if q.startswith("Al kiu "):
        return "al:recipient"
    if q.startswith("Per kio kaj el "):
        return "per:from_what"
    if q.startswith("Per kio "):
        return "per:instrument"
    first = q.split()[0].lower() if q.split() else "?"
    if first in ("kie", "kien") and a:
        m = _KIE_PREP_RE.match(a.lstrip("la ").lstrip())
        if m is None:
            # Try after "la"-stripping fallback: just match anywhere
            m = _KIE_PREP_RE.match(a)
        if m:
            return f"{first}:{m.group(1).lower()}"
    return first


def generate_qas_for_trace(
    rec: dict, rng: random.Random, max_per_trace: int = 4,
    all_concepts: frozenset[str] | None = None,
    type_counts: dict[str, int] | None = None,
    skip_undisclosed: bool = True,
    skip_counter: Optional["Counter[str]"] = None,
) -> list[dict]:
    """Yield up to max_per_trace Q/A pairs sampled across generators.
    Selection weighted by inverse cumulative frequency so
    underrepresented question types get boosted.

    `type_counts`: running counter of emitted Q/A types across all
    traces. Updated in-place by this function. Each emitted Q/A is
    tagged with `disclosed: bool` reflecting whether the trace's
    `sentence_facts` cover the Q/A's required facts.

    `skip_undisclosed`: when True (default), Q/A pairs whose required
    facts aren't disclosed in the trace's prose are dropped from the
    candidate pool. The picker then samples from the remaining
    grounded Q/A. When all candidates of a type are undisclosed, that
    type effectively contributes nothing for this trace.

    `skip_counter`: optional Counter that this function increments
    (per-template) with the number of undisclosed Q/A skipped."""
    is_planned = "drive" in rec
    candidates: list[dict] = []
    for gen in GENERATORS:
        if gen == _q_enablement and not is_planned:
            continue
        if gen == _q_existence:
            candidates.extend(gen(rec, rng, all_concepts=all_concepts))
        else:
            candidates.extend(gen(rec, rng))
    if not candidates:
        return []
    disclosed_facts = _flatten_facts(rec.get("sentence_facts"))
    have_facts = bool(disclosed_facts) or bool(rec.get("sentence_facts"))
    for qa in candidates:
        if have_facts:
            qa["disclosed"] = _qa_disclosed(
                qa.get("requires", []), disclosed_facts)
        else:
            qa["disclosed"] = None  # unknown (legacy record without facts)
    by_type: dict[str, list[dict]] = {}
    for qa in candidates:
        key = _qa_type_key(qa["q"], qa.get("a"))
        by_type.setdefault(key, []).append(qa)
    for qas in by_type.values():
        rng.shuffle(qas)
    seen_qs: set = set()
    picked: list[dict] = []
    while len(picked) < max_per_trace and by_type:
        if type_counts is not None:
            # Square the inverse-frequency: rare types get pulled
            # harder toward the front of the picking queue.
            weights = [1.0 / (1 + type_counts.get(k, 0)) ** 2
                       for k in by_type]
        else:
            weights = [1.0] * len(by_type)
        keys = list(by_type.keys())
        chosen_key = rng.choices(keys, weights=weights, k=1)[0]
        qas = by_type[chosen_key]
        added = False
        while qas:
            qa = qas.pop()
            if qa["q"] in seen_qs:
                continue
            # In-loop disclosure filter: skip undisclosed candidates
            # within this type and try the next one. The chosen type
            # only gets dropped from `by_type` when its bucket runs
            # out of candidates entirely — preserving the
            # inverse-frequency weighting across types regardless of
            # how many of a type's candidates happen to be disclosed.
            if (skip_undisclosed and have_facts
                    and qa["disclosed"] is False):
                if skip_counter is not None:
                    skip_counter[chosen_key] += 1
                continue
            seen_qs.add(qa["q"])
            picked.append(qa)
            if type_counts is not None:
                type_counts[chosen_key] = (
                    type_counts.get(chosen_key, 0) + 1)
            added = True
            break
        if not added or not qas:
            del by_type[chosen_key]
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
    p.add_argument("--max-per-qa", type=int, default=50,
                   help="Cap each unique (question, answer) string at this "
                        "many emitted records. Set 0 to disable. Prevents "
                        "heavy-template patterns (e.g. tablo→bruna, ~3.6k "
                        "hits in a 60k run) from over-imprinting and "
                        "overriding context.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-undisclosed", action="store_true",
                   help="Don't drop Q/A whose required facts weren't "
                        "disclosed in prose. Default is to skip them.")
    args = p.parse_args()
    skip_undisclosed = not args.keep_undisclosed

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
    n_with_requires = 0
    n_disclosed = 0
    n_undisclosed = 0
    n_unknown_legacy = 0
    by_template_disclosed: dict[str, Counter] = {}
    type_counts: dict[str, int] = {}
    skip_counter: Counter = Counter()
    # Per-(question, answer)-string cap. Prevents heavy templates
    # (e.g. "Kia estis la koloro de la tablo?" → "bruna" with 3,605
    # hits in a 60k run) from over-imprinting and overriding context.
    # Set --max-per-qa 0 to disable.
    qa_string_counts: Counter = Counter()
    n_capped = 0
    max_per_qa = args.max_per_qa
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
                all_concepts=all_concepts_frozen,
                type_counts=type_counts,
                skip_undisclosed=skip_undisclosed,
                skip_counter=skip_counter)
            for qa in qas:
                qa_key = (qa["q"], qa["a"])
                if max_per_qa and qa_string_counts[qa_key] >= max_per_qa:
                    n_capped += 1
                    continue
                qa_string_counts[qa_key] += 1
                fout.write(json.dumps(
                    format_sft_record(prose, qa),
                    ensure_ascii=False) + "\n")
                n_qas += 1
                tpl = _qa_type_key(qa["q"], qa.get("a"))
                bucket = by_template_disclosed.setdefault(tpl, Counter())
                bucket["total"] += 1
                if qa.get("requires"):
                    n_with_requires += 1
                    bucket["tagged"] += 1
                if qa.get("disclosed") is True:
                    n_disclosed += 1
                    bucket["disclosed"] += 1
                elif qa.get("disclosed") is False:
                    n_undisclosed += 1
                    bucket["undisclosed"] += 1
                else:
                    n_unknown_legacy += 1
            n_traces += 1
    print(f"Wrote {n_qas} Q/A pairs from {n_traces} traces to {args.out}")
    if max_per_qa:
        print(f"  Capped at max-per-qa={max_per_qa}: dropped {n_capped} "
              f"records ({100*n_capped/max(n_qas+n_capped,1):.1f}%)")
    print(f"  Tagged with requires:  {n_with_requires}")
    print(f"  Disclosed:             {n_disclosed} "
          f"({100*n_disclosed/max(n_qas,1):.1f}%)")
    print(f"  Undisclosed:           {n_undisclosed} "
          f"({100*n_undisclosed/max(n_qas,1):.1f}%)")
    print(f"  Unknown (no facts):    {n_unknown_legacy}")
    skipped_total = sum(skip_counter.values())
    if skip_undisclosed and skipped_total:
        print(f"  Skipped (undisclosed): {skipped_total}")
        for tpl in sorted(skip_counter, key=lambda k: -skip_counter[k]):
            print(f"    {tpl:<22s} {skip_counter[tpl]}")
    if any(by_template_disclosed.values()):
        print()
        print(f"{'Template':<22s} {'Total':>7s} {'Tagged':>7s} "
              f"{'Disc':>7s} {'Undisc':>7s} {'%Undisc':>8s}")
        print("-" * 65)
        for tpl in sorted(by_template_disclosed,
                          key=lambda k: -by_template_disclosed[k]["total"]):
            c = by_template_disclosed[tpl]
            tagged = c.get("tagged", 0)
            disc = c.get("disclosed", 0)
            undisc = c.get("undisclosed", 0)
            pct = 100 * undisc / max(disc + undisc, 1) if (disc + undisc) else 0
            print(f"{tpl:<22s} {c['total']:>7d} {tagged:>7d} "
                  f"{disc:>7d} {undisc:>7d} {pct:>7.1f}%")


if __name__ == "__main__":
    main()
