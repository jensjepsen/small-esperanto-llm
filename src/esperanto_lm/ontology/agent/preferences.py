"""Slot-level preferences. Drives the agent's autonomous goal
generation (`displeased_slots(entity)` returns slots whose current
value isn't the preferred one — the basis for self-slot drives).

`SLOT_PREFERENCES` is the static baseline. `effective_preferences`
overlays trace-wide context on top — currently the only override is
`sleep_state=dormanta` when the world's `tempo_de_tago=nokto`, since
animates prefer sleep at night. Adding more conditional prefs (e.g.
posture preferences shifting with weather) is the same pattern: read
the relevant `mondo` slot and rewrite an entry."""
from __future__ import annotations

from typing import Optional


SLOT_PREFERENCES: dict[str, str] = {
    "hunger":      "sata",
    "thirst":      "satigita",
    "sleep_state": "vekita",
    "wetness":     "seka",
    "cleanliness": "pura",
}


def effective_preferences(trace=None) -> dict[str, str]:
    """Return `SLOT_PREFERENCES` adjusted for trace-wide context.
    Reads the `mondo` singleton's slots; falls back to the static
    baseline when no trace or no mondo entity is present.

    Current overrides:
      `sleep_state` flips to `dormanta` when `mondo.tempo_de_tago=nokto`.

    Add a new override by reading another `mondo` slot here. Keep
    the structure flat — nested context conditions belong in
    documentation, not in nested if-trees."""
    prefs = dict(SLOT_PREFERENCES)
    if trace is None:
        return prefs
    mondo = trace.entities.get("mondo")
    if mondo is None:
        return prefs
    tdt = mondo.properties.get("tempo_de_tago")
    if isinstance(tdt, list):
        tdt = tdt[0] if tdt else None
    if tdt == "nokto":
        prefs["sleep_state"] = "dormanta"
    return prefs
