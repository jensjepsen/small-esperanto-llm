"""Static slot-level preferences. Drives the agent's autonomous goal
generation (`displeased_slots(entity)` returns slots whose current
value isn't the preferred one — the basis for self-slot drives).

Preferences are universal in this version. The architecture supports
a conditional version (e.g. "people prefer dormanta when scene
time_of_day=nokto") with the same downstream API; the dict literal
just becomes a function `preference_for(slot, ctx)` returning the
right default for the situation. We've kept the static version while
the rest of the planner is being shaken out."""
from __future__ import annotations


SLOT_PREFERENCES = {
    "hunger":      "sata",
    "thirst":      "satigita",
    "sleep_state": "vekita",
    "wetness":     "seka",
    "cleanliness": "pura",
}
