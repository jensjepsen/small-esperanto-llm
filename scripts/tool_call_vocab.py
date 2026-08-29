"""Danish tool-call schema vocabulary.

The model sees Danish keys throughout (navn, argumenter, parametre,
påkrævet, type: tekst/heltal/...). For server-side validation with the
`jsonschema` library we translate to standard JSON Schema keys.

The model NEVER sees the English translation — this is purely a
verifier bridge.
"""
from __future__ import annotations

from typing import Any


# ── Schema key mapping (Danish → JSON Schema) ────────────────────────────

# Top-level tool-call fields (assistant output).
CALL_NAME_KEY = "navn"           # → "name"
CALL_ARGS_KEY = "argumenter"     # → "arguments"

# Tool schema fields.
SCHEMA_KEYS = {
    "navn": "name",
    "beskrivelse": "description",
    "parametre": "parameters",
    "påkrævet": "required",       # per-param bool → aggregated into JSON Schema `required` list
    "type": "type",
    "min": "minimum",
    "maks": "maximum",
    "valg": "enum",
    "mønster": "pattern",
    "format": "format",
    "længde_min": "minLength",
    "længde_maks": "maxLength",
    "elementer": "items",         # array element schema
    "egenskaber": "properties",   # nested object properties
}

# Type value mapping.
TYPES = {
    "tekst": "string",
    "heltal": "integer",
    "tal": "number",
    "boolsk": "boolean",
    "liste": "array",
    "objekt": "object",
    # Enum sugar — treat as string, `valg` decorator carries the choices.
    "enum_valg": "string",
    # Shorthand array-of-X (expanded to array+items).
    "liste_af_tekst": ("array", "string"),
    "liste_af_heltal": ("array", "integer"),
    "liste_af_tal": ("array", "number"),
    "liste_af_objekt": ("array", "object"),
}


# ── Danish-schema → JSON-Schema translator ───────────────────────────────

def _translate_param(param_da: dict) -> tuple[dict, bool]:
    """Translate a single Danish param spec to a JSON Schema fragment.

    Returns (json_schema_fragment, is_required).
    """
    js: dict[str, Any] = {}
    is_required = bool(param_da.get("påkrævet", False))

    # Type — handle shorthand liste_af_* which becomes array+items.
    t_da = param_da.get("type", "tekst")
    if t_da in TYPES:
        t = TYPES[t_da]
        if isinstance(t, tuple):
            js["type"] = t[0]
            js["items"] = {"type": t[1]}
        else:
            js["type"] = t
    else:
        # Unknown type — leave as-is; jsonschema will reject if it matters.
        js["type"] = t_da

    # Description passthrough.
    if "beskrivelse" in param_da:
        js["description"] = param_da["beskrivelse"]

    # Numeric bounds.
    if "min" in param_da:
        js["minimum"] = param_da["min"]
    if "maks" in param_da:
        js["maximum"] = param_da["maks"]

    # String constraints.
    if "længde_min" in param_da:
        js["minLength"] = param_da["længde_min"]
    if "længde_maks" in param_da:
        js["maxLength"] = param_da["længde_maks"]
    if "mønster" in param_da:
        js["pattern"] = param_da["mønster"]
    if "format" in param_da:
        js["format"] = param_da["format"]

    # Enum.
    if "valg" in param_da:
        js["enum"] = param_da["valg"]

    # Array item schema (explicit — overrides shorthand from liste_af_*).
    if "elementer" in param_da:
        el, _ = _translate_param(param_da["elementer"])
        js["items"] = el

    # Nested object properties.
    if "egenskaber" in param_da:
        props = {}
        req = []
        for pname, pspec in param_da["egenskaber"].items():
            sub, r = _translate_param(pspec)
            props[pname] = sub
            if r:
                req.append(pname)
        js["type"] = "object"
        js["properties"] = props
        if req:
            js["required"] = req
        js["additionalProperties"] = False

    return js, is_required


def tool_schema_to_json_schema(tool_da: dict) -> dict:
    """Convert a Danish tool schema to a JSON Schema for the `argumenter` payload.

    Given:
        {"navn": "book_bord",
         "beskrivelse": "...",
         "parametre": {
             "tidspunkt": {"type": "tekst", "påkrævet": True, ...},
             ...
         }}

    Returns a JSON Schema validating the `argumenter` dict.
    """
    props: dict[str, dict] = {}
    required: list[str] = []
    for pname, pspec in tool_da.get("parametre", {}).items():
        sub, is_req = _translate_param(pspec)
        props[pname] = sub
        if is_req:
            required.append(pname)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": props,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def validate_call(call: dict, catalog: list[dict]) -> tuple[bool, str]:
    """Validate a model's tool-call output against the catalog.

    Returns (ok, error_reason). `call` shape: {"navn": ..., "argumenter": {...}}.
    """
    from jsonschema import Draft202012Validator, ValidationError

    if not isinstance(call, dict):
        return False, "call is not a dict"
    if CALL_NAME_KEY not in call:
        return False, f"missing '{CALL_NAME_KEY}'"
    if CALL_ARGS_KEY not in call:
        return False, f"missing '{CALL_ARGS_KEY}'"

    name = call[CALL_NAME_KEY]
    args = call[CALL_ARGS_KEY]

    tool = next((t for t in catalog if t.get("navn") == name), None)
    if tool is None:
        return False, f"function '{name}' not in catalog"

    try:
        js = tool_schema_to_json_schema(tool)
        v = Draft202012Validator(js)
        errs = sorted(v.iter_errors(args), key=lambda e: e.path)
    except Exception as e:
        # Malformed tool schema (unknown type, bad regex, invalid enum, ...).
        return False, f"tool schema unusable: {type(e).__name__}: {str(e)[:120]}"
    if errs:
        e = errs[0]
        path = ".".join(str(p) for p in e.path) or "<root>"
        return False, f"schema violation at {path}: {e.message}"
    return True, ""


# ── Round-trip self-check ────────────────────────────────────────────────

if __name__ == "__main__":
    tool = {
        "navn": "book_bord",
        "beskrivelse": "Reserver et bord",
        "parametre": {
            "tidspunkt": {"type": "tekst", "påkrævet": True,
                          "beskrivelse": "Formatet TT:MM",
                          "mønster": r"^\d{2}:\d{2}$"},
            "antal_gæster": {"type": "heltal", "påkrævet": True,
                             "min": 1, "maks": 20},
            "gæstenavn": {"type": "tekst", "påkrævet": True,
                          "længde_min": 1, "længde_maks": 100},
            "notat": {"type": "tekst", "påkrævet": False},
            "allergier": {"type": "liste_af_tekst", "påkrævet": False},
        },
    }
    catalog = [tool]

    good = {"navn": "book_bord",
            "argumenter": {"tidspunkt": "19:00", "antal_gæster": 4,
                           "gæstenavn": "Anna"}}
    bad_missing = {"navn": "book_bord",
                   "argumenter": {"tidspunkt": "19:00"}}
    bad_type = {"navn": "book_bord",
                "argumenter": {"tidspunkt": "19:00", "antal_gæster": "fire",
                               "gæstenavn": "Anna"}}
    bad_pattern = {"navn": "book_bord",
                   "argumenter": {"tidspunkt": "syv", "antal_gæster": 4,
                                  "gæstenavn": "Anna"}}
    bad_enum = {"navn": "not_a_tool",
                "argumenter": {}}

    for label, call in [("good", good), ("missing", bad_missing),
                        ("bad_type", bad_type), ("bad_pattern", bad_pattern),
                        ("not_in_catalog", bad_enum)]:
        ok, err = validate_call(call, catalog)
        print(f"  {label:16s} ok={ok}  err={err!r}")
