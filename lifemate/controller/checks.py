"""Argument hygiene before every call, x-ref provenance logging, and the loop guard."""
from __future__ import annotations

import ast
import json
import re
import unicodedata

EMPTY = (None, "", [], {})
_DT_MIN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
_DT_SEC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


def _nfc(v):
    if isinstance(v, str):
        return unicodedata.normalize("NFC", v)
    if isinstance(v, list):
        return [_nfc(x) for x in v]
    if isinstance(v, dict):
        return {k: _nfc(x) for k, x in v.items()}
    return v


_INT = re.compile(r"[+-]?\d+")
_INT_GROUPED = re.compile(r"\d{1,3}(?:[.,]\d{3})+")      # 845.000 / 2,800,000 (thousands separators)


def coerce(v, prop: dict):
    """Repair a value to the schema type without changing its meaning; anything ambiguous is returned unchanged."""
    t = prop.get("type")
    if isinstance(v, str):
        s = v.strip()
        if t == "integer":
            if _INT.fullmatch(s):
                return int(s)
            if _INT_GROUPED.fullmatch(s):
                return int(re.sub(r"[.,]", "", s))
        elif t == "number":
            if _INT.fullmatch(s):
                return int(s)
            try:
                return float(s)
            except ValueError:
                return v
        elif t == "boolean" and s.lower() in ("true", "false"):
            return s.lower() == "true"
        elif t in ("array", "object"):
            for load in (json.loads, ast.literal_eval):      # JSON, then Python-literal lists/dicts ("['a']")
                try:
                    x = load(s)
                    break
                except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                    x = None
            if (t == "array" and isinstance(x, list)) or (t == "object" and isinstance(x, dict)):
                return x
            if t == "array" and x is None and s and not s.startswith(("[", "{")):
                return [s]                                      # a bare item where a list is expected
    elif t == "integer" and isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def clean_args(args: dict, spec: dict, tz_offset: str = "+07:00", coerce_types: bool = False) -> tuple[dict, list[str]]:
    """Drop undeclared keys and empty optional values, NFC-normalise strings, write date-times in the gold style
    (with seconds and offset). Fuzzy free text is left as generated: 100 of the 740 gold fuzzy values end with ".",
    so stripping punctuation would not move us towards the annotators' style."""
    params = spec["parameters"]
    props, required = params.get("properties", {}), set(params.get("required", []))
    out, notes = {}, []
    for k, v in args.items():
        if k not in props:
            notes.append(f"dropped undeclared key {k!r}")
            continue
        if v in EMPTY and k not in required:
            notes.append(f"dropped empty optional {k!r}")
            continue
        v = _nfc(v)
        p = props[k]
        if coerce_types:
            c = coerce(v, p)
            if c != v or type(c) is not type(v):
                notes.append(f"coerced {k!r}: {v!r} -> {c!r}")
                v = c
        # gold writes every datetime with seconds and offset, also in fields without a declared format
        if isinstance(v, str) and p.get("format") in (None, "date-time"):
            if _DT_MIN.match(v):
                v += ":00" + tz_offset
            elif _DT_SEC.match(v):
                v += tz_offset
        out[k] = v
    return out, notes


def unsourced_refs(args: dict, spec: dict, visible_text: str) -> list[str]:
    """x-ref argument values that never appeared in anything the model has seen (logged, not blocked, in v1)."""
    props = spec["parameters"].get("properties", {})
    bad = []
    for k, v in args.items():
        if props.get(k, {}).get("x-ref") and isinstance(v, str) and v not in visible_text:
            bad.append(f"{k}={v}")
    return bad


def same_call(a: dict | None, b: dict | None) -> bool:
    return a is not None and b is not None and a["name"] == b["name"] and a["arguments"] == b["arguments"]
