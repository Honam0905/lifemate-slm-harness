"""Tool specs for the prompt: the split catalog's own schemas, minus bookkeeping fields (no meaning dropped)."""
from __future__ import annotations

KEEP_TOOL = ("name", "kind", "description")


def compact_tool(spec: dict) -> dict:
    out = {k: spec[k] for k in KEEP_TOOL if spec.get(k)}
    out["parameters"] = spec["parameters"]          # already compact: type/properties/required/x-one-of/enum/format/x-ref/x-match
    if spec.get("returns"):
        out["returns"] = spec["returns"]
    return out


def tool_specs(tool_names: list[str], index: dict[str, dict]) -> list[dict]:
    """Compact specs for the visible tools, in the given order. Unknown names are an error, never silently dropped."""
    missing = [t for t in tool_names if t not in index]
    if missing:
        raise KeyError(f"tools not in catalog: {missing}")
    return [compact_tool(index[t]) for t in tool_names]


def param_types(spec: dict) -> dict[str, str | None]:
    """top-level argument name -> JSON-schema type"""
    return {k: p.get("type") for k, p in spec["parameters"].get("properties", {}).items()}
