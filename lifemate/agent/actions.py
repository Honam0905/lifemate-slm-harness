"""Render / parse one agent action (a tool call or STOP) in the model's own chat-template tool-call format.

Formats:
  minicpm_xml  MiniCPM5: <function name="app.tool"><param name="k">v</param></function>
               The template prints values with Jinja, so non-string values are passed in as JSON text and parsed back
               with the tool schema's type (string params stay raw; everything else is json.loads).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

STOP_TEXT = "DONE"


@dataclass
class Action:
    kind: str                    # "call" | "stop" | "invalid"
    name: str | None = None
    arguments: dict = field(default_factory=dict)
    raw: str = ""
    note: str = ""


def _to_text(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def call_message(call: dict, fmt: str) -> dict:
    """Assistant chat message carrying one tool call, ready for tokenizer.apply_chat_template."""
    if fmt == "minicpm_xml":
        args = {k: _to_text(v) for k, v in (call.get("arguments") or {}).items()}
        return {"role": "assistant", "content": "", "tool_calls": [{"name": call["name"], "arguments": args}]}
    raise ValueError(f"unknown tool format {fmt!r}")


def stop_message() -> dict:
    return {"role": "assistant", "content": STOP_TEXT}


def tool_message(result: dict) -> dict:
    return {"role": "tool", "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))}


_FUNC = re.compile(r'<function name="([^"]+)">(.*?)</function>', re.S)
_PARAM = re.compile(r'<param name="([^"]+)">(?:<!\[CDATA\[(.*?)\]\]>|(.*?))</param>', re.S)


def _from_text(text: str, typ: str | None):
    if typ == "string":
        return text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def parse_action(text: str, fmt: str, types_of: dict[str, dict[str, str | None]]) -> Action:
    """Parse generated text (special tokens stripped). `types_of`: tool name -> {param: schema type}."""
    t = text.strip()
    if fmt != "minicpm_xml":
        raise ValueError(f"unknown tool format {fmt!r}")
    m = _FUNC.search(t)
    if not m:
        if t == STOP_TEXT:
            return Action("stop", raw=text)
        if "<function" not in t:  # a plain-text answer ends the turn just like DONE (the untuned model does this)
            return Action("stop", raw=text, note="text reply instead of DONE")
        return Action("invalid", raw=text, note="malformed <function> markup")
    name = m.group(1).strip()
    types = types_of.get(name, {})
    args = {}
    for pm in _PARAM.finditer(m.group(2)):
        k = pm.group(1)
        v = pm.group(2) if pm.group(2) is not None else pm.group(3)
        args[k] = _from_text(v, types.get(k, "string"))
    note = "extra text or calls ignored" if len(_FUNC.findall(t)) > 1 or t[: m.start()].strip() else ""
    return Action("call", name=name, arguments=args, raw=text, note=note)
