"""System + context + full history + current request -> chat messages (shared by training and inference)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from .actions import call_message, stop_message, tool_message

SYSTEM = """You are LifeMate, a voice assistant that carries out a Vietnamese user's spoken request by calling tools.
Rules:
- Make exactly one tool call per reply; its result comes back before your next reply.
- Use only the tools listed below. IDs must come from tool results or from the conversation history; never invent them.
- Make only the calls the request needs: the reads that give you the IDs or facts you need, then the writes. Do not repeat a call whose result is already in the history.
- If a required parameter cannot be determined (an ambiguous person or item, or a value the user never said), call sys.ask_missing_info with the tool name and the missing required parameter names.
- If no listed tool can handle the request, or there is nothing to do, reply DONE without calling any tool.
- When the request has been fully handled, reply DONE.
- Write datetimes as ISO 8601 with offset (2026-11-29T09:00:00+07:00), dates as YYYY-MM-DD, money as integer VND."""

WEEKDAYS = [("Monday", "Thứ Hai"), ("Tuesday", "Thứ Ba"), ("Wednesday", "Thứ Tư"), ("Thursday", "Thứ Năm"),
            ("Friday", "Thứ Sáu"), ("Saturday", "Thứ Bảy"), ("Sunday", "Chủ Nhật")]


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def calendar_lines(current_time: str) -> list[str]:
    """Dates the user is likely to mean, computed here so the model copies them instead of doing date arithmetic
    (covers 565 of the 603 dates in train gold arguments)."""
    today = datetime.fromisoformat(current_time).date()
    monday = today - timedelta(days=today.weekday())

    def d(x) -> str:
        return f"{WEEKDAYS[x.weekday()][1]} {x.isoformat()}"

    return [f"Calendar: today {d(today)}; tomorrow (mai) {d(today + timedelta(1))}; "
            f"day after tomorrow (ngày kia/ngày mốt) {d(today + timedelta(2))}",
            "This week: " + ", ".join(d(monday + timedelta(i)) for i in range(7)),
            "Next week: " + ", ".join(d(monday + timedelta(7 + i)) for i in range(7))]


def user_message(context: dict, history: list[dict], request: str, date_table: bool = False) -> dict:
    ct = context["current_time"]
    en, vi = WEEKDAYS[datetime.fromisoformat(ct).weekday()]
    loc = context.get("location") or {}
    lines = ["## Context",
             f"Current time: {ct} ({en}, {vi})",
             *(calendar_lines(ct) if date_table else []),
             f"Timezone: {context['timezone']}",
             f"Location: lat {loc.get('lat')}, lng {loc.get('lng')}",
             f"User profile: {_j(context['user_profile'])}",
             "",
             "## Conversation history (already handled; its calls were already executed, never repeat them)"]
    if not history:
        lines.append("(none)")
    for t in history:
        if t["role"] == "user":
            lines.append(f"User: {t['text']}")
            continue
        lines.append(f"Assistant: {t.get('text', '')}")
        results = t.get("tool_results") or []
        for i, c in enumerate(t.get("tool_calls") or []):
            r = f" -> {_j(results[i])}" if i < len(results) else ""
            lines.append(f"  [executed] {c['name']} {_j(c.get('arguments') or {})}{r}")
    lines += ["", "## Current request", f"User: {request}"]
    return {"role": "user", "content": "\n".join(lines)}


def prefix_messages(context: dict, history: list[dict], request: str, opts: dict | None = None) -> list[dict]:
    """`opts` is the config's `prompt` section; training and inference must pass the same one."""
    o = opts or {}
    return [{"role": "system", "content": SYSTEM}, user_message(context, history, request, o.get("date_table", False))]


def trajectory_messages(traj: dict, fmt: str, opts: dict | None = None) -> list[dict]:
    """Full training conversation: prefix, then (call, result) per step, then STOP unless the episode ends on an ask."""
    msgs = prefix_messages(traj["context"], traj["history"], traj["request"], opts)
    for i, st in enumerate(traj["steps"]):
        msgs.append(call_message(st["call"], fmt))
        if traj["stop"] or i < len(traj["steps"]) - 1:  # nothing follows a final ask, so its result is not shown
            msgs.append(tool_message(st["result"]))
    if traj["stop"]:
        msgs.append(stop_message())
    return msgs


def render_generation_prompt(tok, msgs: list[dict], tools: list[dict]) -> str:
    """The exact prompt string the model continues from (same call the training collator uses for its prefixes)."""
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
