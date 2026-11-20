"""The agent loop: one ToolEnv per sample, created once and never reset.

Per step: render the prompt -> generate greedily -> parse -> stop on DONE / unparseable / tool not allowed ->
clean the arguments -> loop guard -> env.validate (stop without calling if it fails) -> env.call -> append the result.
Stop right after sys.ask_missing_info, at the step cap, at the context budget, or when the wall-clock budget is spent.
Only validate, call, get_tool_calls and get_state_hash are used on the env (competition rule).

Episodes advance in lock-step rounds so evaluation can batch generation across samples; inference runs batches of one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..agent.actions import call_message, parse_action, tool_message
from ..agent.constrain import RowSpec, collect_values
from ..agent.prompt import prefix_messages, render_generation_prompt
from ..agent.schema import param_types, tool_specs
from ..config import CTX_MAX
from ..data.trajectories import context_view, history_view, visible_tools
from .checks import clean_args, same_call, unsourced_refs

ASK_TOOL = "sys.ask_missing_info"
MAX_WRITE_REPEATS = 2      # an identical write may be issued at most 3 times in a row


@dataclass
class Episode:
    id: str
    env: object
    msgs: list
    tools: list
    specs: dict
    types_of: dict
    allowed: set
    t0: float = field(default_factory=time.time)
    done: bool = False
    stop_reason: str = ""
    steps: list = field(default_factory=list)       # per generation: raw output, parsed action, notes
    last_call: dict | None = None
    repeats: int = 0                                 # consecutive repeats of last_call
    prompt_tokens_max: int = 0
    coerce_types: bool = False
    constrained: bool = False
    seen: set = field(default_factory=set)            # ID-like values visible so far (history + results)


def make_episode(sample: dict, zt, split: str, root, index: dict, transcripts: dict[str, str],
                 hidden_apps: set[str] = frozenset(), cfg: dict | None = None) -> Episode:
    """`cfg` supplies the prompt and harness options (None = the v1 defaults)."""
    cfg = cfg or {}
    env = zt.ToolEnv(split, sample["id"], root=root)
    names = visible_tools(sample, hidden_apps)
    specs = {n: index[n] for n in names}
    msgs = prefix_messages(context_view(sample), history_view(sample, transcripts), transcripts[sample["audio"]],
                           cfg.get("prompt"))
    seen: set = set()
    for t in sample["history"]:
        collect_values(t.get("tool_results") or [], seen)
        collect_values([c.get("arguments") for c in t.get("tool_calls") or []], seen)
    h = cfg.get("harness", {})
    return Episode(id=sample["id"], env=env, msgs=msgs, tools=tool_specs(names, index), specs=specs,
                   types_of={n: param_types(s) for n, s in specs.items()}, allowed=set(names),
                   coerce_types=bool(h.get("coerce_types", False)), constrained=bool(h.get("constrained", False)),
                   seen=seen)


def _finish(ep: Episode, reason: str) -> None:
    ep.done, ep.stop_reason = True, reason


def apply_output(ep: Episode, text: str, fmt: str, max_calls: int) -> None:
    act = parse_action(text, fmt, ep.types_of)
    rec = {"raw": text, "kind": act.kind, "notes": [act.note] if act.note else []}
    ep.steps.append(rec)
    if act.kind == "stop":
        return _finish(ep, "stop_text" if act.note else "stop")
    if act.kind != "call":
        return _finish(ep, "unparseable")
    if act.name not in ep.allowed:
        rec["notes"].append(f"tool not allowed: {act.name}")
        return _finish(ep, "not_allowed")
    args, notes = clean_args(act.arguments, ep.specs[act.name], coerce_types=ep.coerce_types)
    rec["notes"] += notes
    call = {"name": act.name, "arguments": args}
    rec["call"] = call
    # loop guard: re-reading the same thing adds nothing, but a repeated write is a real action (gold redeems the same
    # reward twice in train_0922/1447), so writes may repeat once more before the guard trips
    if same_call(call, ep.last_call):
        if ep.specs[act.name].get("kind") != "write" or ep.repeats >= MAX_WRITE_REPEATS:
            return _finish(ep, "loop")
    seen = "\n".join(m.get("content") or "" for m in ep.msgs)
    unsourced = unsourced_refs(args, ep.specs[act.name], seen)
    if unsourced:
        rec["notes"].append(f"unsourced x-ref: {unsourced}")
    v = ep.env.validate(act.name, args)
    if not v.get("ok"):
        rec["notes"].append(f"validate: {v.get('error')}")
        return _finish(ep, "invalid")
    res = ep.env.call(act.name, args)
    rec["result_ok"] = bool(res.get("ok"))
    collect_values(res, ep.seen)
    ep.repeats = ep.repeats + 1 if same_call(call, ep.last_call) else 0
    ep.last_call = call
    ep.msgs.append(call_message(call, fmt))
    ep.msgs.append(tool_message(res))
    if act.name == ASK_TOOL:
        return _finish(ep, "ask")
    if len(ep.env.get_tool_calls()) >= max_calls:
        return _finish(ep, "max_calls")


def run_episodes(episodes: list[Episode], engine, fmt: str, max_calls: int = 16, budget_s: float | None = None,
                 log=None) -> None:
    """Advance all episodes to completion, generating for every active episode in one batched call per round."""
    rnd = 0
    while True:
        active = [e for e in episodes if not e.done]
        if not active:
            return
        if budget_s is not None:
            for e in active:
                if time.time() - e.t0 > budget_s:
                    _finish(e, "time_budget")
            active = [e for e in active if not e.done]
        prompts, live = [], []
        for e in active:
            p = render_generation_prompt(engine.tok, e.msgs, e.tools)
            n = engine.count_tokens(p)
            e.prompt_tokens_max = max(e.prompt_tokens_max, n)
            if n + engine.max_new_tokens > CTX_MAX:
                _finish(e, "ctx_budget")
                continue
            prompts.append(p); live.append(e)
        if not live:
            continue
        rows = [RowSpec(e.specs, e.seen) for e in live] if any(e.constrained for e in live) else None
        outs = engine.generate(prompts, keys=[e.id for e in live], rows=rows)
        for e, o in zip(live, outs):
            apply_output(e, o, fmt, max_calls)
        rnd += 1
        if log:
            log(f"round {rnd}: {len(live)} generated, {sum(not e.done for e in episodes)} still active")


def submission_line(ep: Episode) -> dict:
    return {"id": ep.id, "tool_calls": ep.env.get_tool_calls(), "state_hash": ep.env.get_state_hash()}
