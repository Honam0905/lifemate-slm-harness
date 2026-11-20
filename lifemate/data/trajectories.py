"""Causal trajectories from gold final turns, replayed in the train ToolEnv.

A trajectory is model-agnostic: context, visible tools, history, the current request and the gold steps with their
*replayed* results. agent/prompt.py turns it into chat messages; the loss covers only the assistant actions.
"""
from __future__ import annotations

from pathlib import Path

ASK = "sys.ask_missing_info"


def history_view(sample: dict, transcripts: dict[str, str]) -> list[dict]:
    """History turns with user audio replaced by a transcript from `transcripts` (audio path -> text)."""
    out = []
    for t in sample["history"]:
        if t["role"] == "user":
            out.append({"role": "user", "text": transcripts[t["audio"]]})
        else:
            turn = {"role": "assistant", "text": t.get("text", "")}
            if t.get("tool_calls"):
                turn["tool_calls"] = t["tool_calls"]
                if t.get("tool_results") is not None:
                    turn["tool_results"] = t["tool_results"]
            out.append(turn)
    return out


def visible_tools(sample: dict, hidden_apps: set[str] = frozenset()) -> list[str]:
    tools = [t for app, v in sample["context"]["installed_apps"].items() if app not in hidden_apps for t in v["tools"]]
    return tools + [ASK]


def context_view(sample: dict) -> dict:
    c = sample["context"]
    return {k: c[k] for k in ("current_time", "timezone", "location", "user_profile")}


def build_trajectory(sample: dict, zt, root: Path, transcripts: dict[str, str], variant: str,
                     hidden_apps: set[str] = frozenset()) -> tuple[dict, list[str]]:
    """Replay the gold calls in a fresh train ToolEnv. Returns the trajectory and a list of problems found."""
    problems = []
    env = zt.ToolEnv("train", sample["id"], root=root)
    calls = sample["label"]["tool_calls"]
    steps = []
    for i, call in enumerate(calls):
        res = env.call(call["name"], call["arguments"])
        if not res.get("ok"):
            problems.append(f"{sample['id']}: gold call {i} rejected: {res}")
        if res != sample["label"]["tool_results"][i]:
            problems.append(f"{sample['id']}: replayed result {i} differs from label.tool_results")
        steps.append({"call": {"name": call["name"], "arguments": call["arguments"]}, "result": res})
    if env.get_tool_calls() != calls:
        problems.append(f"{sample['id']}: env.get_tool_calls() != gold")
    used = {c["name"].split(".")[0] for c in calls} | {str(c["arguments"].get("tool_name", "")).split(".")[0]
                                                       for c in calls if c["name"] == ASK}
    if used & hidden_apps:
        problems.append(f"{sample['id']}: gold uses hidden app(s) {sorted(used & hidden_apps)}")
    traj = {
        "id": sample["id"],
        "variant": variant,
        "context": context_view(sample),
        "tools": visible_tools(sample, hidden_apps),
        "history": history_view(sample, transcripts),
        "request": transcripts[sample["audio"]],
        "steps": steps,
        "stop": not (calls and calls[-1]["name"] == ASK),  # the controller stops right after an ask
        "state_hash": env.get_state_hash(),
    }
    return traj, problems
