"""Renaming augmentation: show the same tools under other names so the agent learns to read schemas, not to memorise
tool names (the public test has 11 apps that never occur in training).

A trajectory is renamed consistently everywhere the model can see a name: the tool specs in the prompt (tool names,
descriptions, parameter names and descriptions, `required`, `x-one-of`, references like "lấy từ contacts.search"), the
target calls (names, argument keys, the ask tool's `tool_name` and `params`) and the history's executed calls.
Tool results, the user's words, enums, x-ref values and the ask tool itself are never changed. The alternative names and
descriptions come from synth/rename/out/rename_part*.json (checked by synth/rename/check_rename.py).
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from pathlib import Path

from ..agent.schema import compact_tool

ASK = "sys.ask_missing_info"
REF = re.compile(r"\b[a-z_0-9]+\.[a-z_0-9]+\b")


def load_maps(paths: list[str | Path]) -> dict:
    maps = {}
    for p in paths:
        maps.update(json.loads(Path(p).read_text(encoding="utf-8"))["apps"])
    return maps


def seeded(*parts) -> random.Random:
    return random.Random(int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


class Renamer:
    def __init__(self, maps: dict, index: dict, app_prob: float = 0.8, param_prob: float = 0.5, desc_prob: float = 0.5):
        self.maps, self.index = maps, index
        self.app_prob, self.param_prob, self.desc_prob = app_prob, param_prob, desc_prob
        self.tools_of = {}
        for name in index:
            self.tools_of.setdefault(name.split(".")[0], []).append(name)

    # ---- one random plan per trajectory ------------------------------------------------------------------------
    def plan(self, tools: list[str], rng: random.Random) -> dict:
        tool, param, tdesc, pdesc = {}, {}, {}, {}
        for app in sorted({t.split(".")[0] for t in tools if t != ASK}):
            m = self.maps.get(app)
            if not m or rng.random() >= self.app_prob:
                continue
            new_app = rng.choice(m["alt_ids"])
            for full in sorted(self.tools_of.get(app, [])):        # every tool of the app: cross-references stay valid
                mt = m["tools"].get(full)
                if not mt:
                    continue
                tool[full] = f"{new_app}.{rng.choice(mt['alt_names'])}"
                if rng.random() < self.desc_prob and mt.get("alt_descriptions"):
                    tdesc[full] = rng.choice(mt["alt_descriptions"])
                pm, pd = {}, {}
                for p, mp in (mt.get("params") or {}).items():
                    if rng.random() < self.param_prob and mp.get("alt_names"):
                        pm[p] = rng.choice(mp["alt_names"])
                    if rng.random() < self.desc_prob and mp.get("alt_descriptions"):
                        pd[p] = rng.choice(mp["alt_descriptions"])
                param[full], pdesc[full] = pm, pd
        return {"tool": tool, "param": param, "tdesc": tdesc, "pdesc": pdesc}

    # ---- applying a plan ----------------------------------------------------------------------------------------
    @staticmethod
    def _refs(text: str, plan: dict, pm: dict | None = None) -> str:
        """Rename tool references (app.tool) and, inside this tool's own texts, its renamed parameter names."""
        text = REF.sub(lambda m: plan["tool"].get(m.group(0), m.group(0)), text or "")
        for old, new in sorted((pm or {}).items(), key=lambda kv: -len(kv[0])):     # longest first
            text = re.sub(r"(?<![\w.])" + re.escape(old) + r"(?![\w])", new, text)
        return text

    def spec(self, full: str, plan: dict) -> dict:
        s = copy.deepcopy(compact_tool(self.index[full]))
        if full not in plan["tool"]:
            if s.get("description"):
                s["description"] = self._refs(s["description"], plan)
            for p in s["parameters"].get("properties", {}).values():
                if p.get("description"):
                    p["description"] = self._refs(p["description"], plan)
            return s
        pm, pd = plan["param"].get(full, {}), plan["pdesc"].get(full, {})
        s["name"] = plan["tool"][full]
        s["description"] = self._refs(plan["tdesc"].get(full, s.get("description", "")), plan, pm)
        params = s["parameters"]
        props = {}
        for k, p in params.get("properties", {}).items():
            p = dict(p)
            if k in pd or p.get("description"):
                p["description"] = self._refs(pd.get(k, p.get("description", "")), plan, pm)
            props[pm.get(k, k)] = p
        params["properties"] = props
        if "required" in params:
            params["required"] = [pm.get(k, k) for k in params["required"]]
        if "x-one-of" in params:
            params["x-one-of"] = [[pm.get(k, k) for k in grp] for grp in params["x-one-of"]]
        return s

    def call(self, c: dict, plan: dict) -> dict:
        name, args = c["name"], dict(c.get("arguments") or {})
        if name == ASK:
            tn = args.get("tool_name")
            if isinstance(tn, str) and tn in plan["tool"]:
                pm = plan["param"].get(tn, {})
                args["tool_name"] = plan["tool"][tn]
                if isinstance(args.get("params"), list):
                    args["params"] = [pm.get(p, p) if isinstance(p, str) else p for p in args["params"]]
            return {"name": name, "arguments": args}
        pm = plan["param"].get(name, {})
        return {"name": plan["tool"].get(name, name), "arguments": {pm.get(k, k): v for k, v in args.items()}}

    def trajectory(self, t: dict, rng: random.Random) -> dict:
        plan = self.plan(t["tools"], rng)
        out = dict(t)
        out["tools"] = [plan["tool"].get(x, x) for x in t["tools"]]
        out["tool_specs"] = [self.spec(x, plan) for x in t["tools"]]
        out["steps"] = [{**st, "call": self.call(st["call"], plan)} for st in t["steps"]]
        hist = []
        for turn in t["history"]:
            if turn.get("tool_calls"):
                turn = {**turn, "tool_calls": [self.call(c, plan) for c in turn["tool_calls"]]}
            hist.append(turn)
        out["history"] = hist
        out["renamed"] = {"apps": sorted({k.split(".")[0] for k in plan["tool"]}), "tools": len(plan["tool"])}
        return out


def check_renamed(t: dict) -> list[str]:
    """Every call and ask target must name a tool in the trajectory's own specs, with parameter names from that spec."""
    specs = {s["name"]: s for s in t["tool_specs"]}
    probs = []
    calls = [st["call"] for st in t["steps"]] + [c for turn in t["history"] for c in (turn.get("tool_calls") or [])]
    target_calls = [st["call"] for st in t["steps"]]
    for c in calls:
        if c["name"] not in specs:
            if c in target_calls:
                probs.append(f"call to {c['name']} not in the specs")
            continue
        props = specs[c["name"]]["parameters"].get("properties", {})
        bad = [k for k in c["arguments"] if k not in props]
        if bad and c["name"] != ASK:
            probs.append(f"{c['name']}: argument(s) {bad} not in its spec")
        if c["name"] == ASK:
            tn = c["arguments"].get("tool_name")
            if tn not in specs:
                probs.append(f"ask names {tn} which is not in the specs")
            else:
                tp = specs[tn]["parameters"].get("properties", {})
                missing = [p for p in c["arguments"].get("params") or [] if p not in tp]
                if missing:
                    probs.append(f"ask params {missing} not in the spec of {tn}")
    if len(specs) != len(t["tool_specs"]):
        probs.append("duplicate tool names after renaming")
    return probs
