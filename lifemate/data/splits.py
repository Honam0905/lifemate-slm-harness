"""Grouped train/dev/audit split with unseen-app slices. Built once, frozen to JSON, never re-randomised."""
from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from pathlib import Path


def apps_used(sample: dict) -> set[str]:
    """Apps touched by the gold calls, ask targets and history calls of a train sample."""
    apps: set[str] = set()
    calls = list(sample["label"]["tool_calls"])
    for t in sample["history"]:
        calls += t.get("tool_calls") or []
    for c in calls:
        apps.add(c["name"].split(".")[0])
        if c["name"] == "sys.ask_missing_info":
            apps.add(str(c["arguments"].get("tool_name", "")).split(".")[0])
    apps.discard("sys")
    return apps


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").lower().split())


def episode_groups(samples: list[dict]) -> list[list[str]]:
    """Union samples that share the first user utterance or the final utterance (same episode or near-duplicate)."""
    parent = {s["id"]: s["id"] for s in samples}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen: dict[str, str] = {}
    for s in samples:
        first = next((t.get("audio_text") for t in s["history"] if t["role"] == "user"), s.get("audio_text"))
        for key in ("first:" + _norm(first), "final:" + _norm(s.get("audio_text"))):
            if key in seen:
                parent[find(s["id"])] = find(seen[key])
            else:
                seen[key] = s["id"]
    groups: dict[str, list[str]] = {}
    for s in samples:
        groups.setdefault(find(s["id"]), []).append(s["id"])
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def make_splits(samples: list[dict], cfg: dict) -> dict:
    sc = cfg["splits"]
    unseen = set(sc["unseen_apps"])
    by_id = {s["id"]: s for s in samples}
    groups = episode_groups(samples)
    rng = random.Random(sc["seed"])
    uses_unseen = [g for g in groups if any(apps_used(by_id[i]) & unseen for i in g)]
    known = [g for g in groups if g not in uses_unseen]
    rng.shuffle(uses_unseen)
    rng.shuffle(known)

    n_unseen = sum(len(g) for g in uses_unseen)
    dev, audit, unseen_dev, unseen_audit = [], [], [], []
    acc = 0
    for g in uses_unseen:  # unseen-app samples go to dev/audit only
        if acc < sc["unseen_dev_fraction"] * n_unseen:
            dev += g; unseen_dev += g
        else:
            audit += g; unseen_audit += g
        acc += len(g)
    train = []
    for g in known:
        if len(dev) < sc["dev"]:
            dev += g
        elif len(audit) < sc["audit"]:
            audit += g
        else:
            train += g
    out = {
        "name": sc["name"], "seed": sc["seed"], "unseen_apps": sorted(unseen),
        "train": sorted(train), "dev": sorted(dev), "audit": sorted(audit),
        "unseen_dev": sorted(unseen_dev), "unseen_audit": sorted(unseen_audit),
    }
    blob = json.dumps({k: out[k] for k in ("train", "dev", "audit")}, sort_keys=True).encode()
    out["sha256"] = hashlib.sha256(blob).hexdigest()
    return out


def load_splits(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
