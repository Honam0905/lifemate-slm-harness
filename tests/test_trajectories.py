"""Causality / leakage: in every training trajectory, each x-ref ID used by a target call must already be visible
before that decision (history, transcript, or an earlier result). Violations are listed, not hidden; the notes found
one gold ID with no visible source, so we allow a tiny number and print them."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.config import data_root, load_config  # noqa: E402
from lifemate.data.loader import gold_transcripts, load_catalog, load_samples, tool_index, zaic_tools  # noqa: E402
from lifemate.data.trajectories import build_trajectory  # noqa: E402

MAX_UNSOURCED = 3


def _ids(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for x in v:
            yield from _ids(x)


def test_xref_ids_visible_before_use():
    cfg = load_config()
    root = data_root(cfg)
    zt = zaic_tools(root, "train")
    index = tool_index(load_catalog(root, "train"))
    bad, n = [], 0
    for s in load_samples(root, "train"):
        t, probs = build_trajectory(s, zt, root, gold_transcripts(s), "gold")
        assert not probs, probs
        seen = json.dumps({"h": t["history"], "r": t["request"], "c": t["context"]}, ensure_ascii=False)
        for st in t["steps"]:
            props = index[st["call"]["name"]]["parameters"].get("properties", {})
            for k, v in st["call"]["arguments"].items():
                if props.get(k, {}).get("x-ref"):
                    for i in _ids(v):
                        n += 1
                        if i not in seen:
                            bad.append((s["id"], st["call"]["name"], k, i))
            seen += json.dumps(st["result"], ensure_ascii=False)
    print(f"  {n} x-ref values checked, {len(bad)} with no earlier source: {bad}")
    assert n > 1000 and len(bad) <= MAX_UNSOURCED


if __name__ == "__main__":
    from tests.runner import run
    run(globals())
