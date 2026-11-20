"""Training-mix filters applied when trajectories are tokenised (pure Python, no torch).

`train.none_keep: {source: fraction}` keeps only that fraction of the "do nothing" examples (no tool call, no ask) of
each source, chosen by a hash of the example id, so the choice is deterministic and identical for a renamed copy and
its original. v2r trained with 11.2% do-nothing per epoch against 7.5% in the real data (synthetic `none` 24.4%,
earlier turns 11.5%) and answered nothing on 23 dev samples where v1 did on 11 (7 Nov analysis).
"""
from __future__ import annotations

import collections
import hashlib


def is_noop(t: dict) -> bool:
    return not t.get("steps") and bool(t.get("stop", True))


def keep_example(t: dict, none_keep: dict | None, seed: int = 0) -> bool:
    if not none_keep or not is_noop(t):
        return True
    f = float(none_keep.get(t.get("source", "final"), 1.0))
    if f >= 1:
        return True
    h = int(hashlib.sha1(f"none_keep|{seed}|{t['id']}".encode()).hexdigest(), 16) / 2 ** 160
    return h < f


def filter_noops(trajs: list[dict], none_keep: dict | None, seed: int = 0) -> tuple[list[dict], dict]:
    """(kept trajectories, stats per source: noop before -> after, total before -> after)."""
    if not none_keep:
        return trajs, {}
    kept = [t for t in trajs if keep_example(t, none_keep, seed)]
    st = collections.defaultdict(lambda: [0, 0, 0, 0])
    for t in trajs:
        s = st[t.get("source", "final")]; s[0] += 1; s[1] += is_noop(t)
    for t in kept:
        s = st[t.get("source", "final")]; s[2] += 1; s[3] += is_noop(t)
    return kept, {k: f"{v[0]}->{v[2]} examples, do-nothing {v[1]}->{v[3]}" for k, v in sorted(st.items())}
