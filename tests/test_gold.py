"""Whole-dataset checks: gold replay = final_state; gold self-score = 100.00; empty submission = 7.18;
gold + duplicated first read = 2|G|/(2|G|+1) on every sample with a read."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.config import data_root, load_config  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples, zaic_tools  # noqa: E402
from lifemate.eval.gold import build_gold_hashes, self_checks  # noqa: E402

_cache = {}


def _setup():
    if not _cache:
        cfg = load_config()
        root = data_root(cfg)
        zt = zaic_tools(root, "train")
        samples = load_samples(root, "train")
        hashes, problems = build_gold_hashes(zt, root, samples)
        _cache.update(cfg=cfg, root=root, zt=zt, samples=samples, hashes=hashes, problems=problems)
    return _cache


def test_gold_replay_matches_final_state():
    c = _setup()
    assert len(c["samples"]) == 1545
    assert c["problems"] == [], c["problems"][:5]


def test_self_checks():
    c = _setup()
    r = self_checks(c["zt"], c["root"], c["samples"], c["hashes"], load_catalog(c["root"], "train"))
    for v in ("primary", "pessimistic", "optimistic"):
        assert r[f"gold_self_{v}"] == 100.0, r
    assert r["n_empty_gold"] == 111
    assert round(r["empty_submission"], 2) == 7.18, r["empty_submission"]
    assert r["dup_read_samples"] > 900 and r["dup_read_failures"] == [], r["dup_read_failures"][:5]


if __name__ == "__main__":
    from tests.runner import run
    run(globals())
