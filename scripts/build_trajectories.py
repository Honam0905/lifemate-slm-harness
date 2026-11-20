"""Build causal gold trajectories for every partition and transcript variant.

    python scripts/build_trajectories.py [--variants gold asr] [--limit N] [--out DIR]
Writes <out>/<part>.<variant>.jsonl (default out = artifacts/trajectories/<split name>). Train trajectories hide the
unseen-app schemas (they only appear there as installed distractors); dev/audit keep every installed schema.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.asr.base import backend_name  # noqa: E402
from lifemate.asr.cache import cache_path, load_cache  # noqa: E402
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import gold_transcripts, load_samples, user_turn_audio, zaic_tools  # noqa: E402
from lifemate.data.splits import load_splits  # noqa: E402
from lifemate.data.trajectories import build_trajectory  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--parts", nargs="+", default=["train", "dev", "audit"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    root, art = data_root(cfg), artifacts_dir(cfg)
    sp = load_splits(art / "splits" / f"{cfg['splits']['name']}.json")
    out = Path(a.out) if a.out else art / "trajectories" / sp["name"]
    out.mkdir(parents=True, exist_ok=True)
    zt = zaic_tools(root, "train")
    by_id = {s["id"]: s for s in load_samples(root, "train")}
    variants = a.variants or cfg["train"]["variants"]
    asr = {}
    if "asr" in variants:
        asr = load_cache(cache_path(art, backend_name(cfg), "train"))
        missing = [p for s in by_id.values() for p in user_turn_audio(s) if p not in asr]
        if missing:
            sys.exit(f"ASR cache incomplete: {len(missing)} clips missing (first: {missing[0]})")
    unseen = set(sp["unseen_apps"])
    all_problems = []
    for part in a.parts:
        ids = sp[part][: a.limit] if a.limit else sp[part]
        for variant in variants:
            stats = collections.Counter()
            path = out / f"{part}.{variant}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for i in ids:
                    s = by_id[i]
                    tr = gold_transcripts(s) if variant == "gold" else {p: asr[p] for p in user_turn_audio(s)}
                    traj, probs = build_trajectory(s, zt, root, tr, variant, unseen if part == "train" else frozenset())
                    all_problems += probs
                    stats["n"] += 1
                    stats["steps"] += len(traj["steps"])
                    stats["stop_targets"] += traj["stop"]
                    f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            print(f"[traj] {part}.{variant}: {stats['n']} trajectories, {stats['steps']} call targets, "
                  f"{stats['stop_targets']} STOP targets -> {path}")
    print(f"[traj] problems: {len(all_problems)}")
    for p in all_problems[:20]:
        print("  ", p)


if __name__ == "__main__":
    main()
