"""Final-fit training data: ALL 1,545 train conversations (train + dev + audit partitions) and ALL apps (no held-out
apps), for the submission model trained after the dev experiments have chosen the recipe. Never evaluate a model trained
on this on dev or audit.

    python scripts/build_final_fit.py --config configs/v2r_all.yaml
Writes artifacts/trajectories/<split>/train.all{gold,asr,earlier,synth}.jsonl, same formats as the v2 files
(final turns replayed in the train ToolEnv; earlier turns one transcript each; synthetic one per template family).
Then scripts/apply_rename.py --config configs/v2r_all.yaml writes the *_r copies the config trains on.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_v2_data import earlier_turns, synthetic  # noqa: E402
from lifemate.asr.base import backend_name  # noqa: E402
from lifemate.asr.cache import cache_path, load_cache  # noqa: E402
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import gold_transcripts, load_samples, user_turn_audio, zaic_tools  # noqa: E402
from lifemate.data.splits import load_splits  # noqa: E402
from lifemate.data.trajectories import build_trajectory  # noqa: E402


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2r_all.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root, art = data_root(cfg), artifacts_dir(cfg)
    sp = load_splits(art / "splits" / f"{cfg['splits']['name']}.json")
    samples = load_samples(root, "train")
    all_ids = {s["id"] for s in samples}
    out = art / "trajectories" / sp["name"]
    out.mkdir(parents=True, exist_ok=True)
    asr = load_cache(cache_path(art, backend_name(cfg), "train"))
    miss = [p for s in samples for p in user_turn_audio(s) if p not in asr]
    if miss:
        sys.exit(f"train ASR cache incomplete: {len(miss)} clips missing (e.g. {miss[0]})")
    zt = zaic_tools(root, "train")
    for variant in ("gold", "asr"):
        rows, probs = [], []
        for s in samples:
            tr = gold_transcripts(s) if variant == "gold" else {p: asr[p] for p in user_turn_audio(s)}
            t, pr = build_trajectory(s, zt, root, tr, variant, frozenset())       # no hidden apps
            t["source"] = "final"
            rows.append(t); probs += pr
        write(out / f"train.all{variant}.jsonl", rows)
        print(f"[final-fit] all{variant}: {len(rows)} final-turn trajectories, {len(probs)} replay problems")
        if probs:
            print("   ", probs[:5])
    st = collections.Counter()
    early = earlier_turns(samples, set(), asr, st)
    write(out / "train.allearlier.jsonl", early)
    print(f"[final-fit] allearlier: {len(early)} | {dict(sorted(st.items()))}")
    cards = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "synth/conv/cards").glob("batch_*/*.json")}
    st = collections.Counter()
    syn = synthetic(cards, set(), all_ids, st)
    write(out / "train.allsynth.jsonl", syn)
    print(f"[final-fit] allsynth: {len(syn)} | {dict(sorted(st.items()))}")
    print(f"[final-fit] partitions used: train {len(sp['train'])} + dev {len(sp['dev'])} + audit {len(sp['audit'])} = {len(all_ids)}")


if __name__ == "__main__":
    main()
