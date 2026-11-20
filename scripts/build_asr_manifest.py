"""ASR fine-tuning manifest from the TRAIN partition of the frozen split only (dev/audit never enter ASR training).

    python scripts/build_asr_manifest.py [--holdout 0.1] [--out artifacts/asr_ft]
Every user-turn clip (history + final request) of a train-partition sample becomes one example with its annotator
transcript. About 10% of the episode groups (grouped as in the split, so near-duplicates stay together) are held out
for monitoring the ASR run. Writes <out>/train.jsonl and <out>/holdout.jsonl: {"audio", "text", "sample_id", "group"}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import gold_transcripts, load_samples, user_turn_audio  # noqa: E402
from lifemate.data.splits import episode_groups, load_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--holdout", type=float, default=0.1, help="fraction of train-partition episode groups held out")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    root, art = data_root(cfg), artifacts_dir(cfg)
    sp = load_splits(art / "splits" / f"{cfg['splits']['name']}.json")
    keep = set(sp["train"])
    samples = [s for s in load_samples(root, "train") if s["id"] in keep]
    groups = episode_groups(samples)
    rng = random.Random(a.seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    n_hold = max(1, round(a.holdout * len(groups)))
    held = {i for g in (groups[k] for k in order[:n_hold]) for i in g}
    group_of = {i: g[0] for g in groups for i in g}
    by_id = {s["id"]: s for s in samples}
    out = Path(a.out) if a.out else art / "asr_ft"
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "holdout": 0}
    files = {k: open(out / f"{k}.jsonl", "w", encoding="utf-8") for k in counts}
    for sid in sorted(by_id):
        s = by_id[sid]
        gold = gold_transcripts(s)
        part = "holdout" if sid in held else "train"
        for clip in user_turn_audio(s):
            text = (gold.get(clip) or "").strip()
            if not text:
                continue
            files[part].write(json.dumps({"audio": clip, "text": text, "sample_id": sid, "group": group_of[sid]},
                                         ensure_ascii=False) + "\n")
            counts[part] += 1
    for f in files.values():
        f.close()
    assert not (held & (set(sp["dev"]) | set(sp["audit"]))), "dev/audit sample in the ASR manifest"
    print(f"[asr-manifest] {len(samples)} train-partition samples in {len(groups)} groups; held out {n_hold} groups "
          f"({len(held)} samples) -> train {counts['train']} clips, holdout {counts['holdout']} clips -> {out}")


if __name__ == "__main__":
    main()
