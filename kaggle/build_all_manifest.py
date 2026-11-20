"""Model B manifest: every user-turn clip of ALL 1,545 train samples (train + dev + audit partitions) with its
annotator transcript. Only for the final ASR fit — a model trained on it must never be scored on dev/audit.

    python kaggle/build_all_manifest.py --out /tmp/zalo/artifacts/asr_ft_all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lifemate.config import data_root, load_config  # noqa: E402
from lifemate.data.loader import gold_transcripts, load_samples, user_turn_audio  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for s in load_samples(data_root(cfg), "train"):
            gold = gold_transcripts(s)
            for clip in user_turn_audio(s):
                text = (gold.get(clip) or "").strip()
                if text:
                    f.write(json.dumps({"audio": clip, "text": text, "sample_id": s["id"], "group": s["id"]},
                                       ensure_ascii=False) + "\n")
                    n += 1
    print(f"[all-manifest] {n} clips from all train samples -> {out / 'train.jsonl'} (no holdout: final fit only)")


if __name__ == "__main__":
    main()
