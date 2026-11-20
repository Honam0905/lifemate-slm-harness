"""Transcribe every user-turn clip of a split (history + final request) into the transcript cache. Resumable.

    python scripts/build_asr_cache.py --split train [--limit 64]
Prints a quick word error rate against the annotator transcripts on train.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.asr.base import backend_name, make_asr  # noqa: E402
from lifemate.asr.cache import append_cache, cache_path, load_cache  # noqa: E402
from lifemate.asr.context import asr_context  # noqa: E402
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import gold_transcripts, load_samples, user_turn_audio  # noqa: E402
from lifemate.data.splits import load_splits  # noqa: E402


def words(t: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", t).lower())


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--split", required=True, choices=["train", "public_test"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
    ap.add_argument("--chunk", type=int, default=256, help="clips per transcribe call / cache flush")
    ap.add_argument("--part", default=None, choices=["train", "dev", "audit"],
                    help="train split only: transcribe just this part of the frozen split (e.g. dev for quick ablations)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root, art = data_root(cfg), artifacts_dir(cfg)
    samples = load_samples(root, a.split)
    if a.part:
        assert a.split == "train", "--part applies to the train split only"
        keep = set(load_splits(art / "splits" / f"{cfg['splits']['name']}.json")[a.part])
        samples = [s for s in samples if s["id"] in keep]
    clips = [p for s in samples for p in user_turn_audio(s)]
    ctx_of = {p: asr_context(s, cfg["asr"].get("context_mode", "full")) for s in samples for p in user_turn_audio(s)} if cfg["asr"].get("context") else None
    if a.limit:
        clips = clips[: a.limit]
    path = cache_path(art, backend_name(cfg), a.split)
    done = load_cache(path)
    todo = [p for p in clips if p not in done]
    print(f"[asr] {a.split}: {len(clips)} clips, {len(done)} cached, {len(todo)} to do -> {path}", flush=True)
    if todo:
        asr = make_asr(cfg)
        t0 = time.time()
        for i in range(0, len(todo), a.chunk):
            part = todo[i: i + a.chunk]
            texts = asr.transcribe([str(root / a.split / p) for p in part], [ctx_of[p] for p in part] if ctx_of else None)
            append_cache(path, list(zip(part, texts)))
            el = time.time() - t0
            print(f"[asr] {i + len(part)}/{len(todo)} clips, {el:.0f}s, {el / (i + len(part)):.3f}s/clip", flush=True)
    done = load_cache(path)
    if a.split == "train":
        gold = {k: v for s in samples for k, v in gold_transcripts(s).items()}
        errs = n = 0
        for p in clips:
            g, h = words(gold[p]), words(done[p])
            errs += edit_distance(g, h)
            n += len(g)
        print(f"[asr] WER vs annotator transcripts (lowercase, punctuation stripped): {100 * errs / max(n, 1):.2f}% over {len(clips)} clips")
        for p in clips[:8]:
            print(f"  gold: {gold[p]}\n  asr : {done[p]}")


if __name__ == "__main__":
    main()
