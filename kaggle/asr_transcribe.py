"""Transcribe clips with Qwen3-ASR (optionally + a LoRA adapter, merged in memory) — one shard per process/GPU.

    CUDA_VISIBLE_DEVICES=0 python kaggle/asr_transcribe.py --model /tmp/hf/qwen3-asr --adapter out/A/step100 \
        --split train --part dev --shard 0 --nshards 2 --out /tmp/shards/x.0.jsonl
Writes {"audio": <path relative to the split folder>, "text": ...} lines (the format of lifemate/asr/cache.py), resumable.
Context and decoding match the pipeline: names-only context per sample, forced language, greedy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lifemate.asr.context import asr_context  # noqa: E402
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import load_samples, user_turn_audio  # noqa: E402
from lifemate.data.splits import load_splits  # noqa: E402

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def clip_list(cfg: dict, split: str, part: str, context_mode: str) -> list[tuple[str, str]]:
    """(clip path, context) in a fixed order, each clip once."""
    root = data_root(cfg)
    samples = load_samples(root, split)
    if split == "train" and part != "all":
        keep = set(load_splits(artifacts_dir(cfg) / "splits" / f"{cfg['splits']['name']}.json")[part])
        samples = [s for s in samples if s["id"] in keep]
    out, seen = [], set()
    for s in samples:
        ctx = asr_context(s, "names") if context_mode == "names" else ""
        for p in user_turn_audio(s):
            if p not in seen:
                seen.add(p)
                out.append((p, ctx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--model", required=True, help="local Qwen3-ASR snapshot dir")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir from kaggle/asr_ddp.py (omit for the base model)")
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    ap.add_argument("--split", required=True, choices=["train", "public_test"])
    ap.add_argument("--part", default="all", choices=["all", "train", "dev", "audit"])
    ap.add_argument("--context_mode", default="names", choices=["names", "none"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(ROOT / a.config)
    root = data_root(cfg)
    clips = clip_list(cfg, a.split, a.part, a.context_mode)[a.shard::a.nshards]
    if a.limit:
        clips = clips[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["audio"] for l in open(out, encoding="utf-8") if l.strip()}
    todo = [c for c in clips if c[0] not in done]
    print(f"[transcribe] shard {a.shard}/{a.nshards}: {len(clips)} clips, {len(todo)} to do, adapter {a.adapter}", flush=True)
    if not todo:
        return
    from qwen_asr import Qwen3ASRModel
    w = Qwen3ASRModel.from_pretrained(a.model, dtype=DTYPES[a.dtype], device_map="cuda:0",
                                      max_inference_batch_size=a.batch, max_new_tokens=cfg["asr"]["max_new_tokens"])
    if a.adapter:
        from peft import PeftModel
        w.model = PeftModel.from_pretrained(w.model, a.adapter).merge_and_unload().eval()
    gc = w.model.generation_config
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None   # greedy, as the pipeline decodes
    lang = cfg["asr"].get("language")
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), a.chunk):
            part = todo[i: i + a.chunk]
            res = w.transcribe(audio=[str(root / a.split / p) for p, _ in part],
                               context=[c for _, c in part] if a.context_mode == "names" else "",
                               language=[lang] * len(part) if lang else None)
            for (p, _), r in zip(part, res):
                f.write(json.dumps({"audio": p, "text": r.text.strip()}, ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"[transcribe] shard {a.shard}: {i + len(part)}/{len(todo)} clips, {el / (i + len(part)):.3f}s/clip", flush=True)


if __name__ == "__main__":
    main()
