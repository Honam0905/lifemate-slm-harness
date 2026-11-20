"""Tokenise trajectory files for training and enforce the 16K budget (fail loudly; never truncate).

    python scripts/tokenize_trajectories.py --parts train dev [--variants gold asr] [--show 2]
Writes artifacts/tokenized/<run>/<part>.<variant>.pt (list of dicts) and prints length / mask statistics.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.config import CTX_MAX, artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import load_catalog, tool_index  # noqa: E402
from lifemate.data.mix import filter_noops  # noqa: E402
from lifemate.train.collator import decoded_targets, encode  # noqa: E402


def main():
    import torch
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--parts", nargs="+", default=["train", "dev"])
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--show", type=int, default=2, help="print decoded targets of N examples")
    a = ap.parse_args()
    cfg = load_config(a.config)
    art = artifacts_dir(cfg)
    ag = cfg["agent"]
    tok = AutoTokenizer.from_pretrained(ag["model_id"], revision=ag["revision"])
    index = tool_index(load_catalog(data_root(cfg), "train"))
    tdir = art / "trajectories" / cfg["splits"]["name"]
    odir = art / "tokenized" / (cfg["train"].get("data_tag") or cfg["run_name"])
    odir.mkdir(parents=True, exist_ok=True)
    budget = CTX_MAX - ag["max_new_tokens"]
    over = []
    for part in a.parts:
        for variant in a.variants or cfg["train"]["variants"]:
            src = tdir / f"{part}.{variant}.jsonl"
            if not src.exists():
                print(f"[tok] skip {src} (not built)")
                continue
            trajs = [json.loads(l) for l in open(src, encoding="utf-8")]
            if not trajs:   # e.g. train.synth.jsonl when no synthetic conversations are present
                print(f"[tok] skip {src} (empty)")
                continue
            if part == "train" and cfg["train"].get("none_keep"):
                trajs, st = filter_noops(trajs, cfg["train"]["none_keep"], cfg.get("seed", 0))
                print(f"[tok] {part}.{variant}: none_keep {cfg['train']['none_keep']}: {st}")
            encs = [encode(tok, t, index, ag["tool_format"], prompt_opts=cfg.get("prompt")) for t in trajs]
            lens = sorted(len(e.input_ids) for e in encs)
            bad = sum(e.boundary_misaligned for e in encs)
            over += [(e.id, variant, len(e.input_ids)) for e in encs if len(e.input_ids) > budget]
            print(f"[tok] {part}.{variant}: n={len(encs)} tokens median={statistics.median(lens):.0f} p95={lens[int(.95 * len(lens))]} "
                  f"max={lens[-1]} total={sum(lens) / 1e6:.2f}M target_tokens={sum(e.n_target for e in encs)} "
                  f"misaligned_span_edges={bad}")
            torch.save([e.__dict__ for e in encs], odir / f"{part}.{variant}.pt")
            for e in encs[: a.show]:
                print(f"  {e.id} targets: {decoded_targets(tok, e)}")
    if over:
        print(f"[tok] {len(over)} sequences exceed the {budget}-token budget (ctx {CTX_MAX} - max_new_tokens): {over[:10]}")
        sys.exit(1)
    print(f"[tok] all sequences within budget; written to {odir}")


if __name__ == "__main__":
    main()
