"""Inference: /data/<split> -> result/submission.jsonl (one line per sample, input order), then schema validation.

    python -m lifemate.infer.run --data /data --model artifacts/models/v1-epoch2 [--asr_cache FILE] [--limit N]

Per sample: transcribe its user-turn clips (online, or from a transcript cache for development runs), run the agent
loop in ONE ToolEnv of that split, write {"id", "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}.
A sample that fails still gets a line (whatever its env recorded). The split's own zaic_tools and catalog are loaded
from the data folder at runtime; no app list is hard-coded. Latency (median/p95/max) goes to <out>.stats.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from lifemate.asr.cache import load_cache  # noqa: E402
from lifemate.asr.context import asr_context  # noqa: E402
from lifemate.config import SAMPLE_BUDGET_S, load_config  # noqa: E402
from lifemate.controller.loop import make_episode, run_episodes, submission_line  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples, tool_index, user_turn_audio, zaic_tools  # noqa: E402


def log(msg: str) -> None:
    print(f"[infer {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_split(data: Path) -> tuple[Path, str]:
    """Accept either a folder holding <split>/ or the split folder itself (as mounted at /data). Returns (root, split)
    such that <root>/<split>/<split>.jsonl exists, linking a temporary root when the folder name differs."""
    data = data.resolve()
    for d in sorted(p for p in data.iterdir() if p.is_dir()):
        if (d / f"{d.name}.jsonl").exists() and (d / f"{d.name}_tools_catalog.json").exists():
            if d.name != "train":
                return data, d.name
    for f in sorted(data.glob("*_tools_catalog.json")):
        split = f.name[: -len("_tools_catalog.json")]
        if (data / f"{split}.jsonl").exists():
            if data.name == split:
                return data.parent, split
            tmp = Path(tempfile.mkdtemp(prefix="lifemate_root_"))
            (tmp / split).symlink_to(data, target_is_directory=True)
            return tmp, split
    raise FileNotFoundError(f"no <split>.jsonl + <split>_tools_catalog.json under {data}")


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate(root: Path, split: str, out: Path) -> int:
    p = subprocess.run([sys.executable, "-m", "zaic_tools", "--split", split, "--root", "..", "--validate", str(out.resolve())],
                       cwd=root / split, capture_output=True, text=True)
    print(p.stdout[-2000:], p.stderr[-2000:], sep="\n")
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--data", default=os.environ.get("LIFEMATE_INFER_DATA", "/data"))
    ap.add_argument("--model", required=True, help="merged agent model dir (or HF id)")
    ap.add_argument("--out", default="result/submission.jsonl")
    ap.add_argument("--asr_cache", default=None, help="transcript cache jsonl (development); default: online ASR")
    ap.add_argument("--budget_s", type=float, default=SAMPLE_BUDGET_S - 5, help="stop starting new steps after this")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(a.config)
    ag = cfg["agent"]
    seed_everything(cfg["seed"])
    root, split = resolve_split(Path(a.data))
    zt = zaic_tools(root, split)
    samples = load_samples(root, split)
    if a.limit:
        samples = samples[: a.limit]
    index = tool_index(load_catalog(root, split))
    log(f"split {split} at {root}: {len(samples)} samples, {len(index)} tools in catalog")

    from lifemate.agent.model import HFEngine
    engine = HFEngine(a.model, max_new_tokens=ag["max_new_tokens"], max_batch=1,
                      prefix_cache=bool(cfg["harness"].get("prefix_cache")))
    asr, cache = None, None
    if a.asr_cache:
        cache = load_cache(Path(a.asr_cache))
        n_asr = 0
    else:
        from lifemate.asr.base import make_asr
        asr = make_asr(cfg)
        n_asr = sum(p.numel() for p in asr.model.model.parameters())
    log(f"parameters loaded: agent {engine.n_params:,} + ASR {n_asr:,} = {engine.n_params + n_asr:,}"
        + (" (ASR from cache, not loaded)" if cache is not None else ""))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = []
    with open(out, "w", encoding="utf-8") as f:
        for k, s in enumerate(samples):
            t0 = time.time()
            ep, err = None, None
            try:
                clips = user_turn_audio(s)
                if cache is not None:
                    texts = [cache[p] for p in clips]
                else:
                    ctx = [asr_context(s, cfg["asr"].get("context_mode", "full"))] * len(clips) if cfg["asr"].get("context") else None
                    texts = asr.transcribe([str(root / split / p) for p in clips], ctx)
                t_asr = time.time() - t0
                ep = make_episode(s, zt, split, root, index, dict(zip(clips, texts)), cfg=cfg)
                ep.t0 = t0
                run_episodes([ep], engine, ag["tool_format"], budget_s=a.budget_s)
                line = submission_line(ep)
            except Exception as e:  # a failed sample still gets its line
                err = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                env = ep.env if ep is not None else zt.ToolEnv(split, s["id"], root=root)
                line = {"id": s["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
                t_asr = t_asr if "t_asr" in locals() else 0.0
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            dt = time.time() - t0
            stats.append({"id": s["id"], "seconds": round(dt, 3), "asr_seconds": round(t_asr, 3), "n_calls": len(line["tool_calls"]),
                          "stop_reason": ep.stop_reason if ep else "error", "error": err})
            if (k + 1) % 25 == 0 or k + 1 == len(samples):
                lat = sorted(x["seconds"] for x in stats)
                log(f"{k + 1}/{len(samples)} done; latency median {statistics.median(lat):.2f}s p95 {lat[int(.95 * (len(lat) - 1))]:.2f}s max {lat[-1]:.2f}s")
    lat = sorted(x["seconds"] for x in stats)
    summary = {"split": split, "n": len(stats), "latency_median": statistics.median(lat), "latency_p95": lat[int(.95 * (len(lat) - 1))],
               "latency_max": lat[-1], "over_30s": sum(x > SAMPLE_BUDGET_S for x in lat), "errors": sum(bool(x["error"]) for x in stats),
               "stop_reasons": {r: sum(x["stop_reason"] == r for x in stats) for r in sorted({x["stop_reason"] for x in stats})},
               "params_agent": engine.n_params, "params_asr": n_asr, "per_sample": stats}
    Path(str(out) + ".stats.json").write_text(json.dumps(summary, indent=1))
    log(f"wrote {out} ({len(stats)} lines); errors {summary['errors']}, over 30 s {summary['over_30s']}")
    rc = validate(root, split, out)
    log("validation " + ("passed" if rc == 0 else f"FAILED (exit {rc})"))
    sys.exit(rc)


if __name__ == "__main__":
    main()
