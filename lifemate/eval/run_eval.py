"""End-to-end rollouts on dev/audit (train-split samples, our own ToolEnvs) in transcript conditions A/B/C.

    python -m lifemate.eval.run_eval --name base --limit 50 --conditions A C               # untuned baseline
    python -m lifemate.eval.run_eval --name v1-ep2 --model artifacts/models/v1-epoch2 --conditions A C

Conditions: A = annotator transcripts for every user turn; B = ASR for the current request, annotator history;
C = ASR for every user turn (the deployment setting, used to choose between models).
Writes artifacts/eval/<name>/<part>.<cond>/rows.jsonl and summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from lifemate.asr.base import backend_name  # noqa: E402
from lifemate.asr.cache import cache_path, load_cache  # noqa: E402
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.controller.loop import make_episode, run_episodes  # noqa: E402
from lifemate.data.loader import gold_transcripts, load_catalog, load_samples, tool_index, user_turn_audio, zaic_tools  # noqa: E402
from lifemate.data.splits import episode_groups, load_splits  # noqa: E402
from lifemate.eval.gold import load_or_build_hashes  # noqa: E402
from lifemate.eval.metric import VARIANTS, explain_sample, fuzzy_keys  # noqa: E402
from lifemate.eval.report import hist_bucket, summarize  # noqa: E402


def log(msg: str) -> None:
    print(f"[eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def transcripts_for(sample: dict, cond: str, asr: dict[str, str]) -> dict[str, str]:
    gold = gold_transcripts(sample)
    if cond == "A":
        return gold
    if cond == "B":
        return {**gold, sample["audio"]: asr[sample["audio"]]}
    if cond == "C":
        return {p: asr[p] for p in user_turn_audio(sample)}
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=None, help="HF id or local dir (default: config agent.model_id)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir, merged at load time (quick checkpoint eval)")
    ap.add_argument("--part", default="dev", choices=["dev", "audit", "train"])
    ap.add_argument("--conditions", nargs="+", default=["A", "C"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_batch", type=int, default=16)
    ap.add_argument("--max_batch_tokens", type=int, default=64000)
    ap.add_argument("--sequential", action="store_true", help="one sample at a time, exactly the inference path")
    a = ap.parse_args()

    cfg = load_config(a.config)
    ag = cfg["agent"]
    root, art = data_root(cfg), artifacts_dir(cfg)
    sp = load_splits(art / "splits" / f"{cfg['splits']['name']}.json")
    zt = zaic_tools(root, "train")
    samples = load_samples(root, "train")
    by_id = {s["id"]: s for s in samples}
    ids = sp[a.part][: a.limit] if a.limit else sp[a.part]
    part = [by_id[i] for i in ids]
    catalog = load_catalog(root, "train")
    index = tool_index(catalog)
    fz = fuzzy_keys(catalog)
    kinds = {n: t["kind"] for n, t in index.items()}
    hashes = load_or_build_hashes(cfg)
    asr = load_cache(cache_path(art, backend_name(cfg), "train"))
    unseen = set(sp.get("unseen_dev", [])) | set(sp.get("unseen_audit", []))
    group_of = {i: g[0] for g in episode_groups(part) for i in g}

    from lifemate.agent.model import HFEngine
    model = a.model or ag["model_id"]
    revision = a.revision or (ag["revision"] if model == ag["model_id"] else None)
    engine = HFEngine(model, revision=revision, adapter=a.adapter, max_new_tokens=ag["max_new_tokens"],
                      max_batch=1 if a.sequential else a.max_batch, max_batch_tokens=a.max_batch_tokens,
                      prefix_cache=bool(cfg["harness"].get("prefix_cache")))
    log(f"model {model}{' + ' + a.adapter if a.adapter else ''}: {engine.n_params:,} params; {len(part)} {a.part} samples")

    for cond in a.conditions:
        out = art / "eval" / a.name / f"{a.part}.{cond}"
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        engine.gen_seconds = 0.0
        eps = [make_episode(s, zt, "train", root, index, transcripts_for(s, cond, asr), cfg=cfg) for s in part]
        if a.sequential:
            for ep in eps:
                run_episodes([ep], engine, ag["tool_format"])
        else:
            run_episodes(eps, engine, ag["tool_format"], log=log)
        rows = []
        for ep, s in zip(eps, part):
            sub, h = ep.env.get_tool_calls(), ep.env.get_state_hash()
            gold = s["label"]["tool_calls"]
            recs = {v: explain_sample(sub, h, gold, hashes[s["id"]], fz, v, kinds=kinds) for v in VARIANTS}
            p = recs["primary"]
            rows.append({
                "id": s["id"], "condition": cond, "group": group_of[s["id"]],
                "app_slice": "unseen" if s["id"] in unseen else "known", "gold_type": p["gold_type"],
                "hist": hist_bucket(len(s["history"])), "n_gold": len(gold),
                **{f"score_{v}": float(recs[v]["score"]) for v in VARIANTS},
                "hash_ok": p["hash_ok"], "m": p["m"], "n_sub": p["n_sub"], "n_rejected": p["n_rejected"],
                "unmatched_sub": p["unmatched_sub"], "unmatched_gold": p["unmatched_gold"],
                "stop_reason": ep.stop_reason, "prompt_tokens_max": ep.prompt_tokens_max, "steps": ep.steps,
                "submitted": sub,
            })
        with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summ = summarize(rows)
        summ.update(model=model, adapter=a.adapter, condition=cond, part=a.part, seconds=round(time.time() - t0, 1),
                    gen_seconds=round(engine.gen_seconds, 1), config=cfg["_path"], sequential=a.sequential,
                    prompt=cfg.get("prompt"), harness=cfg.get("harness"), asr_backend=backend_name(cfg),
                    cached_tokens=engine.cached_tokens, computed_tokens=engine.computed_tokens,
                    constrained_steps=engine.constrained_steps)
        (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
        log(f"{a.part}.{cond}: primary {summ['primary']} (95% CI {summ['primary_ci95']}), pessimistic {summ['pessimistic']}, "
            f"optimistic {summ['optimistic']}; hash agreement {summ['diagnostics']['hash_agreement']}; "
            f"stops {summ['diagnostics']['stop_reasons']}; {summ['seconds']}s -> {out}")


if __name__ == "__main__":
    main()
