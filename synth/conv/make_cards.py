"""Build one batch of environment cards for synthetic conversation generation (TRAIN partition only).

    python3 -I synth/conv/make_cards.py --batch 1 [--n 30] [--seed 1]
A card = one real train-partition sample's environment: its context, installed tools, the app data the simulator holds
(env.get_state(), allowed on the train split), its real conversation history, and 3 real gold examples from OTHER
train-partition samples that use the same apps. Writes synth/conv/cards/batch_NN/<card_id>.json; samples already used
by earlier batches are skipped. Dev/audit samples are never used.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import installed_tools, load_samples, zaic_tools  # noqa: E402
from lifemate.data.splits import load_splits  # noqa: E402

ASK = "sys.ask_missing_info"


def outcome(calls: list) -> str:
    return "none" if not calls else "ask" if calls[-1]["name"] == ASK else "act"


def history_card(s: dict) -> list:
    out = []
    for t in s["history"]:
        if t["role"] == "user":
            out.append({"role": "user", "text": t["audio_text"]})
        else:
            turn = {"role": "assistant", "text": t.get("text", "")}
            if t.get("tool_calls"):
                turn["tool_calls"] = t["tool_calls"]
                turn["tool_results"] = t.get("tool_results")
            out.append(turn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    root, art = data_root(cfg), artifacts_dir(cfg)
    sp = load_splits(art / "splits" / f"{cfg['splits']['name']}.json")
    train_ids = set(sp["train"])
    samples = {s["id"]: s for s in load_samples(root, "train") if s["id"] in train_ids}
    zt = zaic_tools(root, "train")
    used_path = HERE / "cards" / "used.json"
    used = set(json.loads(used_path.read_text())) if used_path.exists() else set()
    pool = sorted(set(samples) - used)
    rng = random.Random(a.seed * 1000 + a.batch)
    pick = rng.sample(pool, min(a.n, len(pool)))
    out = HERE / "cards" / f"batch_{a.batch:02d}"
    out.mkdir(parents=True, exist_ok=True)
    apps_of = {i: set(s["context"]["installed_apps"]) for i, s in samples.items()}
    for sid in pick:
        s = samples[sid]
        env = zt.ToolEnv("train", sid, root=root)
        mine = apps_of[sid]
        # 3 real examples from other train-partition samples with the most overlapping apps, different outcomes first
        others = sorted((i for i in samples if i != sid), key=lambda i: (-len(apps_of[i] & mine), i))[:40]
        random.Random(sid).shuffle(others)
        chosen = []
        for want in ("act", "ask", "none"):          # one example of each outcome when the neighbours have one
            i = next((i for i in others if outcome(samples[i]["label"]["tool_calls"]) == want and i not in chosen), None)
            if i:
                chosen.append(i)
        for i in others:                             # fill up to 3 with whatever is closest
            if len(chosen) >= 3:
                break
            if i not in chosen:
                chosen.append(i)
        ex = []
        for i in chosen[:3]:
            t = samples[i]
            ex.append({"from_sample": i, "installed_apps": sorted(apps_of[i]), "request": t["audio_text"],
                       "had_history": bool(t["history"]), "calls": t["label"]["tool_calls"],
                       "results": t["label"]["tool_results"], "outcome": outcome(t["label"]["tool_calls"])})
        card = {
            "card_id": f"c{a.batch:02d}_{sid}", "env_sample": sid,
            "context": {k: s["context"][k] for k in ("current_time", "timezone", "location", "user_profile")},
            "installed_tools": installed_tools(s) + [ASK],
            "app_data": env.get_state(),
            "history": history_card(s),
            "examples": ex,
        }
        (out / f"{card['card_id']}.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    used_path.parent.mkdir(parents=True, exist_ok=True)
    used_path.write_text(json.dumps(sorted(used | set(pick))))
    print(f"[cards] batch {a.batch:02d}: {len(pick)} cards -> {out} ({len(pool) - len(pick)} train samples left unused)")


if __name__ == "__main__":
    main()
