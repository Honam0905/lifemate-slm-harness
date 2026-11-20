"""v2 training trajectories beyond the gold final turns (which scripts/build_trajectories.py already builds).

    python scripts/build_v2_data.py --config configs/v2.yaml [--asr]      # --asr needs the train ASR cache
Writes to artifacts/trajectories/<split>/ :
  train.earlier.jsonl  earlier turns of train-partition conversations (history turn k becomes its own example),
                       ONE transcript per example: gold text or ASR, chosen 50/50 by a hash of the turn (with --asr;
                       without it every example uses gold text)
  train.synth.jsonl    validated synthetic items (synth/conv/accepted), one per template family, gold text only
Both hide the split's unseen apps exactly like the final-turn trajectories, and skip any example that uses them.
Every trajectory carries a `weight` (per-trajectory loss weight) and a `source`.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lifemate.asr.base import backend_name  # noqa: E402
from lifemate.asr.cache import cache_path, load_cache  # noqa: E402
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import gold_transcripts, load_samples, user_turn_audio  # noqa: E402
from lifemate.data.splits import load_splits  # noqa: E402
from lifemate.data.trajectories import ASK, context_view, history_view, visible_tools  # noqa: E402

SYNTH_DIR = ROOT / "synth" / "conv" / "accepted"
NEAR = 0.5   # bigram Jaccard: items at or above this are one template family (keep the first)


def apps_used(calls: list[dict]) -> set[str]:
    used = {c["name"].split(".")[0] for c in calls}
    used |= {str(c["arguments"].get("tool_name", "")).split(".")[0] for c in calls if c["name"] == ASK}
    return used - {"sys", ""}


def pick_asr(key: str) -> bool:
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 2 == 1


def earlier_turns(samples: list[dict], hidden: set[str], asr: dict[str, str] | None, stats: collections.Counter):
    """One example per (user turn, assistant reply) pair in a conversation's history."""
    out = []
    for s in samples:
        gold = gold_transcripts(s)
        h = s["history"]
        for i, t in enumerate(h):
            if t["role"] != "user" or i + 1 >= len(h) or h[i + 1]["role"] != "assistant":
                continue
            a = h[i + 1]
            calls = a.get("tool_calls") or []
            results = a.get("tool_results") or []
            key = f"{s['id']}#{i}"
            if len(results) != len(calls):
                stats["skip: results missing"] += 1; continue
            if apps_used(calls) & hidden:
                stats["skip: uses a held-out app"] += 1; continue
            tools = visible_tools(s, hidden)
            if any(c["name"] not in tools for c in calls):
                stats["skip: tool not installed"] += 1; continue
            if any(isinstance(r, dict) and r.get("ok") is False for r in results):
                stats["skip: a call failed"] += 1; continue
            if not calls and "?" in (a.get("text") or ""):
                # the assistant asked back in plain text; annotators use sys.ask_missing_info for that, so as a
                # "do nothing" target it would be label noise
                stats["skip: text-only clarifying question"] += 1; continue
            use_asr = asr is not None and pick_asr(key)
            tr = {p: asr[p] for p in user_turn_audio(s)} if use_asr else gold
            if any(p not in tr for p in [x["audio"] for x in h[: i + 1] if x["role"] == "user"]):
                stats["skip: transcript missing"] += 1; continue
            ask_end = bool(calls) and calls[-1]["name"] == ASK
            out.append({
                "id": key, "variant": "earlier-asr" if use_asr else "earlier-gold", "source": "earlier", "weight": 1.0,
                "context": context_view(s), "tools": tools,
                "history": history_view({"history": h[:i]}, tr),
                "request": tr[t["audio"]],
                "steps": [{"call": {"name": c["name"], "arguments": c["arguments"]}, "result": r} for c, r in zip(calls, results)],
                "stop": not ask_end,
            })
            stats["kept: " + ("none" if not calls else "ask" if ask_end else "act")] += 1
    return out


def _bigrams(t: str) -> set:
    w = re.findall(r"[^\W_]+", unicodedata.normalize("NFC", t).lower())
    return set(zip(w, w[1:]))


def synthetic(cards: dict, hidden: set[str], train_ids: set[str], stats: collections.Counter):
    out, fams = [], []
    for f in sorted(SYNTH_DIR.glob("batch_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line)
            card = cards[it["card"]]
            if card["env_sample"] not in train_ids:
                stats["skip: not a train-partition environment"] += 1; continue
            bg = _bigrams(it["request"])
            if any(len(bg & b) / max(1, len(bg | b)) >= NEAR for b in fams):
                stats["skip: template duplicate"] += 1; continue
            fams.append(bg)
            if apps_used(it["calls"]) & hidden:
                stats["skip: uses a held-out app"] += 1; continue
            tools = [t for t in card["installed_tools"] if t.split(".")[0] not in hidden]
            if ASK not in tools:
                tools.append(ASK)
            out.append({
                "id": f"{f.stem}:{it['card']}:{len(out)}", "variant": "synth", "source": "synth", "weight": 1.0,
                "context": card["context"], "tools": tools,
                "history": card["history"] if it["use_history"] else [],
                "request": it["request"],
                "steps": [{"call": c, "result": r} for c, r in zip(it["calls"], it["results"])],
                "stop": it["outcome"] != "ask",
            })
            stats[f"kept: {it['outcome']}"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--asr", action="store_true", help="use the train ASR cache for half of the earlier-turn examples")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root, art = data_root(cfg), artifacts_dir(cfg)
    sp = load_splits(art / "splits" / f"{cfg['splits']['name']}.json")
    hidden = set(sp["unseen_apps"])
    train_ids = set(sp["train"])
    samples = [s for s in load_samples(root, "train") if s["id"] in train_ids]
    out = art / "trajectories" / sp["name"]
    out.mkdir(parents=True, exist_ok=True)

    asr = None
    if a.asr:
        asr = load_cache(cache_path(art, backend_name(cfg), "train"))
        need = {p for s in samples for x in s["history"] if x["role"] == "user" for p in [x["audio"]]}
        miss = need - set(asr)
        if miss:
            sys.exit(f"train ASR cache incomplete for history clips: {len(miss)} missing (e.g. {sorted(miss)[0]})")

    st = collections.Counter()
    early = earlier_turns(samples, hidden, asr, st)
    with open(out / "train.earlier.jsonl", "w", encoding="utf-8") as f:
        for t in early:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[v2] earlier turns: {len(early)} -> {out / 'train.earlier.jsonl'} | {dict(sorted(st.items()))}")

    cards = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "synth/conv/cards").glob("batch_*/*.json")}
    st = collections.Counter()
    syn = synthetic(cards, hidden, train_ids, st)
    with open(out / "train.synth.jsonl", "w", encoding="utf-8") as f:
        for t in syn:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[v2] synthetic: {len(syn)} -> {out / 'train.synth.jsonl'} | {dict(sorted(st.items()))}")


if __name__ == "__main__":
    main()
