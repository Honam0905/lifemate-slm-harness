"""Compare dev rows of several arms against a baseline arm: score, paired difference, samples won/lost, points lost
per failure category, and the decision slices.

    python3 -I scripts/compare_arms.py BASE_ROWS ARM_ROWS [ARM_ROWS ...]      (rows.jsonl from lifemate.eval.run_eval)
Arm names are taken from the eval directory (…/eval/<name>/dev.C/rows.jsonl).
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lifemate.config import data_root, load_config  # noqa: E402
from lifemate.data.loader import load_catalog  # noqa: E402
from lifemate.eval.report import paired, summarize  # noqa: E402

ASK = "sys.ask_missing_info"
catalog = load_catalog(data_root(load_config(ROOT / "configs/v1.yaml")), "train")
kinds = {t["name"]: t.get("kind") for app in catalog["apps"] for t in app["tools"]}


def category(r: dict) -> str:
    sub, ug, us = r["submitted"], r["unmatched_gold"], r["unmatched_sub"]
    if r["gold_type"] == "empty":
        return "acted on out-of-scope"
    sub_ask = bool(sub) and sub[-1]["name"] == ASK
    if r["gold_type"] == "ask":
        return "ask: wrong tool/params" if sub_ask else "ask: did not ask"
    if sub_ask:
        return "act: asked instead"
    if not sub:
        return "act: returned []"
    if not r["hash_ok"]:
        ew = [c for c in us if kinds.get(c["name"]) == "write"]
        mw = [c for c in ug if kinds.get(c["name"]) == "write"]
        if ew and mw and {c["name"] for c in ew} & {c["name"] for c in mw}:
            return "state: wrong args"
        if ew and mw:
            return "state: wrong write tool"
        return "state: extra write" if ew else "state: missing write" if mw else "state: other"
    return "state ok: F1 < 1"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


base = load(sys.argv[1])
arms = {Path(sys.argv[1]).parent.parent.name: base, **{Path(p).parent.parent.name: load(p) for p in sys.argv[2:]}}
cats = sorted({category(r) for rows in arms.values() for r in rows if r["score_primary"] < 1})
b = {r["id"]: r["score_primary"] for r in base}
print(f"{'arm':22s} {'score':>6s} {'diff':>6s} {'95% CI':>16s} {'won':>4s} {'lost':>4s} {'ask acc':>7s} {'noop':>5s} {'unseen':>6s} {'known':>6s}")
for name, rows in arms.items():
    s, pr = summarize(rows), paired(base, rows)
    won = sum(r["score_primary"] > b[r["id"]] for r in rows)
    lost = sum(r["score_primary"] < b[r["id"]] for r in rows)
    sl = s["slices"]["apps"]
    print(f"{name:22s} {s['primary']:6.2f} {pr['mean_diff']:+6.2f} {str(pr['ci95']):>16s} {won:4d} {lost:4d} "
          f"{s['diagnostics']['ask_accuracy']:7.2f} {s['diagnostics']['noop_accuracy']:5.1f} "
          f"{sl['unseen']['primary']:6.2f} {sl['known']['primary']:6.2f}")
print("\npoints lost by category:")
print(f"{'':26s}" + "".join(f"{n[:14]:>15s}" for n in arms))
for c in cats:
    vals = [100 * sum(1 - r["score_primary"] for r in rows if r["score_primary"] < 1 and category(r) == c) / len(rows)
            for rows in arms.values()]
    print(f"{c:26s}" + "".join(f"{v:15.2f}" for v in vals))
