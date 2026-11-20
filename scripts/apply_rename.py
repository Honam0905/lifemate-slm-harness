"""Write renamed copies of training trajectory files (renaming augmentation, lifemate/data/rename.py).

    python scripts/apply_rename.py --config configs/v2r.yaml
Reads   artifacts/trajectories/<split>/train.<v>.jsonl         for v in rename.sources (default gold asr earlier synth)
Writes  artifacts/trajectories/<split>/train.<v>_r.jsonl       same examples, in the same order; a fraction of them
        (rename.fraction, chosen by a hash of the example) renamed with their own random plan, the rest unchanged.
Every renamed example is checked (calls, ask targets and argument names must match its own specs); a failure stops.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import load_catalog, tool_index  # noqa: E402
from lifemate.agent.schema import compact_tool  # noqa: E402
from lifemate.data.rename import Renamer, check_renamed, load_maps, seeded  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2r.yaml")
    ap.add_argument("--show", type=int, default=1, help="print N renamed examples (before -> after)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rc = cfg["rename"]
    art = artifacts_dir(cfg)
    tdir = art / "trajectories" / cfg["splits"]["name"]
    index = tool_index(load_catalog(data_root(cfg), "train"))
    maps = load_maps([ROOT / p for p in rc["maps"]])
    ren = Renamer(maps, index, rc.get("app_prob", 0.8), rc.get("param_prob", 0.5), rc.get("desc_prob", 0.5))
    frac, seed = float(rc.get("fraction", 0.5)), rc.get("seed", 11)
    shown = 0
    for v in rc.get("sources", ["gold", "asr", "earlier", "synth"]):
        src, dst = tdir / f"train.{v}.jsonl", tdir / f"train.{v}_r.jsonl"
        stats = collections.Counter()
        with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
            for line in f:
                t = json.loads(line)
                key = f"{t['id']}|{t.get('variant')}|{v}"
                if seeded(seed, "pick", key).random() < frac:
                    r = ren.trajectory(t, seeded(seed, "plan", key))
                    probs = check_renamed(r)
                    # only problems introduced by renaming count: a few gold calls already use an argument their
                    # schema does not have (e.g. calendar.list_events {date}); those pass through unchanged
                    before = check_renamed({**t, "tool_specs": [compact_tool(index[x]) for x in t["tools"]]})
                    if len(probs) > len(before):
                        sys.exit(f"[rename] {key}: {probs[:3]} (before renaming: {before[:3]})")
                    stats["inherited problems"] += len(before)
                    stats["renamed"] += 1
                    stats["apps renamed"] += len(r["renamed"]["apps"])
                    if shown < a.show and r["renamed"]["apps"] and t["steps"]:
                        shown += 1
                        print(f"[rename] example {key}: apps {r['renamed']['apps']}")
                        for s0, s1 in zip(t["steps"], r["steps"]):
                            print(f"    {json.dumps(s0['call'], ensure_ascii=False)}\n -> {json.dumps(s1['call'], ensure_ascii=False)}")
                    t = r
                else:
                    stats["unchanged"] += 1
                g.write(json.dumps(t, ensure_ascii=False) + "\n")
        n = stats["renamed"] + stats["unchanged"]
        print(f"[rename] train.{v}: {n} examples, {stats['renamed']} renamed "
              f"({stats['apps renamed'] / max(stats['renamed'], 1):.1f} apps each; {stats['inherited problems']} gold quirks "
              f"passed through unchanged) -> {dst}")


if __name__ == "__main__":
    main()
