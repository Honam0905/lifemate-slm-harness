"""One summary line for a dev run against one or more baselines (paired, same samples), plus a JSON record.

    python cloud/compare.py NAME ROWS --vs v2r-molab=ROWS --vs v1-colab=ROWS --out results/NAME/paired.json
The first --vs is the decision baseline (cloud/molab_job.sh picks the recipe with the largest mean_diff against it).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.eval.report import load_rows, paired, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("rows")
    ap.add_argument("--vs", action="append", default=[], help="LABEL=rows.jsonl (first = decision baseline)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = load_rows(a.rows)
    s = summarize(rows)
    d = s["diagnostics"]
    empty = sum(1 for r in rows if r["n_sub"] == 0)
    right = sum(1 for r in rows if r["n_sub"] == 0 and r["n_gold"] == 0)
    apps = s["slices"]["apps"]
    rec = {"name": a.name, "n": s["n"], "primary": s["primary"], "ci95": s["primary_ci95"],
           "known": apps.get("known", {}).get("primary"), "unseen": apps.get("unseen", {}).get("primary"),
           "empty_answers": empty, "empty_correct": right, "stops": d["stop_reasons"], "vs": {}}
    line = (f"{a.name}: dev C {s['primary']} (n {s['n']}) | known {rec['known']} unseen {rec['unseen']} | "
            f"empty answers {empty} ({right} right) | stops {d['stop_reasons']}")
    for v in a.vs:
        label, path = v.split("=", 1)
        if not Path(path).exists():
            line += f" | vs {label}: (no rows)"
            continue
        base = {r["id"]: r for r in load_rows(path)}
        p = paired(list(base.values()), rows)
        won = sum(1 for r in rows if r["id"] in base and r["score_primary"] > base[r["id"]]["score_primary"] + 1e-9)
        lost = sum(1 for r in rows if r["id"] in base and r["score_primary"] < base[r["id"]]["score_primary"] - 1e-9)
        rec["vs"][label] = {**p, "won": won, "lost": lost}
        line += f" | vs {label}: {p['mean_diff']:+} CI {list(p['ci95'])} won {won} lost {lost}"
    print(line, flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
