"""Summaries of evaluation rows: three scorer variants, slices, diagnostics, bootstrap CIs over episode groups,
and paired comparisons between two runs.

    python -m lifemate.eval.report ROWS.jsonl [--vs OTHER_ROWS.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics

VARIANTS = ("primary", "pessimistic", "optimistic")


def hist_bucket(n_turns: int) -> str:
    return "0" if n_turns == 0 else "2-6" if n_turns <= 6 else "8+"


def _mean(xs) -> float | None:
    xs = list(xs)
    return round(100 * sum(xs) / len(xs), 2) if xs else None


def bootstrap(rows: list[dict], key: str = "score_primary", n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean score, resampling episode groups (rows carry a 'group' id)."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r[key])
    gs = list(groups.values())
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        pick = [x for _ in gs for x in rng.choice(gs)]
        means.append(100 * sum(pick) / len(pick))
    means.sort()
    return round(means[int(0.025 * n)], 2), round(means[int(0.975 * n) - 1], 2)


def summarize(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for v in VARIANTS:
        out[v] = _mean(r[f"score_{v}"] for r in rows)
    out["primary_ci95"] = bootstrap(rows)
    slices = {}
    for name, key in (("apps", "app_slice"), ("gold_type", "gold_type"), ("history", "hist"), ("n_gold", "n_gold")):
        by = collections.defaultdict(list)
        for r in rows:
            by[str(r[key])].append(r["score_primary"])
        slices[name] = {k: {"n": len(v), "primary": _mean(v)} for k, v in sorted(by.items())}
    out["slices"] = slices
    nonempty = [r for r in rows if r["n_gold"]]
    write = [r for r in rows if r["gold_type"] == "write"]
    hash_ok = [r for r in nonempty if r["hash_ok"]]
    asks = [r for r in rows if r["gold_type"] == "ask"]
    empties = [r for r in rows if r["gold_type"] == "empty"]
    out["diagnostics"] = {
        "hash_agreement": _mean(r["hash_ok"] for r in nonempty),
        "hash_agreement_write": _mean(r["hash_ok"] for r in write),
        "f1_given_hash": _mean(r["score_primary"] for r in hash_ok),
        "ask_accuracy": _mean(not any(g["name"] == "sys.ask_missing_info" for g in r["unmatched_gold"]) for r in asks),
        "noop_accuracy": _mean(r["n_sub"] == 0 for r in empties),
        "extra_calls_per_sample": round(statistics.mean(len(r["unmatched_sub"]) for r in rows), 3) if rows else None,
        "missing_calls_per_sample": round(statistics.mean(len(r["unmatched_gold"]) for r in rows), 3) if rows else None,
        "stop_reasons": dict(collections.Counter(r["stop_reason"] for r in rows)),
        "mean_calls": round(statistics.mean(r["n_sub"] for r in rows), 3) if rows else None,
        "max_prompt_tokens": max((r.get("prompt_tokens_max", 0) for r in rows), default=0),
    }
    return out


def paired(rows_a: list[dict], rows_b: list[dict], key: str = "score_primary", n: int = 1000, seed: int = 0) -> dict:
    """Mean of (b - a) per sample with a 95% group-bootstrap interval. Rows are matched by id."""
    a = {r["id"]: r for r in rows_a}
    diffs = [{"group": r["group"], "d": r[key] - a[r["id"]][key]} for r in rows_b if r["id"] in a]
    lo, hi = bootstrap(diffs, key="d", n=n, seed=seed)
    return {"n": len(diffs), "mean_diff": _mean(d["d"] for d in diffs), "ci95": (lo, hi)}


def load_rows(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--vs", default=None, help="baseline rows file for a paired comparison (this run minus baseline)")
    a = ap.parse_args()
    rows = load_rows(a.rows)
    print(json.dumps(summarize(rows), indent=1, ensure_ascii=False))
    if a.vs:
        print(json.dumps({"paired_vs_baseline": paired(load_rows(a.vs), rows)}, indent=1))


if __name__ == "__main__":
    main()
