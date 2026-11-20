"""Gold replay in the train ToolEnv: cached gold hashes plus whole-dataset self-checks of env + hash + scorer.

Train split only, in its own process. `get_state` is used here (and only here) to compare with label.final_state.

    python -m lifemate.eval.gold            # build artifacts/gold/train_hashes.json and run the self-checks
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples, zaic_tools  # noqa: E402
from lifemate.eval.metric import VARIANTS, final_score, fuzzy_keys, score_sample  # noqa: E402


def replay(zt, root: Path, sample: dict, calls: list[dict]):
    env = zt.ToolEnv("train", sample["id"], root=root)
    results = [env.call(c["name"], c["arguments"]) for c in calls]
    return env, results


def build_gold_hashes(zt, root: Path, samples: list[dict]) -> tuple[dict[str, str], list[str]]:
    hashes, problems = {}, []
    for s in samples:
        gold = s["label"]["tool_calls"]
        env, results = replay(zt, root, s, gold)
        if any(not r.get("ok") for r in results):
            problems.append(f"{s['id']}: a gold call was rejected")
        if results != s["label"]["tool_results"]:
            problems.append(f"{s['id']}: replayed results differ from label.tool_results")
        if env.get_tool_calls() != gold:
            problems.append(f"{s['id']}: get_tool_calls() differs from gold")
        state = env.get_state()
        for app, st in s["label"]["final_state"].items():
            if state.get(app) != st:
                problems.append(f"{s['id']}: final state of {app} differs")
        hashes[s["id"]] = env.get_state_hash()
    return hashes, problems


def hashes_path(cfg: dict) -> Path:
    return artifacts_dir(cfg) / "gold" / "train_hashes.json"


def load_or_build_hashes(cfg: dict) -> dict[str, str]:
    p = hashes_path(cfg)
    if p.exists():
        return json.loads(p.read_text())
    root = data_root(cfg)
    hashes, problems = build_gold_hashes(zaic_tools(root, "train"), root, load_samples(root, "train"))
    if problems:
        raise RuntimeError(f"gold replay problems: {problems[:5]}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hashes, indent=0, sort_keys=True))
    return hashes


def self_checks(zt, root: Path, samples: list[dict], hashes: dict[str, str], catalog: dict) -> dict:
    """Gold vs itself = 100 (all variants); empty submissions = 100*n_empty/N; gold + a duplicated first read scores
    exactly 2|G|/(2|G|+1) on every sample that has a read."""
    fz = fuzzy_keys(catalog)
    kinds = {t["name"]: t["kind"] for app in catalog["apps"] for t in app["tools"]}
    out = {}
    for v in VARIANTS:
        out[f"gold_self_{v}"] = final_score([score_sample(s["label"]["tool_calls"], hashes[s["id"]], s["label"]["tool_calls"],
                                                          hashes[s["id"]], fz, v) for s in samples])
    empty_scores = []
    for s in samples:
        env = zt.ToolEnv("train", s["id"], root=root)
        empty_scores.append(score_sample([], env.get_state_hash(), s["label"]["tool_calls"], hashes[s["id"]], fz))
    out["empty_submission"] = final_score(empty_scores)
    out["n_empty_gold"] = sum(not s["label"]["tool_calls"] for s in samples)
    bad, n = [], 0
    for s in samples:
        gold = s["label"]["tool_calls"]
        reads = [i for i, c in enumerate(gold) if kinds.get(c["name"]) == "read"]
        if not reads:
            continue
        n += 1
        i = reads[0]
        sub = gold[: i + 1] + [gold[i]] + gold[i + 1:]
        env, _ = replay(zt, root, s, sub)
        got = score_sample(env.get_tool_calls(), env.get_state_hash(), gold, hashes[s["id"]], fz)
        if got != Fraction(2 * len(gold), 2 * len(gold) + 1):
            bad.append((s["id"], str(got)))
    out["dup_read_samples"] = n
    out["dup_read_failures"] = bad
    return out


def main():
    cfg = load_config()
    root = data_root(cfg)
    zt = zaic_tools(root, "train")
    samples = load_samples(root, "train")
    hashes, problems = build_gold_hashes(zt, root, samples)
    print(f"[gold] replayed {len(samples)} samples, {len(problems)} problems")
    for p in problems[:10]:
        print("  ", p)
    p = hashes_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hashes, indent=0, sort_keys=True))
    print(f"[gold] hashes -> {p}")
    res = self_checks(zt, root, samples, hashes, load_catalog(root, "train"))
    print(json.dumps(res, indent=1))
    sys.exit(1 if problems or res["dup_read_failures"] else 0)


if __name__ == "__main__":
    main()
