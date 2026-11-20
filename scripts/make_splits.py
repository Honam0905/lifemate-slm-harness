"""Build the grouped train/dev/audit split once and freeze it to artifacts/splits/<name>.json (refuses to overwrite)."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import load_samples  # noqa: E402
from lifemate.data.splits import apps_used, make_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = artifacts_dir(cfg) / "splits" / f"{cfg['splits']['name']}.json"
    if out.exists() and not a.force:
        sys.exit(f"{out} exists; splits are frozen (use --force only if you really mean to re-split)")
    samples = load_samples(data_root(cfg), "train")
    sp = make_splits(samples, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sp, ensure_ascii=False, indent=1), encoding="utf-8")
    by_id = {s["id"]: s for s in samples}
    unseen = set(sp["unseen_apps"])
    for part in ("train", "dev", "audit"):
        ids = sp[part]
        kinds = collections.Counter()
        for i in ids:
            calls = by_id[i]["label"]["tool_calls"]
            kinds["empty" if not calls else "ask" if calls[-1]["name"] == "sys.ask_missing_info" else "act"] += 1
        leak = sum(1 for i in ids if apps_used(by_id[i]) & unseen)
        print(f"{part:6s} n={len(ids):4d} uses-unseen={leak:3d} {dict((k, v) for k, v in kinds.items() if v)}")
    assert not set(sp["train"]) & (set(sp["dev"]) | set(sp["audit"])) and not set(sp["dev"]) & set(sp["audit"])
    assert not any(apps_used(by_id[i]) & unseen for i in sp["train"]), "unseen app leaked into train"
    print(f"unseen_dev={len(sp['unseen_dev'])} unseen_audit={len(sp['unseen_audit'])} sha256={sp['sha256'][:16]} -> {out}")


if __name__ == "__main__":
    main()
