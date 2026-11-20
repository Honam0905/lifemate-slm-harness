"""Parameter inventory of the whole system, counted from the tensors in the checkpoints actually loaded.

    python scripts/param_inventory.py --llm artifacts/models/v1-epoch2 [--asr Qwen/Qwen3-ASR-1.7B --asr_revision ...]
Fails (exit 1) unless total < 6B and LLM < 5B.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.config import load_config  # noqa: E402


def count_dir(d: Path) -> int:
    total = 0
    for f in sorted(d.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for k, v in header.items():
            if k != "__metadata__":
                c = 1
                for s in v["shape"]:
                    c *= s
                total += c
    return total


def resolve(name: str, revision: str | None) -> Path:
    p = Path(name)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name, revision=revision, allow_patterns=["*.safetensors", "*.json"]))


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=cfg["agent"]["model_id"])
    ap.add_argument("--llm_revision", default=cfg["agent"]["revision"])
    ap.add_argument("--asr", default=cfg["asr"]["model_id"])
    ap.add_argument("--asr_revision", default=cfg["asr"]["revision"])
    a = ap.parse_args()
    llm = count_dir(resolve(a.llm, a.llm_revision if not Path(a.llm).is_dir() else None))
    asr = count_dir(resolve(a.asr, a.asr_revision))
    b = cfg["budget"]
    total = llm + asr
    print(f"LLM  {a.llm}: {llm:,}\nASR  {a.asr}: {asr:,}\nTOTAL: {total:,}  (limits: total < {b['params_total_max']:,}, LLM < {b['params_llm_max']:,})")
    ok = total < b["params_total_max"] and llm < b["params_llm_max"]
    print("OK" if ok else "OVER BUDGET")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
