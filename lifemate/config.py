"""Config loading. Everything tunable lives in configs/<name>.yaml; a few hard limits are constants here."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CTX_MAX = 16384          # LLM context limit (competition rule)
SAMPLE_BUDGET_S = 30     # wall-clock per sample (competition rule)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path = "configs/v1.yaml") -> dict:
    """Load a YAML config. A `base: other.yaml` key loads that file first and deep-merges this one over it."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if "base" in cfg:
        base = load_config(p.parent / cfg.pop("base"))
        base.pop("_path", None)
        cfg = _merge(base, cfg)
    cfg["_path"] = str(p)
    return cfg


def data_root(cfg: dict) -> Path:
    r = Path(os.environ.get("LIFEMATE_DATA_ROOT") or cfg["data"]["root"])
    return r if r.is_absolute() else PROJECT_ROOT / r


def artifacts_dir(cfg: dict) -> Path:
    r = Path(os.environ.get("LIFEMATE_ARTIFACTS") or cfg["data"]["artifacts"])
    r = r if r.is_absolute() else PROJECT_ROOT / r
    r.mkdir(parents=True, exist_ok=True)
    return r
