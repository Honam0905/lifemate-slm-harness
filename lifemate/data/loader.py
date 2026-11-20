"""Split files, catalogs, audio paths, and the split's own zaic_tools (one split per process)."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

_LOADED_SPLIT: str | None = None


def load_samples(root: Path, split: str) -> list[dict]:
    with open(root / split / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_catalog(root: Path, split: str) -> dict:
    with open(root / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        return json.load(f)


def tool_index(catalog: dict) -> dict[str, dict]:
    """tool name -> tool spec"""
    return {t["name"]: t for app in catalog["apps"] for t in app["tools"]}


def zaic_tools(root: Path, split: str):
    """Import <root>/<split>/zaic_tools. Train and public ship different copies under the same package name,
    and a second import would silently return the first, so a process may load only one split."""
    global _LOADED_SPLIT
    if _LOADED_SPLIT is not None and _LOADED_SPLIT != split:
        raise RuntimeError(f"zaic_tools for split {_LOADED_SPLIT!r} already loaded; use a separate process for {split!r}")
    if _LOADED_SPLIT is None:
        sys.path.insert(0, str(Path(root) / split))
        _LOADED_SPLIT = split
    mod = importlib.import_module("zaic_tools")
    if Path(mod.__file__).resolve().parent != (Path(root) / split / "zaic_tools").resolve():
        raise RuntimeError(f"imported zaic_tools from {mod.__file__}, expected the {split} copy")
    return mod


def user_turn_audio(sample: dict) -> list[str]:
    """Relative audio paths of every user turn, history first, the final request last."""
    return [t["audio"] for t in sample["history"] if t["role"] == "user"] + [sample["audio"]]


def gold_transcripts(sample: dict) -> dict[str, str]:
    """audio path -> annotator transcript (train only)."""
    out = {t["audio"]: t["audio_text"] for t in sample["history"] if t["role"] == "user"}
    out[sample["audio"]] = sample["audio_text"]
    return out


def installed_tools(sample: dict) -> list[str]:
    return [t for app in sample["context"]["installed_apps"].values() for t in app["tools"]]
