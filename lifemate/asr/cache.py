"""Transcript cache: artifacts/asr/<backend>/<split>.jsonl, one {"audio": <relative path>, "text": ...} per clip."""
from __future__ import annotations

import json
from pathlib import Path


def cache_path(artifacts: Path, backend: str, split: str) -> Path:
    return artifacts / "asr" / backend / f"{split}.jsonl"


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["audio"]] = r["text"]
    return out


def append_cache(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for audio, text in rows:
            f.write(json.dumps({"audio": audio, "text": text}, ensure_ascii=False) + "\n")
