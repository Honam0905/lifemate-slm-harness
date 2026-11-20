"""ASR interface. Backends return plain strings; nothing downstream knows which backend produced them."""
from __future__ import annotations

from typing import Protocol


class ASR(Protocol):
    name: str       # backend id used in cache keys, e.g. "qwen3_asr-7278e1e"

    def transcribe(self, paths: list[str], contexts: list[str] | None = None) -> list[str]:
        """Transcribe audio files, preserving order and boundaries (one string per path). `contexts`: optional
        biasing text per path (see asr/context.py)."""
        ...


def make_asr(cfg: dict) -> ASR:
    a = cfg["asr"]
    if a["backend"] == "qwen3_asr":
        from .qwen3_asr import Qwen3ASR
        if a.get("model_path"):   # a fine-tuned, merged ASR saved locally (relative paths are under the artifacts dir)
            from pathlib import Path
            from ..config import artifacts_dir
            p = Path(a["model_path"])
            p = p if p.is_absolute() else artifacts_dir(cfg) / p
            return Qwen3ASR(str(p), None, a.get("language"), a["batch_size"], a["max_new_tokens"], tag=a.get("tag"))
        return Qwen3ASR(a["model_id"], a["revision"], a.get("language"), a["batch_size"], a["max_new_tokens"])
    raise ValueError(f"unknown asr backend {a['backend']!r}")


def backend_name(cfg: dict) -> str:
    """Cache key of the transcripts: backend, model, revision, fine-tuned tag, and whether context biasing was used."""
    a = cfg["asr"]
    if a.get("model_path") and not a.get("tag"):
        raise ValueError("asr.model_path needs asr.tag so its transcripts get their own cache")
    return (f"{a['backend']}-{a['model_id'].split('/')[-1]}-{a['revision'][:7]}" + (f"-{a['tag']}" if a.get("model_path") else "")
            + (("-ctxnames" if a.get("context_mode") == "names" else "-ctx") if a.get("context") else ""))
