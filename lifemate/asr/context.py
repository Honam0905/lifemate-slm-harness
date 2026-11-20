"""Optional ASR context (biasing text) built only from the sample's own inputs.

The history's assistant replies name the entities the user is talking about ("Vợ Hằng", "Sếp Dũng", "CineViệt");
handing them to the ASR as context helps it spell those names. The same context is used for every clip of a sample.

Modes:
  full   the profile name/city and every history assistant reply, verbatim (measured 4 Nov: Qwen3-ASR copies those
         replies into the user's transcript, dev WER 28.1% -> 31.1%)
  names  only a short comma-separated list of names: the profile name/city and the capitalised word runs in the
         history assistant replies (a sentence-initial single capitalised word is skipped: "Dạ", "Tài khoản", ...)
"""
from __future__ import annotations

import re

MAX_CHARS = 1500
MAX_NAME_CHARS = 300
_SENT = re.compile(r"(?<=[.!?…:;])\s+|\n+")
_STRIP = ".,!?…:;\"'()[]{}“”‘’«»"


def _is_cap(tok: str) -> bool:
    return bool(tok) and tok[0].isupper()


def _names(text: str) -> list[str]:
    """Runs of capitalised tokens; a one-token run at the start of a sentence is ordinary capitalisation."""
    out = []
    for sent in _SENT.split(text):
        toks = [t.strip(_STRIP) for t in sent.split()]
        i = 0
        while i < len(toks):
            if not _is_cap(toks[i]):
                i += 1
                continue
            j = i
            while j < len(toks) and _is_cap(toks[j]):
                j += 1
            run = [t for t in toks[i:j] if t]
            if run and not (i == 0 and len(run) == 1):
                out.append(" ".join(run))
            i = j
    return out


def asr_context(sample: dict, mode: str = "full") -> str:
    parts, seen = [], set()
    prof = sample["context"].get("user_profile") or {}
    for v in (prof.get("name"), prof.get("city")):
        if v and v not in seen:
            parts.append(v); seen.add(v)
    for t in sample["history"]:
        txt = (t.get("text") or "").strip() if t["role"] == "assistant" else ""
        if not txt:
            continue
        for v in (_names(txt) if mode == "names" else [txt]):
            if v not in seen:
                parts.append(v); seen.add(v)
    if mode == "names":
        return ", ".join(parts)[:MAX_NAME_CHARS]
    out = "\n".join(parts)
    return out[-MAX_CHARS:]
