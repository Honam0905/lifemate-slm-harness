"""Constrained decoding for the MiniCPM5 tool-call markup (a logits processor; greedy decoding stays greedy).

While the model is inside one of these slots, only tokens that continue a valid choice are allowed:
  <function name="…">           an installed tool (or sys.ask_missing_info)
  <param name="…">              a parameter declared by that tool
  value of a string enum param  one of the enum values
  value of a string x-ref param an ID-like value already seen in this sample (history or earlier results);
                                gold uses an unseen one in 1 of 1,278 cases
  ask_missing_info.tool_name    an installed tool
Everywhere else generation is free. If the text already left every valid choice the row is left unconstrained.
Works on bytes (the tokenizer is byte-level BPE), so partial UTF-8 tokens are handled exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ASK = "sys.ask_missing_info"
_FUNC_OPEN = re.compile(rb'<function name="([^"]*)\Z')
_FUNC_NAME = re.compile(rb'<function name="([^"]+)">')
_PARAM_OPEN = re.compile(rb'<param name="([^"]*)\Z')
_PARAM_VAL = re.compile(rb'<param name="([^"]+)">((?:(?!</param>)[\s\S])*)\Z')


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte <-> printable-unicode table used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Vocab:
    """Byte surface of every token, and an exact-bytes -> token ids index."""

    PREFIX = 4

    def __init__(self, tok):
        dec = {c: b for b, c in _bytes_to_unicode().items()}
        special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}) or {})
        n = len(tok)
        self.size = n
        self.surface: list[bytes] = []
        for i in range(n):
            s = tok.convert_ids_to_tokens(i)
            if s is None:
                self.surface.append(b"")
            elif i in special or any(ch not in dec for ch in s):
                self.surface.append(s.encode("utf-8"))
            else:
                self.surface.append(bytes(dec[ch] for ch in s))
        self.by_bytes: dict[bytes, list[int]] = {}
        self.by_head: dict[bytes, list[int]] = {}       # first PREFIX bytes -> tokens (for tokens that run past a slot)
        for i, b in enumerate(self.surface):
            if b:
                self.by_bytes.setdefault(b, []).append(i)
                if len(b) > 1:
                    self.by_head.setdefault(b[:self.PREFIX], []).append(i)

    def text(self, ids: list[int]) -> bytes:
        return b"".join(self.surface[i] for i in ids if i < self.size)

    def allowed(self, partial: bytes, completions: list[bytes]) -> set[int]:
        """Tokens that extend `partial` along some completion (each completion includes its closing marker)."""
        out: set[int] = set()
        for c in completions:
            if not c.startswith(partial):
                continue
            rest = c[len(partial):]
            for k in range(1, len(rest) + 1):
                out.update(self.by_bytes.get(rest[:k], ()))
            # a token may finish the choice and run on (e.g. '">' + '[{"' merges into '">[{"' before a JSON value)
            heads = [rest[:self.PREFIX]] if len(rest) >= self.PREFIX else [h for h in self.by_head if h.startswith(rest)]
            for h in heads:
                out.update(i for i in self.by_head.get(h, ()) if len(self.surface[i]) > len(rest)
                           and self.surface[i].startswith(rest))
        return out


@dataclass
class RowSpec:
    """What is valid for one episode: its tools (name -> catalog spec) and the ID-like values seen so far."""
    tools: dict
    seen: set = field(default_factory=set)


def slot(gen: bytes, row: RowSpec) -> tuple[bytes, list[bytes]] | None:
    """(partial text of the open slot, its valid completions) or None when generation is free."""
    m = _FUNC_OPEN.search(gen)
    if m:
        return m.group(1), [n.encode() + b'">' for n in row.tools]
    fm = list(_FUNC_NAME.finditer(gen))
    if not fm:
        return None
    name = fm[-1].group(1).decode("utf-8", "replace")
    spec = row.tools.get(name)
    body = gen[fm[-1].end():]
    if spec is None or b"</function>" in body:
        return None
    props = spec["parameters"].get("properties", {})
    m = _PARAM_OPEN.search(body)
    if m:
        return m.group(1), [k.encode() + b'">' for k in props]
    m = _PARAM_VAL.search(body)
    if not m:
        return None
    key = m.group(1).decode("utf-8", "replace")
    p = props.get(key, {})
    if name == ASK and key == "tool_name":
        values = [t for t in row.tools if t != ASK]
    elif p.get("type") == "string" and p.get("enum"):
        values = [str(v) for v in p["enum"]]
    elif p.get("type") == "string" and p.get("x-ref") and row.seen:
        values = sorted(row.seen)
    else:
        return None
    return m.group(2), [v.encode() + b"</param>" for v in values]


class ToolCallConstraint:
    """HF logits processor. `prompt_len`: where generated tokens start in input_ids (same for every row)."""

    def __init__(self, vocab: Vocab, rows: list[RowSpec], prompt_len: int):
        self.vocab, self.rows, self.prompt_len = vocab, rows, prompt_len
        self.n_constrained = 0

    def __call__(self, input_ids, scores):
        import torch  # torch-free at import time, so the controller stays importable by the stdlib-only tests
        for r, row in enumerate(self.rows):
            gen = self.vocab.text(input_ids[r, self.prompt_len:].tolist())
            s = slot(gen, row)
            if s is None:
                continue
            ok = self.vocab.allowed(*s)
            if not ok:
                continue
            mask = torch.full_like(scores[r], float("-inf"))
            mask[torch.tensor(sorted(ok), device=scores.device)] = 0
            scores[r] = scores[r] + mask
            self.n_constrained += 1
        return scores


_ID_MAX = 80


def collect_values(obj, out: set) -> None:
    """ID-like string leaves (no whitespace, short) of tool results / call arguments."""
    if isinstance(obj, str):
        if obj and len(obj) <= _ID_MAX and not any(c.isspace() for c in obj):
            out.add(obj)
    elif isinstance(obj, list):
        for x in obj:
            collect_values(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            collect_values(x, out)
