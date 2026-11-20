"""Tokenise trajectories with an assistant-only loss mask.

The whole conversation is rendered once with the model's chat template and tokenised once (as inference does), and
the label mask is derived from character spans: for every assistant message k, the target is the text between the
generation prompt for messages[:k] and the end of message k (through <|im_end|>). Training and inference therefore
see byte-identical prefixes.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..agent.prompt import render_generation_prompt, trajectory_messages
from ..agent.schema import tool_specs

IGNORE = -100


@dataclass
class Encoded:
    id: str
    variant: str
    input_ids: list[int]
    labels: list[int]
    n_target: int
    boundary_misaligned: int     # target spans whose edges fall inside a token (should be 0)
    weight: float = 1.0          # per-trajectory loss weight (v2 loss_norm: trajectory)
    source: str = "final"        # final / earlier / synth


def render(tok, msgs, tools, **kw) -> str:
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, **kw)


def target_spans(tok, msgs: list[dict], tools: list[dict], end_marker: str) -> tuple[str, list[tuple[int, int]]]:
    full = render(tok, msgs, tools)
    spans = []
    for k, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prefix = render_generation_prompt(tok, msgs[:k], tools)   # the very string inference generates from
        upto = render(tok, msgs[: k + 1], tools)
        if not (upto.startswith(prefix) and full.startswith(upto)):
            raise ValueError("chat template is not prefix-stable; training and inference prefixes would differ")
        end = upto.rindex(end_marker) + len(end_marker)
        spans.append((len(prefix), end))
    return full, spans


def encode(tok, traj: dict, index: dict[str, dict], fmt: str, end_marker: str = "<|im_end|>",
           prompt_opts: dict | None = None) -> Encoded:
    msgs = trajectory_messages(traj, fmt, prompt_opts)
    # renamed trajectories (lifemate/data/rename.py) carry their own specs; everything else uses the catalog
    tools = traj["tool_specs"] if traj.get("tool_specs") else tool_specs(traj["tools"], index)
    full, spans = target_spans(tok, msgs, tools, end_marker)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    labels = [IGNORE] * len(ids)
    starts = {s for s, _ in offs}
    ends = {e for _, e in offs}
    bad = sum((s not in starts) + (e not in ends) for s, e in spans)
    for i, (s, e) in enumerate(offs):
        if e > s and any(a <= s and e <= b for a, b in spans):
            labels[i] = ids[i]
    return Encoded(traj["id"], traj["variant"], ids, labels, sum(l != IGNORE for l in labels), bad,
                   float(traj.get("weight", 1.0)), traj.get("source", "final"))


def decoded_targets(tok, e: Encoded) -> list[str]:
    """The labelled token runs, decoded — read these to confirm the mask covers exactly the actions."""
    runs, cur = [], []
    for t, l in zip(e.input_ids, e.labels):
        if l != IGNORE:
            cur.append(t)
        elif cur:
            runs.append(tok.decode(cur)); cur = []
    if cur:
        runs.append(tok.decode(cur))
    return runs
