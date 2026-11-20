"""Oracle rollouts without a model: each gold call, written the way MiniCPM5's template writes a tool call, goes
through parse -> clean -> loop guard -> validate -> call. Every train sample must reproduce its gold calls and gold
hash exactly (score 1), so the controller never alters a correct action."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.config import data_root, load_config  # noqa: E402
from lifemate.controller.loop import apply_output, make_episode  # noqa: E402
from lifemate.data.loader import gold_transcripts, load_catalog, load_samples, tool_index, zaic_tools  # noqa: E402
from lifemate.eval.gold import load_or_build_hashes  # noqa: E402
from lifemate.eval.metric import fuzzy_keys, score_sample  # noqa: E402


def minicpm_xml(call: dict) -> str:
    """Python mirror of the MiniCPM5 chat template's tool-call rendering (checked against the real template on the VM
    by tests/test_actions.py)."""
    out = f'<function name="{call["name"]}">'
    for k, v in (call.get("arguments") or {}).items():
        t = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        if "<" in t or "&" in t or "\n" in t:
            t = f"<![CDATA[{t}]]>"
        out += f'<param name="{k}">{t}</param>'
    return out + "</function>"


def test_oracle_rollouts_reproduce_gold():
    _oracle(load_config())


def test_oracle_rollouts_with_all_harness_options():
    cfg = load_config()
    cfg["prompt"] = {**cfg["prompt"], "date_table": True}
    cfg["harness"] = {**cfg["harness"], "coerce_types": True}
    _oracle(cfg)


def _oracle(cfg):
    root = data_root(cfg)
    zt = zaic_tools(root, "train")
    cat = load_catalog(root, "train")
    index, fz = tool_index(cat), fuzzy_keys(cat)
    hashes = load_or_build_hashes(cfg)
    bad = []
    for s in load_samples(root, "train"):
        ep = make_episode(s, zt, "train", root, index, gold_transcripts(s), cfg=cfg)
        for c in s["label"]["tool_calls"]:
            if ep.done:
                break
            apply_output(ep, minicpm_xml(c), "minicpm_xml", max_calls=16)
        if not ep.done:
            apply_output(ep, "DONE", "minicpm_xml", max_calls=16)
        sub, h = ep.env.get_tool_calls(), ep.env.get_state_hash()
        sc = score_sample(sub, h, s["label"]["tool_calls"], hashes[s["id"]], fz)
        if sub != s["label"]["tool_calls"] or sc != 1:
            notes = [n for st in ep.steps for n in st["notes"]]
            bad.append((s["id"], ep.stop_reason, float(sc), notes[:3]))
        coerced = [n for st in ep.steps for n in st["notes"] if n.startswith("coerced")]
        assert not coerced, f"{s['id']}: type repair changed a gold argument: {coerced}"
    assert not bad, f"{len(bad)} samples: {bad[:8]}"


if __name__ == "__main__":
    from tests.runner import run
    run(globals())
