"""Every gold call rendered by the REAL chat template parses back to exactly the same call (type-aware), and the
template's text equals the Python mirror used by test_controller. Needs transformers + the tokenizer (run on the VM)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.agent.actions import call_message, parse_action  # noqa: E402
from lifemate.agent.prompt import render_generation_prompt  # noqa: E402
from lifemate.agent.schema import param_types  # noqa: E402
from lifemate.config import data_root, load_config  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples, tool_index  # noqa: E402
from lifemate.eval.metric import same  # noqa: E402
from tests.test_controller import minicpm_xml  # noqa: E402


def test_gold_calls_roundtrip_through_real_template():
    from tests.runner import Skip
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise Skip("transformers not installed")
    cfg = load_config()
    ag = cfg["agent"]
    tok = AutoTokenizer.from_pretrained(ag["model_id"], revision=ag["revision"])
    root = data_root(cfg)
    index = tool_index(load_catalog(root, "train"))
    types_of = {n: param_types(s) for n, s in index.items()}
    base = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    prefix = render_generation_prompt(tok, base, [])
    n, bad = 0, []
    for s in load_samples(root, "train"):
        for c in s["label"]["tool_calls"]:
            full = tok.apply_chat_template(base + [call_message(c, ag["tool_format"])], tokenize=False)
            assert full.startswith(prefix)
            text = full[len(prefix): full.rindex("<|im_end|>")]
            act = parse_action(text, ag["tool_format"], types_of)
            ok = act.kind == "call" and act.name == c["name"] and same(act.arguments, c["arguments"], lenient=False)
            if not ok or text != minicpm_xml(c):
                bad.append((s["id"], c, text, act.arguments))
            n += 1
    assert n > 3000 and not bad, f"{len(bad)}/{n} bad, first: {bad[:2]}"


if __name__ == "__main__":
    from tests.runner import run
    run(globals())
