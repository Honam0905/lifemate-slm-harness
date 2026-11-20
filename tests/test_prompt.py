"""Budget guard on the public split: every initial prompt (system + tool specs + context + full history + current
request, ASR transcripts) plus room for in-turn results plus max_new_tokens fits in 16,384 tokens with the real
tokenizer. Train/dev/audit full trajectories are checked by scripts/tokenize_trajectories.py. Run on the VM."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.agent.prompt import prefix_messages, render_generation_prompt  # noqa: E402
from lifemate.agent.schema import tool_specs  # noqa: E402
from lifemate.asr.base import backend_name  # noqa: E402
from lifemate.asr.cache import cache_path, load_cache  # noqa: E402
from lifemate.config import CTX_MAX, artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples, tool_index, user_turn_audio  # noqa: E402
from lifemate.data.trajectories import context_view, history_view, visible_tools  # noqa: E402

RESULT_RESERVE = 3000   # tokens kept free for this turn's tool results (train max full trajectory is 10.6K)


def test_public_prompts_fit_budget():
    from tests.runner import Skip
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise Skip("transformers not installed")
    cfg = load_config()
    ag = cfg["agent"]
    root, art = data_root(cfg), artifacts_dir(cfg)
    asr = load_cache(cache_path(art, backend_name(cfg), "public_test"))
    if not asr:
        raise Skip("no public ASR cache")
    tok = AutoTokenizer.from_pretrained(ag["model_id"], revision=ag["revision"])
    index = tool_index(load_catalog(root, "public_test"))
    lens = []
    for s in load_samples(root, "public_test"):
        tr = {p: asr[p] for p in user_turn_audio(s)}
        msgs = prefix_messages(context_view(s), history_view(s, tr), tr[s["audio"]])
        p = render_generation_prompt(tok, msgs, tool_specs(visible_tools(s), index))
        lens.append((len(tok(p, add_special_tokens=False)["input_ids"]), s["id"]))
    lens.sort()
    print(f"  public initial prompts: median {lens[len(lens) // 2][0]}, max {lens[-1]}")
    assert lens[-1][0] + RESULT_RESERVE + ag["max_new_tokens"] <= CTX_MAX, lens[-5:]


if __name__ == "__main__":
    from tests.runner import run
    run(globals())
