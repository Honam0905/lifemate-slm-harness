"""Print a fingerprint of rendered prompts + token ids for some trajectories, to compare two environments
(e.g. the training venv vs the inference/ASR venv). Identical output = byte-identical prompts and tokens."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.agent.prompt import render_generation_prompt, trajectory_messages  # noqa: E402
from lifemate.agent.schema import tool_specs  # noqa: E402
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import load_catalog, tool_index  # noqa: E402


def main():
    import transformers
    from transformers import AutoTokenizer
    cfg = load_config()
    ag = cfg["agent"]
    tok = AutoTokenizer.from_pretrained(ag["model_id"], revision=ag["revision"])
    index = tool_index(load_catalog(data_root(cfg), "train"))
    path = artifacts_dir(cfg) / "trajectories" / cfg["splits"]["name"] / "dev.asr.jsonl"
    h_text, h_ids, n = hashlib.sha256(), hashlib.sha256(), 0
    for line in list(open(path, encoding="utf-8"))[:60]:
        t = json.loads(line)
        msgs, tools = trajectory_messages(t, ag["tool_format"]), tool_specs(t["tools"], index)
        for k, m in enumerate(msgs):
            if m["role"] == "assistant":
                p = render_generation_prompt(tok, msgs[:k], tools)
                h_text.update(p.encode()); h_ids.update(json.dumps(tok(p, add_special_tokens=False)["input_ids"]).encode())
                n += 1
    print(f"transformers {transformers.__version__}: {n} prompts  text {h_text.hexdigest()[:16]}  ids {h_ids.hexdigest()[:16]}")


if __name__ == "__main__":
    main()
