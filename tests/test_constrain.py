"""Constrained decoding never blocks a gold token, and does block invalid choices. Needs the tokenizer (VM)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.agent.constrain import RowSpec, Vocab, slot  # noqa: E402
from lifemate.config import data_root, load_config  # noqa: E402
from lifemate.controller.loop import apply_output, make_episode  # noqa: E402
from lifemate.data.loader import gold_transcripts, load_catalog, load_samples, tool_index, zaic_tools  # noqa: E402
from tests.test_controller import minicpm_xml  # noqa: E402

_c = {}


def _setup():
    if not _c:
        from tests.runner import Skip
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise Skip("transformers not installed")
        cfg = load_config()
        tok = AutoTokenizer.from_pretrained(cfg["agent"]["model_id"], revision=cfg["agent"]["revision"])
        _c.update(cfg=cfg, tok=tok, vocab=Vocab(tok))
    return _c


def test_gold_tokens_never_blocked():
    c = _setup()
    cfg, tok, vocab = c["cfg"], c["tok"], c["vocab"]
    root = data_root(cfg)
    zt = zaic_tools(root, "train")
    index = tool_index(load_catalog(root, "train"))
    n_pos = n_con = 0
    blocked = []
    for s in load_samples(root, "train"):
        ep = make_episode(s, zt, "train", root, index, gold_transcripts(s), cfg=cfg)
        for call in s["label"]["tool_calls"]:
            text = minicpm_xml(call)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            assert vocab.text(ids) == text.encode()
            row = RowSpec(ep.specs, set(ep.seen))
            for k in range(len(ids)):
                n_pos += 1
                sl = slot(vocab.text(ids[:k]), row)
                ok = vocab.allowed(*sl) if sl else None
                if ok:
                    n_con += 1
                    if ids[k] not in ok:
                        blocked.append((s["id"], call["name"], vocab.text(ids[:k])[-40:]))
            apply_output(ep, text, "minicpm_xml", max_calls=16)
    print(f"  {n_pos} gold token positions, {n_con} constrained, {len(blocked)} blocked: {sorted(set(blocked))[:8]}")
    assert n_con > 10000
    assert all(b[0] == "train_1696" for b in blocked), blocked[:5]   # the one gold ID with no visible source


def test_invalid_choices_blocked():
    c = _setup()
    vocab = c["vocab"]
    spec = {"parameters": {"properties": {"contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                                          "mode": {"type": "string", "enum": ["cool", "heat"]}}}}
    row = RowSpec({"sms.send": spec, "sys.ask_missing_info": spec}, {"c_08", "c_04"})
    for prefix, good, bad in [(b'<function name="', b'sms.send">', b'email.send">'),
                              (b'<function name="sms.send"><param name="', b'contact_id">', b'phone">'),
                              (b'<function name="sms.send"><param name="contact_id">', b"c_08</param>", b"c_99</param>"),
                              (b'<function name="sms.send"><param name="mode">', b"cool</param>", b"auto</param>")]:
        sl = slot(prefix, row)
        assert sl is not None, prefix
        assert sl[1] and good in sl[1] and bad not in sl[1]
    assert slot(b'<function name="sms.send"><param name="content">xin ch', row) is None   # free text stays free


if __name__ == "__main__":
    from tests.runner import run
    run(globals())
