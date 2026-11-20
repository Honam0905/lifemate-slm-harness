"""Make a text-only Qwen3.5-4B: drop the vision tower (model.visual.*) and the MTP head (mtp.*), keep the language model.

    python scripts/strip_qwen35_vision.py --out /content/models/Qwen3.5-4B-text [--verify]

Weights are renamed model.language_model.* -> model.* and saved with a Qwen3_5ForCausalLM config built from the
original text_config, so `AutoModelForCausalLM.from_pretrained(out)` loads only text tensors. --verify loads the
stripped model, counts the parameters actually loaded, and checks that its next-token logits on a text prompt match
the original multimodal checkpoint's text path.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DROP_FILES = {"preprocessor_config.json", "video_preprocessor_config.json", "model.safetensors.index.json", "config.json"}


def strip(out: Path) -> None:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(snapshot_download(REPO, revision=REVISION))
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    text = dict(cfg["text_config"])
    text["architectures"] = ["Qwen3_5ForCausalLM"]
    text["model_type"] = "qwen3_5_text"
    text["tie_word_embeddings"] = cfg.get("tie_word_embeddings", text.get("tie_word_embeddings", True))
    text["mtp_num_hidden_layers"] = 0          # MTP head removed
    text["torch_dtype"] = "bfloat16"
    (out / "config.json").write_text(json.dumps(text, indent=2))

    kept, dropped = {}, {"visual": 0, "mtp": 0}
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    dropped["visual"] += _numel(f.get_slice(k).get_shape())
                elif k.startswith("mtp."):
                    dropped["mtp"] += _numel(f.get_slice(k).get_shape())
                elif k.startswith("model.language_model."):
                    kept["model." + k[len("model.language_model."):]] = f.get_tensor(k)
                elif k == "lm_head.weight" and not text["tie_word_embeddings"]:
                    kept[k] = f.get_tensor(k)
                else:
                    raise KeyError(f"unexpected tensor {k}")
    n = sum(t.numel() for t in kept.values())
    # two shards keep each file small enough for quick uploads
    keys = sorted(kept)
    half = len(keys) // 2
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in kept.values())}, "weight_map": {}}
    for i, part in enumerate((keys[:half], keys[half:])):
        name = f"model-{i + 1:05d}-of-00002.safetensors"
        save_file({k: kept[k].contiguous() for k in part}, str(out / name), metadata={"format": "pt"})
        index["weight_map"].update({k: name for k in part})
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name not in DROP_FILES:
            shutil.copy(f, out / f.name)
    (out / "STRIPPED.md").write_text(
        f"Text-only {REPO}@{REVISION}: model.visual.* ({dropped['visual']:,} params) and mtp.* ({dropped['mtp']:,} params) "
        f"removed; {n:,} params kept (embeddings tied).\n")
    print(f"[strip] kept {n:,} params; dropped vision {dropped['visual']:,}, mtp {dropped['mtp']:,} -> {out}")


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def verify(out: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(out)
    msgs = [{"role": "user", "content": "Đặt báo thức 6 giờ sáng mai giúp tôi nhé."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").input_ids.cuda()

    m = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16).cuda().eval()
    n = sum(p.numel() for p in m.parameters())
    names = [k for k, _ in m.named_parameters()]
    assert not any("visual" in k or "mtp" in k for k in names), "vision/MTP tensors still present"
    print(f"[verify] {type(m).__name__}: {n:,} parameters loaded (tied embeddings counted once)")
    with torch.no_grad():
        a = m(ids).logits[0, -1].float()
        gen = m.generate(ids, max_new_tokens=40, do_sample=False)
    print("[verify] greedy reply:", tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True))
    del m
    torch.cuda.empty_cache()

    full = AutoModelForImageTextToText.from_pretrained(REPO, revision=REVISION, dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        b = full(input_ids=ids).logits[0, -1].float()
    diff = (a - b).abs().max().item()
    same_top = int(a.argmax()) == int(b.argmax())
    print(f"[verify] max |logit diff| vs original text path: {diff:.4g}; same argmax: {same_top}")
    assert same_top and diff < 0.5, "stripped model does not reproduce the original text path"
    print("[verify] OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if not (out / "config.json").exists():
        strip(out)
    if a.verify:
        verify(out)


if __name__ == "__main__":
    main()
