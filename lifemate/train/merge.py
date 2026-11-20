"""Merge a LoRA checkpoint into the base model, save it, and print the parameter inventory of the loaded tensors.
A checkpoint without adapter_config.json is a FULL fine-tuning checkpoint (train.full_ft): it is loaded as is and saved
in the same inference-ready form (bf16, base tokenizer files, classic rope_theta key).

    python -m lifemate.train.merge --ckpt artifacts/runs/v1/ckpt-epoch2 --out artifacts/models/v1-epoch2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from lifemate.config import load_config  # noqa: E402


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ag = load_config(a.config)["agent"]
    full = not (Path(a.ckpt) / "adapter_config.json").exists()
    if full:
        model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16)
        n_base = sum(p.numel() for p in AutoModelForCausalLM.from_pretrained(
            ag["model_id"], revision=ag["revision"], dtype=torch.bfloat16).parameters())
        print(f"[merge] {a.ckpt}: full fine-tuning checkpoint (no adapter), re-saving it")
    else:
        base = AutoModelForCausalLM.from_pretrained(ag["model_id"], revision=ag["revision"], dtype=torch.bfloat16)
        n_base = sum(p.numel() for p in base.parameters())
        model = PeftModel.from_pretrained(base, a.ckpt).merge_and_unload()
    n = sum(p.numel() for p in model.parameters())
    assert n == n_base, f"merged model has {n:,} params, base has {n_base:,}"
    out = Path(a.out)
    model.save_pretrained(out, safe_serialization=True)
    # copy the base model's own tokenizer files instead of re-saving them: a tokenizer saved by transformers 5.x
    # ("TokenizersBackend") cannot be loaded by the 4.57 inference env, while the original files load in both
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(ag["model_id"], revision=ag["revision"],
                                  allow_patterns=["tokenizer*", "special_tokens_map.json", "chat_template.jinja"]))
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        if (snap / f).exists():
            shutil.copy(snap / f, out / f)
    AutoTokenizer.from_pretrained(out)  # must load
    # transformers 5.x saves RoPE only as "rope_parameters"; the 4.57 inference env ignores that key and silently falls
    # back to rope_theta 10000 (MiniCPM5 uses 5,000,000), which wrecks long prompts. Write the classic keys too.
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    rp = cfg.get("rope_parameters") or {}
    if "rope_theta" in rp and cfg.get("rope_theta") is None:
        cfg["rope_theta"] = rp["rope_theta"]
        if rp.get("rope_type", "default") == "default":
            cfg["rope_scaling"] = None
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[merge] config rope_theta {cfg.get('rope_theta')} | rope_parameters {cfg.get('rope_parameters')}")
    if not full:
        shutil.copy(Path(a.ckpt) / "adapter_config.json", out / "source_adapter_config.json")
    (out / "MERGED_FROM.txt").write_text(f"base {ag['model_id']}@{ag['revision']}\n{'full checkpoint' if full else 'adapter'} "
                                         f"{Path(a.ckpt).resolve()}\n")
    reloaded = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    n2 = sum(p.numel() for p in reloaded.parameters())
    print(f"[merge] {a.ckpt} -> {out}: {n2:,} parameters in the reloaded merged model (base {n_base:,})")


if __name__ == "__main__":
    main()
