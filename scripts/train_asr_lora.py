"""LoRA fine-tuning of Qwen3-ASR-1.7B on the train-partition ASR manifest, and merging a checkpoint for inference.

    # train (decoder self-attention only; the audio encoder, embeddings and head stay frozen)
    python scripts/train_asr_lora.py train --out artifacts/asr_ft/run1 [--steps 200 --save 100 200]
    # merge one checkpoint into a standalone model dir that Qwen3ASRModel.from_pretrained (and make_asr) can load
    python scripts/train_asr_lora.py merge --ckpt artifacts/asr_ft/run1/step200 --out artifacts/models/asr-ft-s200

Training text = exactly the inference prompt of qwen_asr with a forced language and empty context
(chat template + generation prompt + "language Vietnamese<asr_text>"), followed by the annotator transcript and the
end token; the loss covers only the transcript and the end token. Holdout loss is logged at every save.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402

TARGET_MODULES = r"thinker\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
IGNORE = -100


def log(msg: str) -> None:
    print(f"[asr-ft {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_base(cfg: dict):
    from qwen_asr import Qwen3ASRModel
    a = cfg["asr"]
    w = Qwen3ASRModel.from_pretrained(a["model_id"], revision=a["revision"], dtype=torch.bfloat16, device_map="cuda:0")
    return w


def prompt_prefix(processor, language: str, context: str = "") -> str:
    """The same prefix qwen_asr builds at inference (Qwen3ASRModel._build_text_prompt with a forced language)."""
    msgs = [{"role": "system", "content": context or ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    base = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return base + f"language {language}<asr_text>"


END_TOKEN = "<|im_end|>"   # the chat turn end (generation stops on it); <|endoftext|> is also the pad token


def end_token(model, processor) -> str:
    eid = processor.tokenizer.convert_tokens_to_ids(END_TOKEN)
    eos = model.generation_config.eos_token_id
    assert eid in (eos if isinstance(eos, (list, tuple)) else [eos]), f"{END_TOKEN} ({eid}) is not a stop token: {eos}"
    return END_TOKEN


def make_batch(processor, rows: list[dict], wavs: list[np.ndarray], prefix: str, eos: str, device, dtype) -> dict:
    """input_ids = the inference prompt (processor: chat template + expanded audio tokens) + transcript tokens + end
    token, built per row so the label mask is exact; right-padded. Audio features come from the same processor call."""
    tok = processor.tokenizer
    pre = processor(text=[prefix] * len(rows), audio=wavs, return_tensors="pt", padding=True)
    end_id = tok.convert_tokens_to_ids(eos)
    seqs, labs = [], []
    for i, r in enumerate(rows):
        p_ids = pre["input_ids"][i][pre["attention_mask"][i].bool()].tolist()
        t_ids = tok.encode(r["text"], add_special_tokens=False) + [end_id]
        seqs.append(p_ids + t_ids)
        labs.append([IGNORE] * len(p_ids) + t_ids)
    n = max(len(s) for s in seqs)
    ids = torch.full((len(rows), n), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(rows), n), dtype=torch.long)
    labels = torch.full((len(rows), n), IGNORE, dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        ids[i, : len(s)] = torch.tensor(s); att[i, : len(s)] = 1; labels[i, : len(l)] = torch.tensor(l)
    out = {"input_ids": ids, "attention_mask": att, "labels": labels,
           "input_features": pre["input_features"], "feature_attention_mask": pre["feature_attention_mask"]}
    out = {k: v.to(device) for k, v in out.items()}
    for k, v in out.items():
        if torch.is_floating_point(v):
            out[k] = v.to(dtype)
    return out


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_wavs(root: Path, rows: list[dict]) -> list[np.ndarray]:
    from qwen_asr.inference.utils import normalize_audios
    return normalize_audios([str(root / "train" / r["audio"]) for r in rows])


@torch.no_grad()
def holdout_loss(thinker, processor, rows, root, prefix, eos, bs: int = 8) -> float:
    thinker.eval()
    tot, n = 0.0, 0
    for i in range(0, len(rows), bs):
        part = rows[i: i + bs]
        b = make_batch(processor, part, load_wavs(root, part), prefix, eos, thinker.device, thinker.dtype)
        k = int((b["labels"] != IGNORE).sum())
        tot += thinker(**b).loss.item() * k
        n += k
    thinker.train()
    return tot / max(n, 1)


def cmd_train(a, cfg):
    from peft import LoraConfig, get_peft_model
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    root, art = data_root(cfg), artifacts_dir(cfg)
    man = Path(a.manifest) if a.manifest else art / "asr_ft"
    train, hold = read_manifest(man / "train.jsonl"), read_manifest(man / "holdout.jsonl")
    if a.limit:
        train, hold = train[: a.limit], hold[: max(2, a.limit // 4)]
    w = load_base(cfg)
    model, processor = w.model, w.processor
    processor.tokenizer.padding_side = "right"
    language = cfg["asr"].get("language") or "Vietnamese"
    prefix, eos = prompt_prefix(processor, language), end_token(model, processor)
    log(f"end token {eos!r}; prefix tail {prefix[-60:]!r}")
    lcfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=TARGET_MODULES, bias="none")
    model = get_peft_model(model, lcfg)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n, _ in trainable if "audio_tower" in n or "lora_" not in n or ".model.layers." not in n]
    assert trainable and not bad, f"unexpected trainable parameters: {bad[:5]}"
    log(f"LoRA r{a.r}/alpha{a.alpha} on {len(trainable) // 2} decoder attention projections: "
        f"{sum(p.numel() for _, p in trainable):,} trainable parameters")
    thinker = model.base_model.model.thinker
    if a.grad_ckpt:
        thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    thinker.train()
    # one sanity look at the labels: decode the supervised part of the first example
    b0 = make_batch(processor, train[:1], load_wavs(root, train[:1]), prefix, eos, thinker.device, thinker.dtype)
    sup = b0["labels"][0][b0["labels"][0] != IGNORE]
    log(f"supervised text of example 0: {processor.tokenizer.decode(sup)!r}")
    log(f"manifest text of example 0  : {train[0]['text'] + eos!r}")
    opt = torch.optim.AdamW([p for _, p in trainable], lr=a.lr, weight_decay=0.0)
    total = a.steps
    def lr_factor(s: int) -> float:   # linear warmup, then linear decay to 0 at the last step
        if s < a.warmup:
            return (s + 1) / a.warmup
        return max(0.0, (total - s) / max(1, total - a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    log(f"holdout loss before training: {holdout_loss(thinker, processor, hold, root, prefix, eos):.4f} ({len(hold)} clips)")
    order, pos, step, t0 = list(range(len(train))), 0, 0, time.time()
    random.Random(a.seed).shuffle(order)
    while step < total:
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.accum):
            if pos + a.micro > len(order):
                random.Random(a.seed + step).shuffle(order); pos = 0
            rows = [train[i] for i in order[pos: pos + a.micro]]
            pos += a.micro
            b = make_batch(processor, rows, load_wavs(root, rows), prefix, eos, thinker.device, thinker.dtype)
            loss = thinker(**b).loss / a.accum
            loss.backward()
            tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step(); sched.step(); step += 1
        if step % 10 == 0 or step == 1:
            log(f"step {step}/{total} loss {tot:.4f} gnorm {float(gn):.3f} lr {sched.get_last_lr()[0]:.2e} "
                f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G elapsed {(time.time() - t0) / 60:.1f}m")
        if not math.isfinite(tot):
            sys.exit(f"non-finite loss at step {step}")
        if step in a.save:
            d = out / f"step{step}"
            model.save_pretrained(d)
            log(f"saved {d}; holdout loss {holdout_loss(thinker, processor, hold, root, prefix, eos):.4f}")
    log("training done")


def cmd_merge(a, cfg):
    from peft import PeftModel
    from huggingface_hub import snapshot_download
    w = load_base(cfg)
    n_base = sum(p.numel() for p in w.model.parameters())
    merged = PeftModel.from_pretrained(w.model, a.ckpt).merge_and_unload()
    n = sum(p.numel() for p in merged.parameters())
    assert n == n_base, f"merged {n:,} != base {n_base:,}"
    out = Path(a.out)
    gc = merged.generation_config   # the shipped config pairs do_sample=False with temperature=1e-6, which save rejects;
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None   # greedy, as the pipeline decodes
    merged.save_pretrained(out, safe_serialization=True)
    snap = Path(snapshot_download(cfg["asr"]["model_id"], revision=cfg["asr"]["revision"]))
    for f in snap.iterdir():   # processor / tokenizer / chat template files, so the dir loads standalone
        if f.is_file() and not f.name.endswith(".safetensors") and not f.name.endswith(".safetensors.index.json") \
                and not (out / f.name).exists():
            shutil.copy(f, out / f.name)
    (out / "MERGED_FROM.txt").write_text(f"base {cfg['asr']['model_id']}@{cfg['asr']['revision']}\nadapter {Path(a.ckpt).resolve()}\n")
    del merged, w
    torch.cuda.empty_cache()
    # reload through the pipeline's own backend and transcribe a few held-out clips (integration check)
    from lifemate.asr.qwen3_asr import Qwen3ASR
    asr = Qwen3ASR(str(out), None, cfg["asr"].get("language"), 4, cfg["asr"]["max_new_tokens"], tag="check")
    n2 = sum(p.numel() for p in asr.model.model.parameters())
    log(f"merged {a.ckpt} -> {out}: {n2:,} parameters reloaded (base {n_base:,})")
    man = Path(a.manifest) if a.manifest else artifacts_dir(cfg) / "asr_ft"
    rows = read_manifest(man / "holdout.jsonl")[:3]
    for r, txt in zip(rows, asr.transcribe([str(data_root(cfg) / "train" / r["audio"]) for r in rows])):
        log(f"  gold: {r['text']}\n                  asr : {txt}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/v1.yaml")
    t.add_argument("--manifest", default=None, help="dir with train.jsonl / holdout.jsonl (default artifacts/asr_ft)")
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=200)
    t.add_argument("--save", type=int, nargs="+", default=[100, 200])
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--warmup", type=int, default=10)
    t.add_argument("--micro", type=int, default=2)
    t.add_argument("--accum", type=int, default=8)
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--alpha", type=int, default=32)
    t.add_argument("--dropout", type=float, default=0.05)
    t.add_argument("--grad_ckpt", action="store_true")
    t.add_argument("--limit", type=int, default=0, help="smoke test: only the first N train clips")
    t.add_argument("--seed", type=int, default=1234)
    m = sub.add_parser("merge")
    m.add_argument("--config", default="configs/v1.yaml")
    m.add_argument("--ckpt", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--manifest", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    (cmd_train if a.cmd == "train" else cmd_merge)(a, cfg)


if __name__ == "__main__":
    main()
