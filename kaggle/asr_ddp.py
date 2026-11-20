"""Qwen3-ASR LoRA fine-tuning on several GPUs (Kaggle T4 x2), launched with torchrun.

    python -m torch.distributed.run --standalone --nproc_per_node 2 kaggle/asr_ddp.py \
        --model /tmp/hf/qwen3-asr --manifest /tmp/zalo/artifacts/asr_ft --out /kaggle/working/out/A --steps 200 --save 100 200

Same recipe as scripts/train_asr_lora.py (LoRA r16/alpha32 on the decoder self-attention q/k/v/o only, lr 5e-5,
linear warmup/decay, loss on the transcript + end token only), with three differences:
  * data parallel: every optimizer step takes micro x accum x world clips; each rank runs its own share and the LoRA
    gradients are averaged across ranks before the step (the frozen 2.3B weights are never synchronised, LoRA is ~6M
    parameters, so the all-reduce is a few MB). Default micro 2 x accum 4 x 2 GPUs = 16 = the single-GPU recipe.
  * --dtype: Kaggle T4/P100 have no bfloat16, so float32 is the default here (float16 can overflow in Qwen layers).
  * --context_mode names: the training prompt carries the same names-only context the pipeline passes at inference
    (lifemate/asr/context.py), instead of an empty one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_asr_lora as T  # noqa: E402  (prompt_prefix, end_token, read_manifest, load_wavs, TARGET_MODULES, IGNORE)
from lifemate.asr.context import asr_context  # noqa: E402
from lifemate.config import data_root, load_config  # noqa: E402
from lifemate.data.loader import load_samples  # noqa: E402

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
RANK, WORLD = 0, 1


def log(msg: str) -> None:
    if RANK == 0:
        print(f"[asr-ddp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_batch(processor, rows: list[dict], wavs, prefixes: list[str], eos: str, device, dtype) -> dict:
    """T.make_batch with one prompt prefix per row (each clip carries its own sample's context)."""
    tok = processor.tokenizer
    pre = processor(text=prefixes, audio=wavs, return_tensors="pt", padding=True)
    end_id = tok.convert_tokens_to_ids(eos)
    seqs, labs = [], []
    for i, r in enumerate(rows):
        p_ids = pre["input_ids"][i][pre["attention_mask"][i].bool()].tolist()
        t_ids = tok.encode(r["text"], add_special_tokens=False) + [end_id]
        seqs.append(p_ids + t_ids)
        labs.append([T.IGNORE] * len(p_ids) + t_ids)
    n = max(len(s) for s in seqs)
    ids = torch.full((len(rows), n), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(rows), n), dtype=torch.long)
    labels = torch.full((len(rows), n), T.IGNORE, dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
        labels[i, : len(l)] = torch.tensor(l)
    out = {"input_ids": ids, "attention_mask": att, "labels": labels,
           "input_features": pre["input_features"], "feature_attention_mask": pre["feature_attention_mask"]}
    out = {k: v.to(device) for k, v in out.items()}
    return {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in out.items()}


@torch.no_grad()
def holdout_loss(thinker, processor, rows, root, prefix_of, eos, bs: int = 8) -> float:
    """Token-weighted loss over the holdout clips, split across ranks and summed."""
    thinker.eval()
    tot = torch.zeros(2, device=thinker.device, dtype=torch.float64)
    mine = rows[RANK::WORLD]
    for i in range(0, len(mine), bs):
        part = mine[i: i + bs]
        b = make_batch(processor, part, T.load_wavs(root, part), [prefix_of(r) for r in part], eos,
                       thinker.device, thinker.dtype)
        k = int((b["labels"] != T.IGNORE).sum())
        tot[0] += thinker(**b).loss.double().item() * k
        tot[1] += k
    if WORLD > 1:
        dist.all_reduce(tot)
    thinker.train()
    return float(tot[0] / tot[1].clamp(min=1))


def main():
    global RANK, WORLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--model", required=True, help="local Qwen3-ASR snapshot dir (download once before launching)")
    ap.add_argument("--manifest", required=True, help="dir with train.jsonl (and holdout.jsonl unless --no_holdout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--save", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="per rank; effective batch = micro x accum x world")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    ap.add_argument("--context_mode", default="names", choices=["names", "none"])
    ap.add_argument("--no_holdout", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N train clips")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        RANK, WORLD = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        local = 0
    torch.cuda.set_device(local)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)

    cfg = load_config(ROOT / a.config)
    root = data_root(cfg)
    man = Path(a.manifest)
    train = T.read_manifest(man / "train.jsonl")
    hold = [] if a.no_holdout or not (man / "holdout.jsonl").exists() else T.read_manifest(man / "holdout.jsonl")
    if a.limit:
        train, hold = train[: a.limit], hold[: max(2, a.limit // 4)] if hold else []
    by_id = {s["id"]: s for s in load_samples(root, "train")}
    language = cfg["asr"].get("language") or "Vietnamese"

    from peft import LoraConfig, get_peft_model
    from qwen_asr import Qwen3ASRModel
    w = Qwen3ASRModel.from_pretrained(a.model, dtype=DTYPES[a.dtype], device_map=f"cuda:{local}")
    model, processor = w.model, w.processor
    processor.tokenizer.padding_side = "right"
    eos = T.end_token(model, processor)
    ctx_cache: dict[str, str] = {}

    def prefix_of(r: dict) -> str:
        sid = r["sample_id"]
        if sid not in ctx_cache:
            ctx_cache[sid] = asr_context(by_id[sid], "names") if a.context_mode == "names" else ""
        return T.prompt_prefix(processor, language, ctx_cache[sid])

    lcfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=T.TARGET_MODULES, bias="none")
    model = get_peft_model(model, lcfg)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n, _ in trainable if "audio_tower" in n or "lora_" not in n or ".model.layers." not in n]
    assert trainable and not bad, f"unexpected trainable parameters: {bad[:5]}"
    params = [p for _, p in trainable]
    if WORLD > 1:   # identical LoRA start on every rank
        for p in params:
            dist.broadcast(p.data, src=0)
    torch.manual_seed(a.seed + RANK)   # different dropout masks per rank from here on
    thinker = model.base_model.model.thinker
    if a.grad_ckpt:
        thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    thinker.train()
    log(f"{WORLD} GPU(s), dtype {a.dtype}, context {a.context_mode}; LoRA r{a.r}/alpha{a.alpha} on {len(params) // 2} "
        f"projections = {sum(p.numel() for p in params):,} trainable; {len(train)} train / {len(hold)} holdout clips; "
        f"effective batch {a.micro * a.accum * WORLD}")
    b0 = make_batch(processor, train[:1], T.load_wavs(root, train[:1]), [prefix_of(train[0])], eos, thinker.device, thinker.dtype)
    log(f"context of example 0: {ctx_cache[train[0]['sample_id']]!r}")
    log(f"supervised text of example 0: {processor.tokenizer.decode(b0['labels'][0][b0['labels'][0] != T.IGNORE])!r}")
    log(f"manifest text of example 0  : {train[0]['text'] + eos!r}")

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = a.steps

    def lr_factor(s: int) -> float:   # linear warmup, then linear decay to 0 at the last step (as train_asr_lora.py)
        return (s + 1) / a.warmup if s < a.warmup else max(0.0, (total - s) / max(1, total - a.warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    out = Path(a.out)
    if RANK == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps({**vars(a), "world": WORLD, "n_train": len(train), "n_holdout": len(hold)}, indent=1))
    history = {"holdout": {}, "loss": {}}
    if hold:
        h = holdout_loss(thinker, processor, hold, root, prefix_of, eos)
        history["holdout"][0] = h
        log(f"holdout loss before training: {h:.4f} ({len(hold)} clips)")

    need = a.micro * a.accum * WORLD
    order, pos, step, t0 = list(range(len(train))), 0, 0, time.time()
    random.Random(a.seed).shuffle(order)
    while step < total:
        if pos + need > len(order):
            random.Random(a.seed + step).shuffle(order)
            pos = 0
        chunk = order[pos: pos + need]
        pos += need
        mine = chunk[RANK * a.micro * a.accum: (RANK + 1) * a.micro * a.accum]
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for j in range(a.accum):
            rows = [train[i] for i in mine[j * a.micro: (j + 1) * a.micro]]
            b = make_batch(processor, rows, T.load_wavs(root, rows), [prefix_of(r) for r in rows], eos,
                           thinker.device, thinker.dtype)
            loss = thinker(**b).loss / a.accum
            loss.backward()
            tot += loss.item()
        stats = torch.tensor([tot], device=thinker.device, dtype=torch.float64)
        if WORLD > 1:   # average LoRA gradients (one flat all-reduce) and the logged loss over ranks
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat)
            flat /= WORLD
            off = 0
            for p in params:
                n = p.numel()
                p.grad = flat[off: off + n].view_as(p).clone()
                off += n
            dist.all_reduce(stats)
            stats /= WORLD
        mean_loss = float(stats[0])
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1} (try --grad_ckpt or float32)")
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        history["loss"][step] = mean_loss
        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            log(f"step {step}/{total} loss {mean_loss:.4f} gnorm {float(gn):.3f} lr {sched.get_last_lr()[0]:.2e} "
                f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G elapsed {el / 60:.1f}m "
                f"eta {el / step * (total - step) / 60:.1f}m")
        if step in a.save:
            if RANK == 0:
                model.save_pretrained(out / f"step{step}")
            if WORLD > 1:
                dist.barrier()
            msg = f"saved {out / f'step{step}'}"
            if hold:
                h = holdout_loss(thinker, processor, hold, root, prefix_of, eos)
                history["holdout"][step] = h
                msg += f"; holdout loss {h:.4f}"
            log(msg)
    if RANK == 0:
        (out / "history.json").write_text(json.dumps({**history, "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    log(f"training done in {(time.time() - t0) / 60:.1f} min")
    if WORLD > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
