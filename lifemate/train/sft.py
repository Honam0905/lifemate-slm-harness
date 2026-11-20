"""LoRA (or full, `train.full_ft: true`) SFT on pre-tokenised trajectories (assistant-only loss), resumable (LoRA).

    python -m lifemate.train.sft --run v1 [--max_steps 3] [--resume] [--push_repo user/name]

One sequence per micro-batch (no padding, no packing), gradient accumulation to the effective batch. The loss is the
mean token cross-entropy over every target token in the accumulation window. Logits are computed only at positions
whose next token is a target (`logits_to_keep`), so a 16K sequence never materialises a 16K x vocab logit matrix.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import shutil
import sys
import time
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from lifemate.config import artifacts_dir, load_config  # noqa: E402

IGNORE = -100
AMP = False   # full fine-tuning: fp32 master weights, bf16 autocast forward/backward (pure-bf16 Adam would round
              # lr-1e-5 updates away: bf16 keeps ~3 significant digits)


def log(msg: str) -> None:
    print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_examples(paths: list[Path]) -> list[dict]:
    ex = []
    for p in paths:
        ex += torch.load(p)
    return [e for e in ex if e["n_target"] > 0]


def epoch_order(n: int, seed: int, epoch: int, sources: list[str] | None = None,
                fractions: dict | None = None) -> list[int]:
    """Shuffled example order for one epoch. With `fractions` ({source: f < 1}), each such source contributes a
    different slice of a fixed permutation every epoch (f = 0.5 over 2 epochs = every example once)."""
    if not fractions:
        order = list(range(n))
    else:
        by = collections.defaultdict(list)
        for i, s in enumerate(sources):
            by[s].append(i)
        order = []
        for s, idx in sorted(by.items()):
            f = float(fractions.get(s, 1.0))
            if f >= 1:
                order += idx
                continue
            perm = idx[:]
            random.Random(seed * 7919 + zlib.crc32(s.encode())).shuffle(perm)   # same permutation every epoch
            k = max(1, round(len(perm) * f))
            start = (epoch * k) % len(perm)
            order += (perm + perm)[start: start + k]
    random.Random(seed * 1000 + epoch).shuffle(order)
    return order


def build_model(cfg: dict):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    ag, tc = cfg["agent"], cfg["train"]
    full = bool(tc.get("full_ft"))
    model = AutoModelForCausalLM.from_pretrained(ag["model_id"], revision=ag["revision"],
                                                 dtype=torch.float32 if full else torch.bfloat16,
                                                 attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if full:
        for p in model.parameters():
            p.requires_grad_(True)
        n_all = sum(p.numel() for p in model.parameters())
        log(f"params: {n_all:,} total, all trainable (FULL fine-tuning, fp32 master weights + bf16 autocast)")
        return model
    linear = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)})
    log(f"linear module names in the base model: {linear}")
    missing = [t for t in tc["lora_targets"] if t not in linear]
    if missing:
        raise ValueError(f"LoRA targets not found in the model: {missing}")
    lcfg = LoraConfig(r=tc["lora_r"], lora_alpha=tc["lora_alpha"], lora_dropout=tc["lora_dropout"],
                      target_modules=tc["lora_targets"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"params: {n_all:,} total, {n_train:,} trainable (LoRA r={tc['lora_r']})")
    return model


def set_grad_ckpt(model, on: bool) -> None:
    """Switch gradient checkpointing per sequence: short sequences fit without it and run faster."""
    for m in model.modules():
        if isinstance(getattr(m, "gradient_checkpointing", None), bool):
            m.gradient_checkpointing = on


def micro_loss(model, ex: dict, device) -> tuple[torch.Tensor, int]:
    """Summed CE over the targets of one sequence, and the number of target tokens."""
    ids = torch.tensor(ex["input_ids"], device=device).unsqueeze(0)
    labels = torch.tensor(ex["labels"], device=device)
    nxt = labels[1:]
    keep = (nxt != IGNORE).nonzero(as_tuple=True)[0]          # positions t whose next token is a target
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
        out = model(input_ids=ids, logits_to_keep=keep, use_cache=False)
    logits = out.logits[0].float()                            # [K, V]
    loss = F.cross_entropy(logits, nxt[keep], reduction="sum")
    return loss, int(keep.numel())


@torch.no_grad()
def eval_loss(model, examples: list[dict], device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for ex in examples:
        l, k = micro_loss(model, ex, device)
        tot += l.item(); n += k
    model.train()
    return tot / max(n, 1)


def save_ckpt(model, opt, sched, state: dict, out: Path, tag: str, keep_last: int = 2) -> Path:
    d = out / f"ckpt-{tag}"
    tmp = out / f".tmp-{tag}"
    shutil.rmtree(tmp, ignore_errors=True)
    if AMP:   # full fine-tuning: save the weights in bf16 (5 GB, what inference uses); no optimizer state (20 GB)
        sd = {k: v.detach().to("cpu", torch.bfloat16) for k, v in model.state_dict().items()}
        try:
            model.save_pretrained(tmp, state_dict=sd, safe_serialization=True)
        except TypeError as e:   # a transformers version without the state_dict argument: save fp32 (10 GB)
            log(f"bf16 save not supported ({e}); saving fp32")
            model.save_pretrained(tmp, safe_serialization=True)
        del sd
        torch.save({"state": state}, tmp / "trainer_state.pt")
    else:
        model.save_pretrained(tmp)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "state": state,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, tmp / "trainer_state.pt")
    (tmp / "state.json").write_text(json.dumps(state, indent=1))
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    (out / "latest").write_text(d.name)
    steps = sorted((p for p in out.glob("ckpt-step*")), key=lambda p: int(p.name.split("step")[1]))
    for p in steps[:-keep_last]:
        shutil.rmtree(p, ignore_errors=True)
    return d


def push(d: Path, repo: str | None, run: str) -> None:
    if not repo:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(d), repo_id=repo, path_in_repo=f"{run}/{d.name}", commit_message=f"{run} {d.name}")
        log(f"pushed {d.name} to {repo} (private)")
    except Exception as e:  # never kill a training run over an upload
        log(f"push of {d.name} failed: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--run", default=None, help="run directory name under artifacts/runs (default: config run_name)")
    ap.add_argument("--train_files", nargs="+", default=None)
    ap.add_argument("--eval_files", nargs="+", default=None)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after N optimizer steps (pilot)")
    ap.add_argument("--eval_limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--longest_first", action="store_true", help="pilot: put the longest sequences first to test peak memory")
    ap.add_argument("--push_repo", default=os.environ.get("LIFEMATE_PUSH_REPO"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    tc = cfg["train"]
    art = artifacts_dir(cfg)
    run = a.run or cfg["run_name"]
    out = art / "runs" / run
    out.mkdir(parents=True, exist_ok=True)
    tokdir = art / "tokenized" / (tc.get("data_tag") or cfg["run_name"])
    train_files = [Path(p) for p in a.train_files] if a.train_files else [tokdir / f"train.{v}.pt" for v in tc["variants"]]
    eval_files = [Path(p) for p in a.eval_files] if a.eval_files else [tokdir / "dev.gold.pt"]
    seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    train = load_examples(train_files)
    dev = load_examples([p for p in eval_files if p.exists()])
    if a.eval_limit:
        dev = dev[: a.eval_limit]
    n_tok = sum(len(e["input_ids"]) for e in train)
    log(f"train: {len(train)} sequences, {n_tok / 1e6:.2f}M tokens, {sum(e['n_target'] for e in train):,} target tokens "
        f"from {[p.name for p in train_files]}; dev: {len(dev)} sequences")

    global AMP
    AMP = bool(tc.get("full_ft"))
    model = build_model(cfg)
    device = torch.device("cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], betas=(0.9, 0.999), weight_decay=tc["weight_decay"], fused=True)
    accum = tc["grad_accum"] * tc["micro_batch"]
    fractions = tc.get("source_fraction") or {}
    sources = [e.get("source", "final") for e in train]
    orders = lambda ep: epoch_order(len(train), cfg["seed"], ep, sources, fractions)  # noqa: E731
    epoch_sizes = [len(orders(ep)) for ep in range(tc["epochs"])]
    steps_per_epoch = math.ceil(epoch_sizes[0] / accum)
    total_steps = sum(math.ceil(n / accum) for n in epoch_sizes)
    warmup = max(1, round(tc["warmup_ratio"] * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    state = {"step": 0, "epoch": 0, "pos": 0, "run": run, "config": cfg["_path"], "total_steps": total_steps}
    if a.resume and AMP and (out / "latest").exists():
        log("full fine-tuning keeps no optimizer state: not resuming, training from scratch")
    elif a.resume and (out / "latest").exists():
        d = out / (out / "latest").read_text().strip()
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(d / "adapter_model.safetensors"))
        ts = torch.load(d / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(ts["opt"]); sched.load_state_dict(ts["sched"]); state = ts["state"]
        torch.set_rng_state(ts["torch_rng"]); torch.cuda.set_rng_state_all(ts["cuda_rng"])
        log(f"resumed from {d.name}: step {state['step']}, epoch {state['epoch']}, pos {state['pos']}")
    log(f"{tc['epochs']} epochs of {epoch_sizes} sequences (source fractions {fractions or 'none'}) = {total_steps} optimizer steps, effective batch {accum}, "
        f"warmup {warmup}, lr {tc['lr']}")

    per_traj = tc.get("loss_norm", "token") == "trajectory"
    ckpt_min = int(tc.get("ckpt_min_tokens", 0) or 0)
    log(f"loss: {'per-trajectory (weighted)' if per_traj else 'token-weighted'}; gradient checkpointing "
        f"{'only for sequences >= %d tokens' % ckpt_min if ckpt_min else 'always'}; "
        f"sources: {dict(__import__('collections').Counter(e.get('source', 'final') for e in train))}")
    model.train()
    t_start = time.time()
    tok_window, t_window = 0, time.time()
    stop = False
    while state["epoch"] < tc["epochs"] and not stop:
        order = orders(state["epoch"])
        if a.longest_first:
            order = sorted(order, key=lambda i: -len(train[i]["input_ids"]))
        while state["pos"] < len(order):
            window = [train[i] for i in order[state["pos"]: state["pos"] + accum]]
            n_target = sum(e["n_target"] for e in window)
            wsum = sum(e.get("weight", 1.0) for e in window)
            tot = 0.0
            for ex in window:
                if ckpt_min:
                    set_grad_ckpt(model, len(ex["input_ids"]) >= ckpt_min)
                loss, k = micro_loss(model, ex, device)
                if per_traj:   # every conversation counts equally: weighted mean of per-sequence mean CE
                    w = ex.get("weight", 1.0) / wsum
                    (loss / k * w).backward()
                    tot += loss.item() / k * w
                else:          # v1: mean over every target token in the window
                    (loss / n_target).backward()
                    tot += loss.item() / n_target
                tok_window += len(ex["input_ids"])
            gn = torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"]).item()
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            state["pos"] += len(window)
            state["step"] += 1
            if not math.isfinite(tot):
                raise FloatingPointError(f"non-finite loss at step {state['step']}")
            el = time.time() - t_window
            log(f"step {state['step']}/{total_steps} ep {state['epoch']} loss {tot:.4f} gnorm {gn:.3f} "
                f"lr {sched.get_last_lr()[0]:.2e} tok/s {tok_window / el:,.0f} "
                f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G elapsed {(time.time() - t_start) / 60:.1f}m")
            tok_window, t_window = 0, time.time()
            if state["step"] % tc["save_every_steps"] == 0:
                save_ckpt(model, opt, sched, state, out, f"step{state['step']}")
            if a.max_steps and state["step"] >= a.max_steps:
                stop = True
                break
        if stop:
            break
        state["epoch"] += 1
        state["pos"] = 0
        d = save_ckpt(model, opt, sched, state, out, f"epoch{state['epoch']}")
        if dev:
            log(f"epoch {state['epoch']} dev loss {eval_loss(model, dev, device):.4f} ({len(dev)} sequences)")
        push(d, a.push_repo, run)
    if stop:
        d = save_ckpt(model, opt, sched, state, out, f"step{state['step']}")
        if dev:
            log(f"dev loss at step {state['step']}: {eval_loss(model, dev, device):.4f} ({len(dev)} sequences)")
    log(f"done: {state['step']} steps in {(time.time() - t_start) / 60:.1f} min; last checkpoint {d}")


if __name__ == "__main__":
    main()
