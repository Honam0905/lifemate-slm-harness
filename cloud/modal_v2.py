"""LifeMate v2 on Modal: tokenize (CPU) -> LoRA training (1x H100) -> merge + dev eval (epoch 1 and 2) + public
prediction (epoch 2), all in the cloud. Written 5 Nov afternoon WITHOUT access to Modal: run `--step plumb` first.

    cd lifemate_v1
    LIFEMATE_TRAIN_CONFIG=configs/v2r.yaml modal run ...   # any step below with the renaming recipe (run "v2r")
    modal run cloud/modal_v2.py --step plumb               # ~15 min, ~$1.5: every stage on tiny amounts (smoke-tested on Colab A100, 5 Nov: SMOKE OK)
    modal run --detach cloud/modal_v2.py --step full       # tonight's run; keeps going if the laptop disconnects
    modal run cloud/modal_v2.py --step status              # what is on the volume (runs, eval summaries, submissions)
    modal run --detach cloud/modal_v2.py --step eval       # only merge + evals + epoch-2 prediction (training already done)
    modal run --detach cloud/modal_v2.py --step predict1   # public prediction from epoch 1 (if epoch 1 wins on dev)
    modal run --detach cloud/modal_v2.py --step eval_asr --asr ftA2-s400      # v2 on ASR round-2 dev transcripts
    modal run --detach cloud/modal_v2.py --step predict_asr --asr ftB2-s627   # public with ASR round-2 transcripts

The pipeline itself runs in a small remote orchestrator function, so a detached run needs no local machine.
Data and artifacts live on the Modal volume `lifemate-v2`; upload this layout first:
    /vol/data/{train,public_test}/...      the official folders WITHOUT audio (transcripts are cached)
    /vol/artifacts/splits/v1.json
    /vol/artifacts/trajectories/v1/{train.gold,train.asr,train.earlier,train.synth,dev.gold}.jsonl
    /vol/artifacts/asr/<ftA-s100 cache>/train.jsonl        dev transcripts for the dev eval
    /vol/artifacts/asr/<ftB-s157 cache>/public_test.jsonl  public transcripts for the submission
Outputs: /vol/artifacts/runs/<run>/, models/, eval/<name>/dev.C/{summary.json,rows.jsonl}, submissions/<name>/.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent.parent          # lifemate_v1/
CODE = "/root/zalo"
VOL = "/vol"
ART = f"{VOL}/artifacts"
GPU = os.environ.get("LIFEMATE_GPU", "H100")
# which training recipe: configs/v2.yaml (v2) or configs/v2r.yaml (v2 + renaming augmentation); the run is named after it
TRAIN_CONFIG = os.environ.get("LIFEMATE_TRAIN_CONFIG", "configs/v2.yaml")
RUN = os.environ.get("LIFEMATE_RUN", Path(TRAIN_CONFIG).stem)

app = modal.App("lifemate-v2")
vol = modal.Volume.from_name("lifemate-v2", create_if_missing=True)
hf = modal.Volume.from_name("lifemate-hf", create_if_missing=True)

# Same versions as the Colab training env (requirements-train.txt). Colab ran torch 2.11.0+cu130; cu128 is used here
# because it needs an older driver. If this wheel does not exist, try the cu126 or cu130 index instead.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.22.0", "peft==0.21.2", "accelerate==1.15.0", "torchao>=0.16", "safetensors",
                 "huggingface_hub[hf_xet]", "pyyaml", "numpy")
    .env({"HF_HOME": "/hf", "LIFEMATE_DATA_ROOT": f"{VOL}/data", "LIFEMATE_ARTIFACTS": ART, "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "LIFEMATE_TRAIN_CONFIG": TRAIN_CONFIG, "LIFEMATE_RUN": RUN})
    .add_local_dir(HERE, CODE, ignore=["artifacts", "train+public_test", ".venv-asr-local", ".kgeval_work", "**/__pycache__",
                                       "synth/conv/out_orig", "*.zip"])
)
VOLS = {VOL: vol, "/hf": hf}


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=CODE)


@app.function(image=image, volumes=VOLS, cpu=8, memory=32768, timeout=3600)
def tokenize() -> None:
    vol.reload()
    import sys
    sys.path.insert(0, CODE)
    from lifemate.config import load_config
    cfg = load_config(f"{CODE}/{TRAIN_CONFIG}")
    if cfg.get("rename"):                       # v2r: write the renamed copies first (seconds, deterministic)
        sh(f"python scripts/apply_rename.py --config {TRAIN_CONFIG}")
    variants = ["gold"] + [v for v in cfg["train"]["variants"] if v != "gold"]     # gold also gives dev.gold (eval loss)
    sh(f"python scripts/tokenize_trajectories.py --config {TRAIN_CONFIG} --parts train dev "
       f"--variants {' '.join(variants)} --show 1")
    vol.commit(); hf.commit()


@app.function(image=image, gpu=GPU, volumes=VOLS, cpu=8, memory=65536, timeout=6 * 3600,
              retries=modal.Retries(max_retries=3, initial_delay=10.0))
def train(run: str = RUN, max_steps: int = 0, config: str = TRAIN_CONFIG) -> None:
    """Always --resume: a preempted or retried container continues from the newest checkpoint on the volume."""
    vol.reload()
    stop = threading.Event()

    def committer():                       # make checkpoints durable while training runs
        while not stop.wait(300):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print(f"[modal] volume commit failed: {e}", flush=True)

    threading.Thread(target=committer, daemon=True).start()
    try:
        sh(f"nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
           f"python -m lifemate.train.sft --config {config} --run {run} --resume"
           + (f" --max_steps {max_steps}" if max_steps else ""))
    finally:
        stop.set()
        vol.commit(); hf.commit()


def _cfg(asr: str) -> str:
    """ASR tag -> config: ftA-s100 -> configs/v2_ftA_s100.yaml, ftB2-s627 -> configs/v2_ftB2_s627.yaml"""
    return f"configs/v2_{asr.replace('-', '_')}.yaml"


def _cache(asr: str, split: str) -> str:
    return f"{ART}/asr/qwen3_asr-Qwen3-ASR-1.7B-7278e1e-{asr}-ctxnames/{split}.jsonl"


def _merge(run: str, ckpt: str, out: str) -> str:
    model = f"{ART}/models/{out}"
    if not Path(model, "config.json").exists():
        sh(f"python -m lifemate.train.merge --config {TRAIN_CONFIG} --ckpt {ART}/runs/{run}/{ckpt} --out {model}")
    return model


@app.function(image=image, gpu=GPU, volumes=VOLS, cpu=8, memory=65536, timeout=3 * 3600)
def evaluate(run: str, ckpt: str, name: str, limit: int = 0, asr: str = "ftA-s100") -> dict:
    """Dev condition C with the given dev transcripts (default Model A s100: directly comparable with v1's 36.10;
    "ftA2-s400" etc. = ASR round 2)."""
    vol.reload()
    model = _merge(run, ckpt, name)          # own folder per job: parallel jobs never write the same files
    sh(f"python -m lifemate.eval.run_eval --config {_cfg(asr)} --name {name} --model {model} --conditions C"
       + (f" --limit {limit}" if limit else ""))
    vol.commit()
    return json.loads(Path(f"{ART}/eval/{name}/dev.C/summary.json").read_text())


@app.function(image=image, gpu=GPU, volumes=VOLS, cpu=8, memory=65536, timeout=3 * 3600)
def predict(run: str, ckpt: str, name: str, limit: int = 0, asr: str = "ftB-s157") -> str:
    """Public submission from cached Model B transcripts (default s157 = the 34.35 submission's ASR; "ftB2-s627" etc. =
    ASR round 2), through the deployment controller, one sample at a time."""
    vol.reload()
    model = _merge(run, ckpt, f"{name}-pred")   # own folder: an evaluate() may be merging the same checkpoint now
    out = f"{ART}/submissions/{name}/submission.jsonl"
    sh(f"python -m lifemate.infer.run --config {_cfg(asr)} --data {VOL}/data/public_test --model {model} "
       f"--asr_cache {_cache(asr, 'public_test')} --out {out}" + (f" --limit {limit}" if limit else ""))
    if not limit:
        sh(f"python -I scripts/check_submission.py {out} {VOL}/data")
    vol.commit()
    return out


@app.function(image=image, volumes=VOLS, cpu=1, memory=2048, timeout=10 * 3600)
def pipeline(step: str, asr: str = "") -> None:
    if step == "plumb":
        tokenize.remote()
        train.remote(run="v2-plumb", max_steps=3)
        print("[plumb] eval:", evaluate.remote("v2-plumb", "ckpt-step3", "v2-plumb", limit=3), flush=True)
        print("[plumb] predict:", predict.remote("v2-plumb", "ckpt-step3", "v2-plumb", limit=2), flush=True)
        print("[plumb] PLUMBING OK", flush=True)
        return
    if step in ("full", "train"):
        import sys
        sys.path.insert(0, CODE)
        from lifemate.config import load_config
        cfg = load_config(f"{CODE}/{TRAIN_CONFIG}")
        if not Path(f"{ART}/tokenized/{cfg['run_name']}/train.{cfg['train']['variants'][-1]}.pt").exists():
            tokenize.remote()
        train.remote(run=RUN)
        if step == "train":
            return
    if step in ("full", "eval"):
        jobs = [evaluate.spawn(RUN, "ckpt-epoch1", f"{RUN}-epoch1"), evaluate.spawn(RUN, "ckpt-epoch2", f"{RUN}-epoch2"),
                predict.spawn(RUN, "ckpt-epoch2", f"{RUN}-epoch2")]
        vol.reload()   # ASR round-2 dev transcripts already on the volume are scored too (epoch 2 only)
        for tag in sorted(p.parent.name.split("-7278e1e-")[1].rsplit("-ctxnames", 1)[0]
                          for p in Path(ART).glob("asr/*ftA2-s*-ctxnames/train.jsonl")):
            if Path(_cfg(tag).replace("configs/", f"{CODE}/configs/")).exists():
                jobs.append(evaluate.spawn(RUN, "ckpt-epoch2", f"{RUN}-epoch2-{tag}", asr=tag))
        r2 = Path(f"{ART}/asr/round2_results.json")       # the Kaggle round-2 results.json, uploaded under this name
        if r2.exists():
            pb = json.loads(r2.read_text()).get("public_b") or []
            if pb and Path(_cache(pb[0]["tag"], "public_test")).exists():   # B2 matching the best A2 by dev WER
                jobs.append(predict.spawn(RUN, "ckpt-epoch2", f"{RUN}-epoch2-{pb[0]['tag']}", asr=pb[0]["tag"]))
        for j in jobs:
            print("[pipeline]", j.get(), flush=True)
        print(f"[pipeline] DONE. Compare dev C with the current best before uploading submissions/{RUN}-epoch2*/submission.jsonl",
              flush=True)
    if step == "predict1":    # only if epoch 1 wins on dev
        print("[pipeline]", predict.remote(RUN, "ckpt-epoch1", f"{RUN}-epoch1"), flush=True)
    if step == "eval_asr":    # --asr ftA2-s400: score the v2 epoch-2 agent on those dev transcripts
        print("[pipeline]", evaluate.remote(RUN, "ckpt-epoch2", f"{RUN}-epoch2-{asr}", asr=asr), flush=True)
    if step == "predict_asr":  # --asr ftB2-s627: public submission with those transcripts (pick by the eval_asr result)
        print("[pipeline]", predict.remote(RUN, "ckpt-epoch2", f"{RUN}-epoch2-{asr}", asr=asr), flush=True)


@app.function(image=image, volumes=VOLS, cpu=1, memory=2048, timeout=600)
def status() -> None:
    vol.reload()
    for p in sorted(Path(ART).glob("eval/*/dev.C/summary.json")):
        s = json.loads(p.read_text())
        print(f"{p.parent.parent.name}: dev C primary {s['primary']} ci95 {s.get('primary_ci95')}")
    for d in sorted(Path(ART).glob("runs/*")):
        latest = (d / "latest").read_text().strip() if (d / "latest").exists() else "-"
        print(f"run {d.name}: latest {latest}")
    for p in sorted(Path(ART).glob("submissions/*/submission.jsonl")):
        print(f"submission {p} ({sum(1 for _ in open(p))} lines)")


@app.local_entrypoint()
def main(step: str = "plumb", asr: str = ""):
    if step == "status":
        status.remote()
    elif step == "plumb":
        pipeline.remote("plumb")
    else:                     # full / train / eval / predict1 / eval_asr / predict_asr: run remotely
        if step in ("eval_asr", "predict_asr") and not asr:
            raise SystemExit("--asr TAG is required, e.g. --asr ftA2-s400 (eval) or --asr ftB2-s627 (predict)")
        pipeline.remote(step, asr)
