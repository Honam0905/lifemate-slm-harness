"""Write kaggle/lifemate_asr_overnight.ipynb (the overnight Kaggle notebook) from the cell sources below."""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


md(r"""
# LifeMate — Qwen3-ASR LoRA, overnight on 2 × T4

**Settings (right panel):** Accelerator **GPU T4 ×2** · Internet **On** · add the two private datasets (data + code).

**Run:** first set `MODE = "smoke"` and *Run All* interactively (~10 min, checks install, both GPUs, training and
transcription end to end). Then set `MODE = "full"` → **Save Version → Save & Run All (Commit)** and close the tab.

**What the full run does** (everything lands in `/kaggle/working/out`, downloadable from the version's *Output* tab):
1. **Model A** — LoRA on the 3,212 train-partition clips (354 held out), 200 steps, adapters at steps 100 and 200.
2. **Dev transcripts** (887 clips) with the base ASR, A-step100 and A-step200 → `out/asr/<cache name>/train.jsonl` + WER.
   Tomorrow the agent's dev score on these (Colab, `run_eval`) decides whether the fine-tuned ASR is adopted.
3. **Model B** — the same recipe on **all** 5,036 train clips (incl. dev/audit), steps scaled to the same single pass
   (157 / 314). Final-fit candidate only: **never score Model B on dev or audit.**
4. **Public transcripts** (1,933 clips) with both Model B checkpoints → ready for the submission.

Both GPUs are used throughout: training runs as 2 data-parallel processes (`torchrun`), transcription as one shard per GPU.
""")

code(r"""
# ---- settings ----------------------------------------------------------------------------------------------
MODE = "smoke"          # "smoke" = quick end-to-end check (run interactively first) | "full" = the overnight run
DTYPE = "float32"       # T4 has no bfloat16; float16 can overflow in Qwen layers -> float32 (fast enough here)
STEPS_A, SAVE_A = 200, [100, 200]
MICRO, ACCUM = 2, 4     # per GPU; x 2 GPUs = effective batch 16, the same recipe as the single-GPU pilot
LR, LORA_R, LORA_ALPHA = 5e-5, 16, 32
GRAD_CKPT = False       # set True only if the smoke run reports out-of-memory
RUN_B = True            # Model B: same recipe on ALL train clips (final fit)
PUBLIC = True           # transcribe the public test clips with both Model B checkpoints
ASR_REPO, ASR_REV = "Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5"
""")

code(r"""
# ---- locate the two datasets, prepare a writable workspace ---------------------------------------------------
import os, sys, json, shutil, subprocess, time
from pathlib import Path

INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
OUT = Path(os.environ.get("LIFEMATE_KAGGLE_OUT", "/kaggle/working/out"))
WORK = Path(os.environ.get("LIFEMATE_KAGGLE_WORK", "/tmp/lifemate"))

def _find_dirs(pred):
    hits = []
    for dp, dns, fns in os.walk(INPUT, followlinks=True):
        dns[:] = [d for d in dns if d not in ("__MACOSX", "audio", "train_environment", "public_test_environment")]
        if pred(Path(dp), fns):
            hits.append(Path(dp))
    return sorted(hits, key=lambda p: len(p.parts))

data_hits = _find_dirs(lambda p, fns: (p / "train" / "train.jsonl").exists() and (p / "train" / "audio").is_dir())
code_hits = _find_dirs(lambda p, fns: (p / "lifemate" / "__init__.py").exists() and (p / "kaggle" / "asr_ddp.py").exists())
assert data_hits, f"no folder with train/train.jsonl + train/audio/ under {INPUT}"
assert code_hits, f"no folder with lifemate/ + kaggle/asr_ddp.py under {INPUT} (attach the code dataset)"
DATA, CODE_SRC = data_hits[0], code_hits[0]

CODE = WORK / "zalo"
if CODE.exists():
    shutil.rmtree(CODE)
shutil.copytree(CODE_SRC, CODE, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "vm", "dist"))
ART = CODE / "artifacts"
for d in (OUT, OUT / "logs", WORK / "shards"):
    d.mkdir(parents=True, exist_ok=True)
os.environ.update(LIFEMATE_DATA_ROOT=str(DATA), LIFEMATE_ARTIFACTS=str(ART), HF_HOME=str(WORK / "hf"),
                  TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1")
sys.path.insert(0, str(CODE))

n_train_wav = sum(1 for _ in (DATA / "train" / "audio").rglob("*.wav"))
has_public = (DATA / "public_test" / "public_test.jsonl").exists() and (DATA / "public_test" / "audio").is_dir()
print("data :", DATA, f"| train wav {n_train_wav} | public_test {'yes' if has_public else 'NO'}")
print("code :", CODE_SRC, "->", CODE)
assert n_train_wav == 5036, f"expected 5,036 train clips, found {n_train_wav} (incomplete upload?)"
if PUBLIC and not has_public:
    print("public_test not attached -> PUBLIC = False"); PUBLIC = False
""")

code(r"""
# ---- helpers: run a command with live output + a log file; run one command per GPU in parallel ------------------
LOGS = OUT / "logs"
_QUIET = ("Setting `pad_token_id`", "The following generation flags")

def run(cmd, log, env=None):
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    with open(LOGS / log, "a") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=CODE,
                             env=env or os.environ.copy())
        for line in p.stdout:
            lf.write(line); lf.flush()
            if not line.startswith(_QUIET):
                print(line, end="", flush=True)
        rc = p.wait()
    if rc:
        raise RuntimeError(f"{log}: exit code {rc} — see {LOGS / log}")

def run_parallel(jobs, poll=30):
    # jobs: [(cmd, log name, gpu index)]
    procs = []
    for cmd, log, gpu in jobs:
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        lf = open(LOGS / log, "a")
        print(f"[gpu {gpu}] $", " ".join(map(str, cmd)), flush=True)
        procs.append((subprocess.Popen([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT, text=True,
                                       cwd=CODE, env=env), log, lf))
    def tail(log):
        lines = [l for l in open(LOGS / log, errors="replace").read().splitlines() if l.strip() and not l.startswith(_QUIET)]
        return lines[-1][:160] if lines else ""
    while any(p.poll() is None for p, _, _ in procs):
        time.sleep(poll)
        print(" | ".join(f"{log}: {tail(log)}" for _, log, _ in procs), flush=True)
    for p, log, lf in procs:
        lf.close()
        if p.returncode:
            print(open(LOGS / log, errors="replace").read()[-3000:])
            raise RuntimeError(f"{log}: exit code {p.returncode}")

def section(title):
    print(f"\n===== {time.strftime('%H:%M:%S')} {title} =====", flush=True)

T0 = time.time()
""")

code(r"""
# ---- install qwen-asr without touching Kaggle's CUDA torch ---------------------------------------------------------
import importlib.metadata as md
section("install")
pins = []
for pkg in ("torch", "torchvision", "torchaudio"):
    try:
        pins.append(f"{pkg}=={md.version(pkg).split('+')[0]}")
    except md.PackageNotFoundError:
        pass
Path("/tmp/constraints.txt").write_text("\n".join(pins) + "\n")
print("keeping:", pins)
pip = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
r = subprocess.run(pip + ["qwen-asr==0.0.6", "peft>=0.17,<0.19", "-c", "/tmp/constraints.txt"])
if r.returncode:
    print("full install failed -> minimal install (no gradio/flask/demo extras)")
    subprocess.run(pip + ["--no-deps", "qwen-asr==0.0.6"], check=True)
    subprocess.run(pip + ["transformers==4.57.6", "accelerate==1.12.0", "peft>=0.17,<0.19", "qwen-omni-utils",
                          "librosa", "soundfile", "pytz", "sox", "-c", "/tmp/constraints.txt"], check=True)

CHECK = r'''
import torch, transformers, peft
from qwen_asr import Qwen3ASRModel
n = torch.cuda.device_count()
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| transformers", transformers.__version__, "| peft", peft.__version__)
print("gpus:", n, [torch.cuda.get_device_name(i) for i in range(n)])
'''
r = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True)
if r.returncode and ("nagisa" in r.stderr or "soynlp" in r.stderr):
    # only the forced aligner (Japanese/Korean word splitting) needs these; install, else stub them out
    if subprocess.run(pip + ["nagisa==0.2.11", "soynlp==0.0.493"]).returncode:
        stub = WORK / "stubs"
        for m in ("nagisa", "soynlp"):
            (stub / m).mkdir(parents=True, exist_ok=True); (stub / m / "__init__.py").write_text("")
        os.environ["PYTHONPATH"] = f"{stub}:{os.environ.get('PYTHONPATH', '')}"
        print("stubbed nagisa/soynlp (unused by ASR)")
    r = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, env=os.environ.copy())
print(r.stdout, r.stderr[-3000:])
assert r.returncode == 0, "qwen-asr import failed (see above)"
NGPU = int(r.stdout.split("gpus:")[1].split()[0])
torch_now = r.stdout.split()[1].split("+")[0]
assert pins and pins[0] == f"torch=={torch_now}", f"pip replaced Kaggle's torch: {pins[0]} -> {torch_now}"
if NGPU < 2:
    print(f"WARNING: {NGPU} GPU visible — set Accelerator to 'GPU T4 x2'. Continuing on {NGPU}.")
NGPU = max(1, NGPU)
subprocess.run(f"{sys.executable} -m pip freeze > {OUT / 'pip_freeze.txt'}", shell=True)
""")

code(r"""
# ---- download Qwen3-ASR once (both training processes then load it from disk) ---------------------------------------
section("download Qwen3-ASR")
token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")      # optional: Add-ons -> Secrets -> HF_TOKEN
except Exception:
    pass
if token:
    os.environ["HF_TOKEN"] = token
r = subprocess.run([sys.executable, "-c", f"from huggingface_hub import snapshot_download; "
                    f"print(snapshot_download({ASR_REPO!r}, revision={ASR_REV!r}))"],
                   capture_output=True, text=True, check=True, env=os.environ.copy())
MODEL = Path(r.stdout.strip().splitlines()[-1])
print("model:", MODEL, f"{sum(f.stat().st_size for f in MODEL.rglob('*') if f.is_file()) / 2**30:.1f} GB")
""")

code(r"""
# ---- manifests: A = train partition (from the code dataset, checked), B = all train clips --------------------------
section("manifests")
from lifemate.config import load_config
from lifemate.asr.base import backend_name
from lifemate.data.loader import load_samples, gold_transcripts
from lifemate.data.splits import load_splits

split = load_splits(ART / "splits" / "v1.json")
MAN_A, MAN_B = ART / "asr_ft", ART / "asr_ft_all"
rows_a = [json.loads(l) for l in open(MAN_A / "train.jsonl", encoding="utf-8")]
rows_h = [json.loads(l) for l in open(MAN_A / "holdout.jsonl", encoding="utf-8")]
held_out = set(split["dev"]) | set(split["audit"])
assert not {r["sample_id"] for r in rows_a + rows_h} & held_out, "dev/audit clips in Model A's manifest"
run([sys.executable, "kaggle/build_all_manifest.py", "--out", MAN_B], "manifest_b.log")
n_a, n_b = len(rows_a), sum(1 for _ in open(MAN_B / "train.jsonl", encoding="utf-8"))
STEPS_B = round(STEPS_A * n_b / n_a)
SAVE_B = [round(s * n_b / n_a) for s in SAVE_A]
print(f"Model A: {n_a} train + {len(rows_h)} holdout clips, {STEPS_A} steps, saves {SAVE_A}")
print(f"Model B: {n_b} clips (all), {STEPS_B} steps, saves {SAVE_B}  (same single pass over its data)")

GOLD = {k: v for s in load_samples(DATA, "train") for k, v in gold_transcripts(s).items()}
CACHE_CFG = {"kgbase": "kaggle_kgbase_names", "ftA-s100": "kaggle_ftA_s100_names", "ftA-s200": "kaggle_ftA_s200_names",
             f"ftB-s{SAVE_B[0]}": f"kaggle_ftB_s{SAVE_B[0]}_names", f"ftB-s{SAVE_B[-1]}": f"kaggle_ftB_s{SAVE_B[-1]}_names"}
_base = backend_name(load_config(CODE / "configs" / "today_names.yaml"))      # ...-7278e1e-ctxnames
CACHE_NAME = {k: (backend_name(load_config(CODE / "configs" / f"{v}.yaml")) if (CODE / "configs" / f"{v}.yaml").exists()
                  else _base.replace("-ctxnames", f"-{k}-ctxnames")) for k, v in CACHE_CFG.items()}
print(json.dumps(CACHE_NAME, indent=1))
""")

code(r"""
# ---- shared steps: train on all GPUs, transcribe with one shard per GPU, WER ----------------------------------------
import re, unicodedata

def train(name, manifest, steps, save, extra=(), limit=0):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={NGPU}", "kaggle/asr_ddp.py",
           "--model", MODEL, "--manifest", manifest, "--out", OUT / name, "--steps", steps, "--save", *save,
           "--micro", MICRO, "--accum", ACCUM, "--lr", LR, "--r", LORA_R, "--alpha", LORA_ALPHA, "--dtype", DTYPE,
           "--context_mode", "names", *extra]
    if GRAD_CKPT:
        cmd.append("--grad_ckpt")
    if limit:
        cmd += ["--limit", limit]
    run(cmd, f"train_{name}.log")

def transcribe(tag, adapter, split_name, part, dest, limit=0):
    shards = [WORK / "shards" / f"{tag}.{split_name}.{part}.{g}.jsonl" for g in range(NGPU)]
    jobs = []
    for g, sh in enumerate(shards):
        cmd = [sys.executable, "kaggle/asr_transcribe.py", "--model", MODEL, "--dtype", DTYPE, "--split", split_name,
               "--part", part, "--context_mode", "names", "--shard", g, "--nshards", NGPU, "--out", sh]
        if adapter:
            cmd += ["--adapter", adapter]
        if limit:
            cmd += ["--limit", limit]
        jobs.append((cmd, f"asr_{tag}_{split_name}_{part}_gpu{g}.log", g))
    run_parallel(jobs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = [l for sh in shards for l in open(sh, encoding="utf-8") if l.strip()]
    dest.write_text("".join(rows), encoding="utf-8")
    print(f"{tag}: {len(rows)} transcripts -> {dest}")
    return {json.loads(l)["audio"]: json.loads(l)["text"] for l in rows}

def _words(t):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", t).lower())

def _ed(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def wer(hyp):
    e = n = 0
    for p, h in hyp.items():
        g = _words(GOLD[p]); e += _ed(g, _words(h)); n += len(g)
    return round(100 * e / max(n, 1), 2)

RESULTS = {"wer_dev": {}, "notes": []}
""")

code(r"""
# ---- SMOKE (MODE = "smoke"): tiny run of every step on both GPUs, then stop -----------------------------------------
if MODE == "smoke":
    section("smoke: training 4 steps on 32 clips")
    train("smoke_A", MAN_A, 4, [2, 4], limit=32)
    section("smoke: transcribing 8 dev clips per GPU, base and adapter")
    hb = transcribe("smoke_base", None, "train", "dev", WORK / "smoke" / "base.jsonl", limit=8)
    ha = transcribe("smoke_A4", OUT / "smoke_A" / "step4", "train", "dev", WORK / "smoke" / "a4.jsonl", limit=8)
    for p in list(hb)[:3]:
        print(f"gold : {GOLD[p]}\nbase : {hb[p]}\nA s4 : {ha[p]}\n")
    print(f"SMOKE OK in {(time.time() - T0) / 60:.1f} min — check the memory (mem ...G) lines above, then set "
          f"MODE = 'full' and Save Version -> Save & Run All.")
    shutil.rmtree(OUT / "smoke_A", ignore_errors=True)
""")

code(r"""
# ---- 1. Model A: train partition only ----------------------------------------------------------------------------
if MODE == "full":
    section("Model A training (train partition)")
    train("A", MAN_A, STEPS_A, SAVE_A)
""")

code(r"""
# ---- 2. dev transcripts + WER: base vs A-step100 vs A-step200 --------------------------------------------------------
if MODE == "full":
    for tag, adapter in [("kgbase", None)] + [(f"ftA-s{s}", OUT / "A" / f"step{s}") for s in SAVE_A]:
        section(f"dev transcripts: {tag}")
        hyp = transcribe(tag, adapter, "train", "dev", OUT / "asr" / CACHE_NAME[tag] / "train.jsonl")
        RESULTS["wer_dev"][tag] = wer(hyp)
        print(f"dev WER {tag}: {RESULTS['wer_dev'][tag]}%")
    best = min((t for t in RESULTS["wer_dev"] if t.startswith("ftA")), key=RESULTS["wer_dev"].get)
    RESULTS["best_A_by_dev_wer"] = best
    print("dev WER:", RESULTS["wer_dev"], "| best A by WER:", best, "(the agent's dev score decides, tomorrow)")
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1))
""")

code(r"""
# ---- 3. Model B: all train clips (final-fit candidate; never evaluate it on dev/audit) ------------------------------
if MODE == "full" and RUN_B:
    section("Model B training (ALL train clips)")
    train("B", MAN_B, STEPS_B, SAVE_B, extra=["--no_holdout"])
""")

code(r"""
# ---- 4. public test transcripts with both Model B checkpoints --------------------------------------------------------
if MODE == "full" and RUN_B and PUBLIC:
    for s in SAVE_B:
        tag = f"ftB-s{s}"
        section(f"public transcripts: {tag}")
        transcribe(tag, OUT / "B" / f"step{s}", "public_test", "all", OUT / "asr" / CACHE_NAME[tag] / "public_test.jsonl")
""")

code(r"""
# ---- summary -------------------------------------------------------------------------------------------------------
section("summary")
RESULTS["minutes_total"] = round((time.time() - T0) / 60, 1)
for name in ("A", "B"):
    h = OUT / name / "history.json"
    if h.exists():
        RESULTS[f"history_{name}"] = {k: v for k, v in json.loads(h.read_text()).items() if k != "loss"}
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=1))
lines = [f"# Kaggle ASR run ({MODE}) — {time.strftime('%Y-%m-%d %H:%M')}", "",
         f"- total {RESULTS['minutes_total']} min on {NGPU} GPU(s), dtype {DTYPE}, effective batch {MICRO * ACCUM * NGPU}",
         f"- dev WER (names context): {RESULTS.get('wer_dev')}", f"- best A by dev WER: {RESULTS.get('best_A_by_dev_wer')}",
         "- Model B = all train clips incl. dev/audit: final fit only, never score it on dev/audit", "",
         "Next (Colab): copy out/asr/* to artifacts/asr/, then run_eval --config configs/kaggle_<tag>_names.yaml "
         "--model artifacts/models/v1-epoch2 --conditions C for kgbase, ftA-s100, ftA-s200; compare with "
         "lifemate.eval.report ... --vs (paired bootstrap).", "", "## files"]
size = 0
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        size += f.stat().st_size
        lines.append(f"- {f.relative_to(OUT)} ({f.stat().st_size / 2**20:.1f} MB)")
(OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:8]))
print(f"output size {size / 2**30:.2f} GB (Kaggle keeps up to 20 GB)")
assert size < 19 * 2**30
""")


def main() -> None:
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    out = Path(__file__).resolve().parent / "lifemate_asr_overnight.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
