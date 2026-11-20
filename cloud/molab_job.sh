#!/usr/bin/env bash
# molab job (free RTX Pro 6000 96 GB): several training recipes BACK TO BACK in one session, each scored on dev C with
# the ASR round-2 dev transcripts (A2-s600) and compared, paired, with v2r (website 41.75) evaluated in the same session
# on the same GPU. Then the public prediction of the best recipe (B2-s941 transcripts) if it beats v2r on dev.
# Every stage is timestamped; a failed recipe is logged and the job moves on to the next one.
#
#     SMOKE=1 bash cloud/molab_job.sh       # ~20-25 min: every recipe 3 steps + 12-sample evals + 5 public samples
#     bash cloud/molab_job.sh               # full: RECIPES (default "v2rb v2rb_full v2rb_r128", ~8-8.5 h)
#     FINAL=1 RECIPES=v2rb_all bash cloud/molab_job.sh   # final fit: all data, no dev eval, public prediction
# Env: RECIPES, SMOKE, MAX_STEPS (0 = full), FINAL, PUBLIC (best | none | <recipe>), SKIP_INSTALL=1, PYTHON,
#      LIFEMATE_PUSH_REPO=<user>/<private repo> + HF_TOKEN in the environment = push each epoch's checkpoint to the Hub.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export HF_HOME=${HF_HOME:-$ROOT/hf} LIFEMATE_DATA_ROOT=$ROOT/data LIFEMATE_ARTIFACTS=$ROOT/artifacts
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=${PYTHON:-python}
SMOKE=${SMOKE:-0}; FINAL=${FINAL:-0}; PUBLIC=${PUBLIC:-best}
if [ "$FINAL" = 1 ]; then RECIPES=${RECIPES:-v2rb_all}; else RECIPES=${RECIPES:-"v2rb v2rb_full v2rb_r128"}; fi
MAX_STEPS=${MAX_STEPS:-0}; LIMIT=0; PUBLIMIT=0
[ "$SMOKE" = 1 ] && { MAX_STEPS=3; LIMIT=12; PUBLIMIT=5; }
EVAL_CFG=configs/v2_ftA2_s600.yaml     # dev C: round-2 Model A2 step 600 transcripts (v1 38.21, v2r 44.24 on Colab)
PUB_CFG=configs/v2_ftB2_s941.yaml      # public: round-2 Model B2 step 941 transcripts (the 38.63 / 41.75 ones)
V1_ROWS=artifacts/vm/asr2_eval/eval/kg-ftA2_s600/dev.C/rows.jsonl          # v1, Colab A100
V2R_COLAB=artifacts/vm/v2r_public/eval/v2r-ftA2_s600/dev.C/rows.jsonl     # v2r, Colab A100 (44.24)
V2R_ADAPTER=artifacts/vm/adapters/v2r/ckpt-epoch2
A=$ROOT/artifacts; L=$ROOT/logs; R=$ROOT/results; mkdir -p "$L" "$R"
SUMMARY=$R/summary.txt
T0=$(date +%s)
step() { echo; echo "=== $(date '+%F %T') (+$(( ($(date +%s) - T0) / 60 )) min) $*"; }
fail() { step "JOB FAILED at: $*"; exit 1; }
note() { echo "$*" | tee -a "$SUMMARY"; }
for r in $RECIPES; do [ -f "configs/$r.yaml" ] || fail "no configs/$r.yaml"; done
note "=== $(date '+%F %T') recipes: $RECIPES | SMOKE=$SMOKE FINAL=$FINAL MAX_STEPS=$MAX_STEPS"

step "0 machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "no GPU attached (notebook specs button -> GPU)"
nproc; free -g | head -2; df -h "$ROOT" | tail -1
gpu_check() { $PY - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0), "| bf16", torch.cuda.is_bf16_supported())
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); y = (x @ x).float().sum().item()
print("bf16 matmul on the GPU: OK")
PY
}
PIP() { if $PY -m pip --version > /dev/null 2>&1; then $PY -m pip install -q --disable-pip-version-check "$@"; else uv pip install -q --python "$PY" "$@"; fi; }
if ! gpu_check; then
  step "0b this PyTorch cannot use the GPU: installing torch 2.11.0 for CUDA 12.8 (Blackwell)"
  PIP "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 || fail "torch install"
  gpu_check || fail "PyTorch still cannot use this GPU"
fi
if [ "${SKIP_INSTALL:-0}" != 1 ]; then
  step "1 pinned training libraries (keeping molab's own torch)"
  PIP "transformers==5.22.0" "peft==0.21.2" "accelerate==1.15.0" "torchao>=0.16" safetensors "huggingface_hub[hf_xet]" pyyaml \
      || fail install
fi
$PY -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)" || fail imports

step "2 MiniCPM5-2B download"
$PY -c "from huggingface_hub import snapshot_download as s; print(s('openbmb/MiniCPM5-2B', revision='f97400052a43d642bbc6e9975e2397e3ae6a6b52'))" || fail download

cfgval() { $PY -c "import sys; sys.path.insert(0, '.'); from lifemate.config import load_config, artifacts_dir; c = load_config('$1'); $2"; }
merge_rope_ok() { $PY -c "import json; c = json.load(open('$1/config.json')); print('rope_theta', c.get('rope_theta')); assert c.get('rope_theta') == 5000000"; }
devC() {  # $1 = run name, $2 = merged model dir -> rows in artifacts/eval/$1/dev.C
  $PY -m lifemate.eval.run_eval --config $EVAL_CFG --name "$1" --model "$2" --conditions C --limit $LIMIT > "$L/$1.eval.log" 2>&1 \
    || { tail -5 "$L/$1.eval.log"; return 1; }
  grep "dev.C: primary" "$L/$1.eval.log" | tail -1
}

BASE=$V2R_COLAB; BASE_LABEL=v2r-colab
if [ "$FINAL" != 1 ]; then
  step "3 baseline: v2r (website 41.75) on THIS GPU with the A2-s600 dev transcripts"
  MV=$A/models/v2r-epoch2
  [ -f "$MV/config.json" ] || $PY -m lifemate.train.merge --config configs/v2r.yaml --ckpt $V2R_ADAPTER --out "$MV" > "$L/v2r.merge.log" 2>&1 \
    || { tail -5 "$L/v2r.merge.log"; fail "v2r merge"; }
  merge_rope_ok "$MV" || fail "v2r merge rope_theta"
  if devC v2r-molab "$MV"; then
    BASE=$A/eval/v2r-molab/dev.C/rows.jsonl; BASE_LABEL=v2r-molab
    $PY cloud/compare.py v2r-molab "$BASE" --vs v2r-colab=$V2R_COLAB --vs v1-colab=$V1_ROWS --out "$R/v2r-molab/paired.json" | tee -a "$SUMMARY"
  else
    note "v2r-molab baseline eval FAILED: comparing against the Colab v2r rows instead"
  fi
fi

recipe() {  # $1 = recipe (configs/$1.yaml); the run directory is artifacts/runs/$1
  local r=$1 CFG=configs/$1.yaml
  step "[$r] 1/6 renaming + tokenisation"
  local VARIANTS TOKDIR LAST
  VARIANTS=$(cfgval "$CFG" "v = c['train']['variants']; print(' '.join(['gold'] + [x for x in v if x != 'gold']))") || return 1
  TOKDIR=$(cfgval "$CFG" "print(artifacts_dir(c) / 'tokenized' / (c['train'].get('data_tag') or c['run_name']))") || return 1
  LAST=${VARIANTS##* }
  if [ ! -f "$TOKDIR/train.$LAST.pt" ]; then
    if cfgval "$CFG" "sys.exit(0 if c.get('rename') else 1)"; then
      $PY scripts/apply_rename.py --config "$CFG" --show 0 > "$L/$r.rename.log" 2>&1 || { tail -5 "$L/$r.rename.log"; return 1; }
      grep "^\[rename\] train" "$L/$r.rename.log"
    fi
    $PY scripts/tokenize_trajectories.py --config "$CFG" --parts train dev --variants $VARIANTS --show 0 > "$L/$r.tok.log" 2>&1 \
      || { tail -5 "$L/$r.tok.log"; return 1; }
  else
    echo "tokenized data already there: $TOKDIR"
  fi
  grep -E "none_keep|^\[tok\] train|within budget" "$L/$r.tok.log" 2>/dev/null | cut -c1-220

  step "[$r] 2/6 memory probe: 3 optimizer steps on the LONGEST sequences, with the recipe's own checkpointing setting"
  local TRAIN_CFG=$CFG
  rm -rf "$A/runs/$r-probe"
  if ! $PY -m lifemate.train.sft --config "$CFG" --run "$r-probe" --max_steps 3 --longest_first --eval_limit 2 > "$L/$r.probe.log" 2>&1; then
    echo "probe failed ($(grep -m1 -o 'OutOfMemoryError' "$L/$r.probe.log" || tail -1 "$L/$r.probe.log")) -> retrying with gradient checkpointing everywhere"
    TRAIN_CFG=configs/_molab_${r}_ckpt.yaml
    printf 'base: %s.yaml\ntrain:\n  ckpt_min_tokens: 0\n' "$r" > "$TRAIN_CFG"
    rm -rf "$A/runs/$r-probe"
    $PY -m lifemate.train.sft --config "$TRAIN_CFG" --run "$r-probe" --max_steps 3 --longest_first --eval_limit 2 > "$L/$r.probe.log" 2>&1 \
      || { tail -5 "$L/$r.probe.log"; rm -rf "$A/runs/$r-probe"; return 1; }
  fi
  grep -E "params:|step [0-9]+/" "$L/$r.probe.log" | tail -4
  rm -rf "$A/runs/$r-probe"

  step "[$r] 3/6 training ($TRAIN_CFG)$( [ "$MAX_STEPS" != 0 ] && echo ", stopping at step $MAX_STEPS")"
  $PY -m lifemate.train.sft --config "$TRAIN_CFG" --run "$r" --resume $( [ "$MAX_STEPS" != 0 ] && echo "--max_steps $MAX_STEPS" ) \
      2>&1 | tee "$L/$r.train.log" | grep --line-buffered -E "step [0-9]*(00|25|50|75)/|step [1-3]/|epoch|dev loss|resumed|done:|params:|loss:|Error|Traceback" \
      || true
  grep -q "done:" "$L/$r.train.log" || { tail -5 "$L/$r.train.log"; return 1; }
  local CKPT M
  CKPT=$(cat "$A/runs/$r/latest")

  step "[$r] 4/6 merge $CKPT"
  M=$A/models/$r-$CKPT
  [ -f "$M/config.json" ] || $PY -m lifemate.train.merge --config "$CFG" --ckpt "$A/runs/$r/$CKPT" --out "$M" > "$L/$r.merge.log" 2>&1 \
    || { tail -5 "$L/$r.merge.log"; return 1; }
  merge_rope_ok "$M" || return 1
  echo "$M" > "$R/.model_$r"

  if [ "$FINAL" != 1 ]; then
    step "[$r] 5/6 dev C (A2-s600 transcripts)"
    devC "$r" "$M" || return 1
    $PY cloud/compare.py "$r" "$A/eval/$r/dev.C/rows.jsonl" --vs $BASE_LABEL=$BASE --vs v2r-colab=$V2R_COLAB --vs v1-colab=$V1_ROWS \
        --out "$R/$r/paired.json" | tee -a "$SUMMARY"
  fi

  step "[$r] 6/6 packaging"
  mkdir -p "$R/$r"
  [ -d "$A/eval/$r" ] && cp -r "$A/eval/$r" "$R/$r/eval"
  if [ -f "$A/runs/$r/$CKPT/adapter_config.json" ]; then   # LoRA: the adapter (200 MB for r32, 800 MB for r128)
    (cd "$A/runs/$r" && $PY -c "
import zipfile, pathlib; z = zipfile.ZipFile('$ROOT/molab_adapter_$r.zip', 'w', zipfile.ZIP_STORED)
[z.write(f, f'$r/{f}') for f in pathlib.Path('$CKPT').rglob('*') if f.is_file() and f.name != 'trainer_state.pt']; z.close()")
    echo "adapter zip: $(du -h "$ROOT/molab_adapter_$r.zip" | cut -f1)"
  else
    echo "full fine-tuning: the 5 GB model stays on molab ($M); push it with LIFEMATE_PUSH_REPO + HF_TOKEN if it wins"
  fi
  return 0
}

for r in $RECIPES; do
  if recipe "$r"; then step "[$r] RECIPE OK"; else step "[$r] RECIPE FAILED (logs/$r.*.log)"; note "$r: FAILED (see logs/$r.*.log)"; fi
done

# public prediction: the best recipe on dev (largest paired gain over v2r, only if > 0), or every recipe in FINAL mode
PICK=""
if [ "$FINAL" = 1 ]; then PICK="$RECIPES"
elif [ "$PUBLIC" = best ]; then
  PICK=$($PY - <<PY
import json, pathlib
best = None
for p in sorted(pathlib.Path("$R").glob("*/paired.json")):
    j = json.loads(p.read_text())
    if j["name"] == "v2r-molab" or "$BASE_LABEL" not in j["vs"]:
        continue
    d = j["vs"]["$BASE_LABEL"]["mean_diff"]
    if (d > 0 or $SMOKE) and (best is None or d > best[1]):
        best = (j["name"], d)
print(best[0] if best else "")
PY
)
  if [ -n "$PICK" ]; then note "best recipe on dev: $PICK (paired gain over $BASE_LABEL > 0)"
  else note "no recipe beat $BASE_LABEL on dev: no new public prediction (v2r's 41.75 file stays the best)"; fi
elif [ "$PUBLIC" != none ]; then PICK=$PUBLIC
fi
for r in $PICK; do
  [ -f "$R/.model_$r" ] || { note "public $r: no merged model"; continue; }
  M=$(cat "$R/.model_$r")
  step "public prediction: $r + B2-s941 transcripts"
  PUB=$(cfgval $PUB_CFG "from lifemate.asr.base import backend_name; print(artifacts_dir(c) / 'asr' / backend_name(c) / 'public_test.jsonl')")
  SUB=$R/submissions/$r-ftB2-s941/submission.jsonl
  $PY -m lifemate.infer.run --config $PUB_CFG --data "$ROOT/data/public_test" --model "$M" --asr_cache "$PUB" --out "$SUB" \
      --limit $PUBLIMIT > "$L/$r.public.log" 2>&1 || { tail -5 "$L/$r.public.log"; note "public $r: FAILED"; continue; }
  grep "^\[infer" "$L/$r.public.log" | tail -2
  [ "$PUBLIMIT" = 0 ] && $PY -I scripts/check_submission.py "$SUB" "$ROOT/data" 2>&1 | tail -2
  note "public $r: $(wc -l < "$SUB") lines, $(grep -o 'latency median.*' "$L/$r.public.log" | tail -1), sha256 $(sha256sum "$SUB" | cut -c1-16)"
done

step "results zip (summary, dev rows, logs, submissions; adapters are separate zips)"
rm -rf "$R/logs"; cp -r "$L" "$R/logs"
OUTZIP=$ROOT/molab_results.zip
rm -f "$OUTZIP"
(cd "$ROOT" && $PY -c "
import zipfile, pathlib
z = zipfile.ZipFile('$OUTZIP', 'w', zipfile.ZIP_DEFLATED)
[z.write(f, f.relative_to('$ROOT')) for f in pathlib.Path('$R').rglob('*') if f.is_file() and not f.name.startswith('.')]
z.close(); print('wrote $OUTZIP', round(pathlib.Path('$OUTZIP').stat().st_size / 2**20, 1), 'MB')")
echo; cat "$SUMMARY"
if [ "$SMOKE" = 1 ]; then
  mv "$OUTZIP" "$ROOT/molab_results_smoke.zip"
  step "smoke cleanup: removing the 3-step runs, their models and evals (tokenized data and the merged v2r stay for the full run)"
  for r in $RECIPES; do rm -rf "$A/runs/$r" "$A/models/$r-"* "$A/eval/$r" "$ROOT/molab_adapter_$r.zip"; done
  rm -rf "$A/eval/v2r-molab" "$R" "$L" configs/_molab_*_ckpt.yaml
  echo "SMOKE OK: now run the full job (SMOKE=0)"
fi
step "JOB OK"
