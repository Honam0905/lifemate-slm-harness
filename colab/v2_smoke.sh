#!/usr/bin/env bash
# v2 smoke test on a Colab A100 (5 Nov). Everything tonight's full run needs, end to end, on tiny amounts:
#   1 train-partition ASR transcripts (base Qwen3-ASR, names context) -> the "asr" training variant
#   2 trajectories (final turns gold+asr, earlier turns, synthetic)    3 tokenize (fails loudly over budget)
#   4 training: 6 steps longest-first (peak memory) -> resume to 8 (checkpoint/resume) -> 4 steps with
#     gradient checkpointing off below 6K tokens (speed)               5 merge   6 dev eval (6 samples, C)
#   7 public prediction (3 samples) from the cached Model B transcripts, in BOTH envs
# Run detached:  colab/vmrun.sh v2smoke train -- bash colab/v2_smoke.sh   (log /content/logs/v2smoke.log)
set -uo pipefail
cd /content/zalo
export HF_HOME=/content/hf LIFEMATE_DATA_ROOT=/content/data LIFEMATE_ARTIFACTS=/content/artifacts PYTHONUNBUFFERED=1 UV_CACHE_DIR=/content/uv-cache
A=/content/artifacts
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
fail() { echo "=== SMOKE FAILED at: $*"; exit 1; }

[ -d /content/venv-asr ] && [ -d /content/venv ] || bash colab/setup_vm.sh || fail setup

step "1 train-partition ASR transcripts (base ASR, names context; dev clips already cached)"
/content/venv-asr/bin/python scripts/build_asr_cache.py --config configs/v2.yaml --split train --part train || fail asr
/content/venv-asr/bin/python scripts/build_asr_cache.py --config configs/v2.yaml --split train --part audit || fail asr_audit   # build_trajectories wants every train-split clip

source /content/venv/bin/activate
python -c "import torch, transformers, peft; print('train env: torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"

step "2 trajectories"
python scripts/build_trajectories.py --config configs/v2.yaml --variants gold asr --parts train dev || fail trajectories
python scripts/build_v2_data.py --config configs/v2.yaml --asr || fail v2_data

step "3 tokenize"
python scripts/tokenize_trajectories.py --config configs/v2.yaml --parts train dev --variants gold asr earlier synth --show 1 || fail tokenize

step "4a training: 6 steps, longest sequences first (peak memory), always checkpointing"
rm -rf $A/runs/v2-smoke
python -m lifemate.train.sft --config configs/v2.yaml --run v2-smoke --max_steps 6 --longest_first --eval_limit 8 || fail train
step "4b resume from step 6 to step 8"
python -m lifemate.train.sft --config configs/v2.yaml --run v2-smoke --max_steps 8 --resume --eval_limit 8 || fail resume
step "4c speed: 4 steps in normal order with checkpointing off below 6000 tokens (separate run dir)"
rm -rf $A/runs/v2-speed
python -m lifemate.train.sft --config configs/v2_ckpt6k.yaml --run v2-speed --max_steps 4 --eval_limit 2 || echo "=== 4c failed (likely out of memory without checkpointing): keep ckpt_min_tokens 0"
step "4d speed reference: 4 steps in normal order, always checkpointing"
rm -rf $A/runs/v2-speedref
python -m lifemate.train.sft --config configs/v2.yaml --run v2-speedref --max_steps 4 --eval_limit 2 || fail speedref

step "5 merge the step-8 adapter"
rm -rf $A/models/v2-smoke
python -m lifemate.train.merge --ckpt $A/runs/v2-smoke/ckpt-step8 --out $A/models/v2-smoke || fail merge

step "6 dev eval, 6 samples, condition C, fine-tuned ASR (Model A s100) transcripts, date table on"
python -m lifemate.eval.run_eval --config configs/v2_ftA_s100.yaml --name v2-smoke --model $A/models/v2-smoke --conditions C --limit 6 || fail eval

step "7a public prediction, 3 samples, cached Model B transcripts, TRAIN env (transformers 5.22)"
PUB=$A/asr/qwen3_asr-Qwen3-ASR-1.7B-7278e1e-ftB-s157-ctxnames/public_test.jsonl
python -m lifemate.infer.run --config configs/v2_ftB_s157.yaml --data /content/data/public_test --model $A/models/v2-smoke \
  --asr_cache $PUB --out $A/submissions/v2-smoke-trainenv/submission.jsonl --limit 3 || echo "=== 7a failed in the train env"
deactivate
step "7b the same in the ASR/inference env (transformers 4.57.6, the Docker env)"
/content/venv-asr/bin/python -m lifemate.infer.run --config configs/v2_ftB_s157.yaml --data /content/data/public_test --model $A/models/v2-smoke \
  --asr_cache $PUB --out $A/submissions/v2-smoke-asrenv/submission.jsonl --limit 3 || fail predict_asrenv
cmp -s $A/submissions/v2-smoke-trainenv/submission.jsonl $A/submissions/v2-smoke-asrenv/submission.jsonl \
  && echo "=== 7 both envs wrote identical predictions" || echo "=== 7 predictions differ between envs (or 7a failed)"
head -c 600 $A/submissions/v2-smoke-asrenv/submission.jsonl; echo

step "SMOKE OK"
