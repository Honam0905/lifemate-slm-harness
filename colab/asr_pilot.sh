#!/usr/bin/env bash
# ASR LoRA pilot on one Colab A100 (agent fixed = v1 epoch 2, names-only ASR context as on 4 Nov, website 31.60).
#  1. LoRA on Qwen3-ASR-1.7B decoder attention, 200 steps x 16 clips (~1 pass) on the train-partition manifest,
#     checkpoints at steps 100 and 200 (holdout loss logged). The v1 agent is merged at the same time.
#  2. Merge both ASR checkpoints, transcribe the dev clips with names-only context (baseline ASR too), print WERs,
#     keep the checkpoint with the lower dev WER.
#  3. Dev condition C for the agent with that ASR and with the baseline ASR, two at a time, then the paired report.
# Expects the code at /content/zalo and the competition data at /content/data/train (full train folder).
set -uo pipefail
cd /content/zalo
export HF_HOME=/content/hf LIFEMATE_DATA_ROOT=/content/data LIFEMATE_ARTIFACTS=/content/artifacts PYTHONUNBUFFERED=1 UV_CACHE_DIR=/content/uv-cache
A=/content/artifacts; M=$A/models/v1-epoch2; L=/content/logs
mkdir -p $A/splits $A/gold $L
cp -n artifacts/splits/v1.json $A/splits/ && cp -n artifacts/gold/train_hashes.json $A/gold/ || exit 1
[ -d /content/venv-asr ] || bash colab/setup_vm.sh || exit 1
uv pip install -q --python /content/venv-asr/bin/python "peft<0.19" || exit 1
echo "=== $(date +%T) versions"
/content/venv-asr/bin/python -c "import torch, transformers, peft; print('asr env: torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"

echo "=== $(date +%T) agent merge in the background (log $L/agent_merge.log)"
(
  if [ ! -f $M/config.json ]; then
    source /content/venv/bin/activate
    python - <<'PY' || exit 1
import peft, torch, transformers
v = (torch.__version__, transformers.__version__, peft.__version__)
print("merge env: torch %s | transformers %s | peft %s" % v)
assert torch.__file__.startswith("/usr/"), f"venv has its own torch at {torch.__file__}"
assert v[1:] == ("5.22.0", "0.21.2"), f"expected transformers 5.22.0 + peft 0.21.2, got {v[1:]}"
PY
    python -m lifemate.train.merge --ckpt artifacts/vm/adapters/v1/ckpt-epoch2 --out $M || exit 1
  fi
) > $L/agent_merge.log 2>&1 &
MERGE_PID=$!

source /content/venv-asr/bin/activate
echo "=== $(date +%T) ASR manifest (train partition only)"
python scripts/build_asr_manifest.py || exit 1
echo "=== $(date +%T) ASR LoRA training"
python scripts/train_asr_lora.py train --out $A/asr_ft/run1 --steps 200 --save 100 200 || exit 1
for s in 100 200; do
  echo "=== $(date +%T) merge ASR step $s"
  python scripts/train_asr_lora.py merge --ckpt $A/asr_ft/run1/step$s --out $A/models/asr-ft-s$s || exit 1
done

echo "=== $(date +%T) dev transcripts with names-only context: baseline ASR, then step 100 and step 200"
python scripts/build_asr_cache.py --config configs/today_names.yaml --split train --part dev 2>&1 | tee $L/wer_base.txt | grep -E "WER|clips,|Traceback" || exit 1
for s in 100 200; do
  python scripts/build_asr_cache.py --config configs/asrft_s${s}_names.yaml --split train --part dev 2>&1 | tee $L/wer_s$s.txt | grep -E "WER|clips,|Traceback" || exit 1
done
wer() { grep -oE "WER vs annotator[^:]*: [0-9.]+" $1 | grep -oE "[0-9.]+$"; }
W0=$(wer $L/wer_base.txt); W1=$(wer $L/wer_s100.txt); W2=$(wer $L/wer_s200.txt)
echo "dev WER: baseline $W0 | step100 $W1 | step200 $W2"
BEST=$(python -c "print('s100' if float('$W1') < float('$W2') else 's200')")
echo "=== $(date +%T) best ASR checkpoint by dev WER: $BEST"

wait $MERGE_PID; echo "agent merge exit $?"; tail -3 $L/agent_merge.log
python - <<PY || exit 1
import json
c = json.load(open("$M/config.json"))
print("merged agent rope_theta", c.get("rope_theta"))
assert c.get("rope_theta") == 5000000
PY

run() {
  python -m lifemate.eval.run_eval --config configs/$1.yaml --name pilot-$1 --model $M --conditions C > $L/eval-$1.log 2>&1
  echo $? > $L/eval-$1.rc
}
echo "=== $(date +%T) agent dev C: names-only baseline ASR vs fine-tuned ASR ($BEST), two at a time"
run today_names & sleep 90; run asrft_${BEST}_names & wait
for c in today_names asrft_${BEST}_names; do
  [ "$(cat $L/eval-$c.rc)" = 0 ] || { echo "=== re-run $c alone"; tail -5 $L/eval-$c.log; run $c; }
  echo "=== $c rc $(cat $L/eval-$c.rc)"; grep "dev.C: primary" $L/eval-$c.log | tail -1
done
echo "=== fine-tuned ASR vs baseline ASR (paired)"
python -m lifemate.eval.report $A/eval/pilot-asrft_${BEST}_names/dev.C/rows.jsonl --vs $A/eval/pilot-today_names/dev.C/rows.jsonl | tail -6
echo "=== $(date +%T) asr pilot done"
