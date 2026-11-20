#!/usr/bin/env bash
# 4 Nov v1 tuning on one Colab A100 (no retraining): merge the epoch-2 adapter, fresh dev transcripts (plain and with
# ASR context), then dev condition C for v1 / date table / ASR context / both with the same model, all in the
# inference env (transformers 4.57, as the public run). The v1 arm is re-run on the fresh transcripts so every
# comparison is paired. Two evaluations share the GPU at a time; a failed one (e.g. out of memory) is re-run alone.
# Expects the code at /content/zalo and the data at /content/data/{train,public_test}.
set -uo pipefail
cd /content/zalo
export HF_HOME=/content/hf LIFEMATE_DATA_ROOT=/content/data LIFEMATE_ARTIFACTS=/content/artifacts PYTHONUNBUFFERED=1 UV_CACHE_DIR=/content/uv-cache
A=/content/artifacts; M=$A/models/v1-epoch2; L=/content/logs
mkdir -p $A/splits $A/gold $L
cp -n artifacts/splits/v1.json $A/splits/ && cp -n artifacts/gold/train_hashes.json $A/gold/ || exit 1
[ -d /content/venv-asr ] || bash colab/setup_vm.sh || exit 1

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
  deactivate
fi

source /content/venv-asr/bin/activate
python - <<PY || exit 1
import json
c = json.load(open("$M/config.json"))
print("merged config rope_theta", c.get("rope_theta"))
assert c.get("rope_theta") == 5000000, "rope_theta missing: the 4.57 env would fall back to 10000"
PY
echo "=== dev transcripts (plain, then with ASR context)"
python scripts/build_asr_cache.py --split train --part dev || exit 1
python scripts/build_asr_cache.py --config configs/today_context.yaml --split train --part dev || exit 1

run() {
  python -m lifemate.eval.run_eval --config configs/$1.yaml --name today-$1 --model $M --conditions C > $L/eval-$1.log 2>&1
  echo $? > $L/eval-$1.rc
}
echo "=== dev evaluations, two at a time"
run v1 & sleep 90; run today_date & wait
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
run today_context & sleep 90; run today_both & wait
for c in v1 today_date today_context today_both; do
  [ "$(cat $L/eval-$c.rc)" = 0 ] || { echo "=== re-run $c alone (rc $(cat $L/eval-$c.rc))"; tail -5 $L/eval-$c.log; run $c; }
done
for c in v1 today_date today_context today_both; do echo "=== $c rc $(cat $L/eval-$c.rc)"; tail -1 $L/eval-$c.log; done
for c in today_date today_context today_both; do
  echo "=== $c vs v1 (paired)"
  python -m lifemate.eval.report $A/eval/today-$c/dev.C/rows.jsonl --vs $A/eval/today-v1/dev.C/rows.jsonl | tail -14
done
echo "=== today tune done"
