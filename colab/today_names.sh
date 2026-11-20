#!/usr/bin/env bash
# 4 Nov v1 tuning, second half: the full-reply ASR context made the ASR copy assistant replies into user transcripts
# (dev WER 28.1% -> 31.1%), so those arms were dropped. This builds the names-only context transcripts for dev, waits
# for the v1 / date-table evaluations started by today_tune.sh, then evaluates names-only and date + names-only.
set -uo pipefail
cd /content/zalo
export HF_HOME=/content/hf LIFEMATE_DATA_ROOT=/content/data LIFEMATE_ARTIFACTS=/content/artifacts PYTHONUNBUFFERED=1 UV_CACHE_DIR=/content/uv-cache
A=/content/artifacts; M=$A/models/v1-epoch2; L=/content/logs
source /content/venv-asr/bin/activate
echo "=== dev transcripts with names-only ASR context"
python scripts/build_asr_cache.py --config configs/today_names.yaml --split train --part dev || exit 1

run() {
  python -m lifemate.eval.run_eval --config configs/$1.yaml --name today-$1 --model $M --conditions C > $L/eval-$1.log 2>&1
  echo $? > $L/eval-$1.rc
}
echo "=== waiting for the v1 and date-table evaluations"
until [ -f $L/eval-v1.rc ] && [ -f $L/eval-today_date.rc ]; do sleep 10; done
echo "=== names-only evaluations, two at a time"
run today_names & sleep 90; run today_date_names & wait
for c in v1 today_date today_names today_date_names; do
  [ "$(cat $L/eval-$c.rc)" = 0 ] || { echo "=== re-run $c alone (rc $(cat $L/eval-$c.rc))"; tail -5 $L/eval-$c.log; run $c; }
done
for c in v1 today_date today_names today_date_names; do echo "=== $c rc $(cat $L/eval-$c.rc)"; tail -1 $L/eval-$c.log; done
for c in today_date today_names today_date_names; do
  echo "=== $c vs v1 (paired)"
  python -m lifemate.eval.report $A/eval/today-$c/dev.C/rows.jsonl --vs $A/eval/today-v1/dev.C/rows.jsonl | tail -6
done
echo "=== today names done"
