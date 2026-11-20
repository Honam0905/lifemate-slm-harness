#!/usr/bin/env bash
# Download the ASR round-2 dev evaluation (eval rows, reports, logs, the public submission) and STOP the Colab VM.
#     bash colab/fetch_asr2_eval.sh            (KEEP_VM=1 to skip stopping)
set -uo pipefail
cd "$(dirname "$0")/.."
SESSION=${SESSION:-lifemate}
WORK=${WORK:-$PWD/.asr2_work}
OUT=artifacts/vm/asr2_eval; mkdir -p "$OUT"
R="ssh -F $WORK/ssh_config"
rsync -a -e "$R" colab:/content/logs/ "$OUT/logs/"
rsync -a -e "$R" --include='kg-ftA2*/***' --exclude='*' colab:/content/artifacts/eval/ "$OUT/eval/"
rsync -a -e "$R" colab:/content/artifacts/submissions/ "$OUT/submissions/" 2>/dev/null || true
echo "== results in $OUT"; grep -hE "Kaggle round 2|baseline rows|dev.C: primary|paired diff|decision|public rc|validation|READY|asr2 eval done|Traceback" "$OUT"/logs/asr2eval.log | tail -16
if [ "${KEEP_VM:-0}" != 1 ]; then
  [ -f "$WORK/runner.pid" ] && kill "$(cat "$WORK/runner.pid")" 2>/dev/null
  colab stop -s "$SESSION"; colab usage | tail -3
fi
