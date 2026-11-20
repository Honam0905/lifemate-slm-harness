#!/usr/bin/env bash
# Download the Kaggle-ASR dev evaluation (eval rows, reports, logs, any public submission) and STOP the Colab VM.
#     bash colab/fetch_kaggle_eval.sh            (KEEP_VM=1 to skip stopping)
set -uo pipefail
cd "$(dirname "$0")/.."
SESSION=${SESSION:-lifemate}
WORK=${WORK:-$PWD/.kgeval_work}
OUT=artifacts/vm/kaggle_eval; mkdir -p "$OUT"
R="ssh -F $WORK/ssh_config"
rsync -a -e "$R" colab:/content/logs/ "$OUT/logs/"
rsync -a -e "$R" --include='kg-*/***' --exclude='*' colab:/content/artifacts/eval/ "$OUT/eval/"
rsync -a -e "$R" colab:/content/artifacts/submissions/ "$OUT/submissions/" 2>/dev/null || true
echo "== results in $OUT"; grep -hE "Kaggle results|dev.C: primary|paired diff|decision|public rc|validation|kaggle eval done|Traceback" "$OUT"/logs/kgeval.log | tail -14
if [ "${KEEP_VM:-0}" != 1 ]; then
  [ -f "$WORK/runner.pid" ] && kill "$(cat "$WORK/runner.pid")" 2>/dev/null
  colab stop -s "$SESSION"; colab usage | tail -3
fi
