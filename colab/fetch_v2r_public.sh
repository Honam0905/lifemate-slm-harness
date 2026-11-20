#!/usr/bin/env bash
# Download the v2r public submission (+ dev eval rows, logs) and STOP the Colab VM.
#     bash colab/fetch_v2r_public.sh            (KEEP_VM=1 to skip stopping, e.g. while the dev eval still runs)
set -uo pipefail
cd "$(dirname "$0")/.."
SESSION=${SESSION:-lifemate}
WORK=${WORK:-$PWD/.v2r_work}
OUT=artifacts/vm/v2r_public; mkdir -p "$OUT"
R="ssh -F $WORK/ssh_config"
rsync -a -e "$R" colab:/content/logs/ "$OUT/logs/"
rsync -a -e "$R" colab:/content/artifacts/submissions/ "$OUT/submissions/" 2>/dev/null || true
rsync -a -e "$R" --include='v2r-*/***' --exclude='*' colab:/content/artifacts/eval/ "$OUT/eval/" 2>/dev/null || true
echo "== results in $OUT"
grep -hE "baseline rows|rope_theta|public rc|\[infer.*(done|wrote)|validation|schema|READY|submission.jsonl$|PUBLIC DONE|dev.C: primary|paired vs|v2r public done|Traceback|Error" "$OUT"/logs/v2rpub.log | tail -16
if [ "${KEEP_VM:-0}" != 1 ]; then
  [ -f "$WORK/runner.pid" ] && kill "$(cat "$WORK/runner.pid")" 2>/dev/null
  colab stop -s "$SESSION"; colab usage | tail -3
fi
