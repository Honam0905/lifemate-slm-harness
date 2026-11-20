#!/usr/bin/env bash
# Download the ASR pilot results (small files only: adapters, eval rows, transcripts, logs) and STOP the Colab VM.
#     bash colab/fetch_asr_pilot.sh            (KEEP_VM=1 to skip stopping)
set -uo pipefail
cd "$(dirname "$0")/.."
SESSION=${SESSION:-lifemate}
WORK=${WORK:-$PWD/.pilot_work}
OUT=artifacts/vm/asr_pilot; mkdir -p "$OUT"
R="ssh -F $WORK/ssh_config"
rsync -a -e "$R" colab:/content/logs/ "$OUT/logs/"
rsync -a -e "$R" colab:/content/artifacts/asr_ft/run1/ "$OUT/asr_lora_run1/"          # LoRA adapters (re-merge locally)
rsync -a -e "$R" --include='pilot-*/***' --exclude='*' colab:/content/artifacts/eval/ "$OUT/eval/"
rsync -a -e "$R" colab:/content/artifacts/asr/ "$OUT/asr_cache/"
rsync -a -e "$R" colab:/content/artifacts/asr_ft/train.jsonl colab:/content/artifacts/asr_ft/holdout.jsonl "$OUT/"
echo "== results in $OUT"; grep -hE "dev WER|best ASR|dev.C: primary|mean_diff|ci95|asr pilot done|Traceback" "$OUT"/logs/pilot.log | tail -12
if [ "${KEEP_VM:-0}" != 1 ]; then
  [ -f "$WORK/runner.pid" ] && kill "$(cat "$WORK/runner.pid")" 2>/dev/null
  colab stop -s "$SESSION"; colab usage | tail -3
fi
