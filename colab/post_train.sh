#!/usr/bin/env bash
# After v1 training: merge both epoch checkpoints, evaluate on dev (C then A), pick the best by dev condition C
# (primary), print the parameter inventory, and produce the public submission with online ASR (deployment path).
set -uo pipefail
cd /content/zalo
export HF_HOME=/content/hf LIFEMATE_DATA_ROOT=/content/data LIFEMATE_ARTIFACTS=/content/artifacts PYTHONUNBUFFERED=1
A=/content/artifacts; RUN=${RUN:-v1}
until [ -f /content/logs/${RUN}_train.rc ]; do sleep 60; done
[ "$(cat /content/logs/${RUN}_train.rc)" = 0 ] || { echo "=== training failed; nothing to do"; exit 1; }
source /content/venv/bin/activate
for e in 1 2; do
  echo "=== merge epoch $e"
  [ -f $A/models/$RUN-epoch$e/config.json ] || python -m lifemate.train.merge --ckpt $A/runs/$RUN/ckpt-epoch$e --out $A/models/$RUN-epoch$e || exit 1
done
for e in 1 2; do
  echo "=== dev eval epoch $e"
  python -m lifemate.eval.run_eval --name $RUN-epoch$e --model $A/models/$RUN-epoch$e --conditions C A || exit 1
done
best=$(python - <<PY
import json
s = {e: json.load(open(f"$A/eval/$RUN-epoch{e}/dev.C/summary.json"))["primary"] for e in (1, 2)}
print(f"$RUN-epoch{max(s, key=lambda e: (s[e], e))}")
PY
)
echo "=== best by dev C primary: $best"
for e in 1 2; do echo "--- $RUN-epoch$e vs untuned base (paired, 50 shared dev ids)"; python -m lifemate.eval.report $A/eval/$RUN-epoch$e/dev.C/rows.jsonl --vs $A/eval/base-minicpm5/dev.C/rows.jsonl | tail -9; done
python scripts/param_inventory.py --llm $A/models/$best
deactivate; source /content/venv-asr/bin/activate
echo "=== public submission with $best (online ASR, one sample at a time)"
python -m lifemate.infer.run --data /content/data/public_test --model $A/models/$best --out $A/submissions/$best/submission.jsonl
echo "=== post-train done"
