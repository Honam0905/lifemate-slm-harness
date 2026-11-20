#!/usr/bin/env bash
# Harness ablations on the best v1 model, dev condition C, one sample at a time (= the inference path).
# Each config adds one change to the previous one; each is compared to h_cache with a paired bootstrap.
set -uo pipefail
cd /content/zalo
export HF_HOME=/content/hf LIFEMATE_DATA_ROOT=/content/data LIFEMATE_ARTIFACTS=/content/artifacts PYTHONUNBUFFERED=1
A=/content/artifacts
source /content/venv/bin/activate
best=${BEST:-$(python - <<PY
import json
s = {e: json.load(open(f"$A/eval/v1-epoch{e}/dev.C/summary.json"))["primary"] for e in (1, 2)}
print(f"v1-epoch{max(s, key=lambda e: (s[e], e))}")
PY
)}
echo "=== model $best"
for c in h_cache h_fix; do
  echo "=== $c"; python -m lifemate.eval.run_eval --config configs/$c.yaml --name abl-$c-$best --model $A/models/$best --conditions C --sequential || exit 1
done
echo "=== ASR transcripts with context (train split; dev clips are part of it)"
deactivate; source /content/venv-asr/bin/activate
python scripts/build_asr_cache.py --config configs/h_asrctx.yaml --split train || exit 1
deactivate; source /content/venv/bin/activate
echo "=== h_asrctx"; python -m lifemate.eval.run_eval --config configs/h_asrctx.yaml --name abl-h_asrctx-$best --model $A/models/$best --conditions C --sequential || exit 1
echo "=== sanity: sequential+cache vs the batched post-train run (should be close)"
python -m lifemate.eval.report $A/eval/abl-h_cache-$best/dev.C/rows.jsonl --vs $A/eval/$best/dev.C/rows.jsonl | tail -5
for c in h_fix h_asrctx; do echo "=== $c vs h_cache"; python -m lifemate.eval.report $A/eval/abl-$c-$best/dev.C/rows.jsonl --vs $A/eval/abl-h_cache-$best/dev.C/rows.jsonl | tail -5; done
for c in h_cache h_fix h_asrctx; do python - <<PY
import json; s = json.load(open("$A/eval/abl-$c-$best/dev.C/summary.json"))
print("$c", "primary", s["primary"], s["primary_ci95"], "| s/sample", round(s["seconds"] / s["n"], 2),
      "| cached/computed tokens", s["cached_tokens"], s["computed_tokens"], "| constrained steps", s["constrained_steps"])
PY
done
echo "=== ablation done"
