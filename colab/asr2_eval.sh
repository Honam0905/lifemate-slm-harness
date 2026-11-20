#!/usr/bin/env bash
# On the Colab VM (queued by colab/run_asr2_eval.sh): the v1 agent's dev score (condition C) with the ASR ROUND-2
# transcripts from Kaggle (lifemate_asr_round2: LoRA attention+MLP, 3 epochs). No ASR runs here.
#   baseline  = this morning's rows with the round-1 transcripts (kg-ftA_s100: dev C 36.10, website 34.35), shipped
#               in the code archive, so it is not re-run
#   candidates = ftA2_s400, ftA2_s600 (dev WER 15.54 / 15.47 vs 20.53 for round 1)
# Decision: the candidate with the larger paired mean difference vs the baseline, if that difference is > 0.
# AUTO_PUBLIC=1 (default): then write the public submission with the matching Model B2 transcripts
# (A2-s400 -> B2-s627, A2-s600 -> B2-s941), ~46 min, and check it with scripts/check_submission.py.
set -uo pipefail
cd /content/zalo
export HF_HOME=/content/hf LIFEMATE_DATA_ROOT=/content/data LIFEMATE_ARTIFACTS=/content/artifacts PYTHONUNBUFFERED=1 UV_CACHE_DIR=/content/uv-cache
A=/content/artifacts; M=$A/models/v1-epoch2; L=/content/logs; K=/content/kaggle_out
TAGS=${TAGS:-"ftA2_s400 ftA2_s600"}
mkdir -p $A/splits $A/gold $A/asr $A/eval/kg-ftA_s100 $L
cp -n artifacts/splits/v1.json $A/splits/ && cp -n artifacts/gold/train_hashes.json $A/gold/ || exit 1
cp -rn $K/asr/* $A/asr/ || exit 1
cp -rn artifacts/vm/kaggle_eval/eval/kg-ftA_s100/dev.C $A/eval/kg-ftA_s100/ || exit 1
BASE=$A/eval/kg-ftA_s100/dev.C/rows.jsonl
echo "baseline rows: $(wc -l < $BASE) (round-1 ASR ftA-s100, dev C $(python3 -c "import json; print(json.load(open('$A/eval/kg-ftA_s100/dev.C/summary.json'))['primary'])"))"

echo "=== $(date +%T) train venv + MiniCPM5 (no ASR needed)"
[ -d /content/venv ] || uv venv -q /content/venv --python "$(which python3)" --system-site-packages || exit 1
uv pip install -q --python /content/venv/bin/python pyyaml "huggingface_hub[hf_xet]" "torchao>=0.16" || exit 1
source /content/venv/bin/activate
python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| gpu', torch.cuda.get_device_name(0))" || exit 1
hf download openbmb/MiniCPM5-2B --revision f97400052a43d642bbc6e9975e2397e3ae6a6b52 --quiet > /dev/null || exit 1

echo "=== $(date +%T) merge the v1 epoch-2 agent (the 34.35 agent)"
[ -f $M/config.json ] || python -m lifemate.train.merge --ckpt artifacts/vm/adapters/v1/ckpt-epoch2 --out $M || exit 1
python -c "import json; c = json.load(open('$M/config.json')); print('rope_theta', c.get('rope_theta')); assert c.get('rope_theta') == 5000000" || exit 1

echo "=== $(date +%T) transcript caches"
for t in $TAGS; do
  python - <<PY || exit 1
from lifemate.config import load_config; from lifemate.asr.base import backend_name; from pathlib import Path
n = backend_name(load_config("configs/kaggle_${t}_names.yaml")); p = Path("$A/asr") / n / "train.jsonl"
k = sum(1 for _ in open(p)) if p.exists() else 0
print(f"${t}: {n} -> {k} dev transcripts"); assert k == 887, "missing Kaggle round-2 dev transcripts"
PY
done
ls -d $A/asr/*ftB2* 2>/dev/null | sed 's|^|public cache: |'
python -c "import json; r = json.load(open('$K/results.json')); print('Kaggle round 2:', r.get('wer_dev'), '| public:', r.get('public_b'))" 2>/dev/null

run() {
  python -m lifemate.eval.run_eval --config configs/kaggle_$1_names.yaml --name kg-$1 --model $M --conditions C > $L/eval-$1.log 2>&1
  echo $? > $L/eval-$1.rc
}
set -- $TAGS
echo "=== $(date +%T) agent dev C with round-2 transcripts: $TAGS (two at a time on the A100)"
run "$1" & sleep 90; [ -n "${2:-}" ] && run "$2" & wait
for t in $TAGS; do
  [ "$(cat $L/eval-$t.rc)" = 0 ] || { echo "=== re-run $t alone"; tail -5 $L/eval-$t.log; run $t; }
  echo "=== $t rc $(cat $L/eval-$t.rc)"; grep "dev.C: primary" $L/eval-$t.log | tail -1
done

echo "=== $(date +%T) round 2 vs round 1 (paired, same 250 dev samples, same v1 agent)"
for t in $TAGS; do
  python -m lifemate.eval.report $A/eval/kg-$t/dev.C/rows.jsonl --vs $BASE > $L/report-$t.json
  python - <<PY
import json; t = open("$L/report-$t.json").read(); s = json.loads(t[: t.index('{\n "paired_vs_baseline"')])
p = json.loads(t[t.index('{\n "paired_vs_baseline"'):])["paired_vs_baseline"]
print(f"$t: primary {s['primary']} | known {s['slices']['apps']['known']['primary']} unseen {s['slices']['apps']['unseen']['primary']} | "
      f"paired diff vs round 1 {p['mean_diff']} CI {p['ci95']}")
PY
done
DECISION=$(python - <<PY
import json
best, out = None, "keep round 1"
for t in "$TAGS".split():
    txt = open(f"$L/report-{t}.json").read()
    p = json.loads(txt[txt.index('{\n "paired_vs_baseline"'):])["paired_vs_baseline"]
    if p["mean_diff"] > 0 and (best is None or p["mean_diff"] > best[1]):
        best = (t, p["mean_diff"])
if best:
    out = {"ftA2_s200": "ftB2-s314", "ftA2_s400": "ftB2-s627", "ftA2_s600": "ftB2-s941"}[best[0]]
print(out)
PY
)
echo "=== decision (larger paired gain vs round 1, if > 0): $DECISION"
echo "$DECISION" > $L/decision.txt

if [ "${AUTO_PUBLIC:-1}" = 1 ] && [ "$DECISION" != "keep round 1" ]; then
  T=${DECISION//-/_}
  CACHE=$(python -c "from lifemate.config import load_config; from lifemate.asr.base import backend_name; print('$A/asr/' + backend_name(load_config('configs/kaggle_${T}_names.yaml')) + '/public_test.jsonl')")
  if [ -f "$CACHE" ]; then
    echo "=== $(date +%T) public submission: v1 agent + $DECISION transcripts ($CACHE)"
    SUB=$A/submissions/v1-$DECISION/submission.jsonl
    python -m lifemate.infer.run --config configs/kaggle_${T}_names.yaml --data /content/data/public_test --model $M \
        --asr_cache $CACHE --out $SUB > $L/public.log 2>&1
    echo "public rc $?"; grep "^\[infer\|validation\|schema" $L/public.log | tail -4
    python -I scripts/check_submission.py $SUB /content/data 2>&1 | tail -3
  else
    echo "=== no public transcripts for $DECISION in the Kaggle output: run the matching B2 transcription first"
  fi
fi
echo "=== $(date +%T) asr2 eval done"
