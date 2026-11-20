#!/usr/bin/env bash
# On the Colab VM (queued by colab/run_kaggle_eval.sh): the agent's dev score (condition C) with the Kaggle ASR
# transcripts. No ASR runs here: the transcripts come from the Kaggle output (out/asr), so only the agent is needed.
#   kgbase    = Kaggle float32 base ASR (no fine-tune)        -> the baseline
#   ftA-s100  = Model A step 100 (train partition only)       -> candidate
#   ftA-s200  = Model A step 200                              -> candidate
# Decision: adopt the better candidate only if its paired 95% interval vs kgbase is above 0.
# AUTO_PUBLIC=1: if adopted, also write a public submission with the matching Model B transcripts
# (A-s100 -> B-s157, A-s200 -> B-s314); ~1 h more.
set -uo pipefail
cd /content/zalo
export HF_HOME=/content/hf LIFEMATE_DATA_ROOT=/content/data LIFEMATE_ARTIFACTS=/content/artifacts PYTHONUNBUFFERED=1 UV_CACHE_DIR=/content/uv-cache
A=/content/artifacts; M=$A/models/v1-epoch2; L=/content/logs; K=/content/kaggle_out
mkdir -p $A/splits $A/gold $A/asr $L
cp -n artifacts/splits/v1.json $A/splits/ && cp -n artifacts/gold/train_hashes.json $A/gold/ || exit 1
cp -rn $K/asr/* $A/asr/ || exit 1

echo "=== $(date +%T) train venv + MiniCPM5 (no ASR needed)"
[ -d /content/venv ] || uv venv -q /content/venv --python "$(which python3)" --system-site-packages || exit 1
uv pip install -q --python /content/venv/bin/python pyyaml "huggingface_hub[hf_xet]" "torchao>=0.16" || exit 1
source /content/venv/bin/activate
python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| gpu', torch.cuda.get_device_name(0))" || exit 1
hf download openbmb/MiniCPM5-2B --revision f97400052a43d642bbc6e9975e2397e3ae6a6b52 --quiet > /dev/null || exit 1

echo "=== $(date +%T) merge the v1 epoch-2 agent"
[ -f $M/config.json ] || python -m lifemate.train.merge --ckpt artifacts/vm/adapters/v1/ckpt-epoch2 --out $M || exit 1
python -c "import json; c = json.load(open('$M/config.json')); print('rope_theta', c.get('rope_theta')); assert c.get('rope_theta') == 5000000" || exit 1

echo "=== $(date +%T) transcript caches"
for t in kgbase ftA_s100 ftA_s200; do
  python - <<PY || exit 1
from lifemate.config import load_config; from lifemate.asr.base import backend_name; from pathlib import Path
n = backend_name(load_config("configs/kaggle_${t}_names.yaml")); p = Path("$A/asr") / n / "train.jsonl"
k = sum(1 for _ in open(p)) if p.exists() else 0
print(f"${t}: {n} -> {k} dev transcripts"); assert k == 887, "missing Kaggle dev transcripts"
PY
done
python -c "import json; print('Kaggle results:', json.dumps(json.load(open('$K/results.json')).get('wer_dev')))" 2>/dev/null

run() {
  python -m lifemate.eval.run_eval --config configs/kaggle_$1_names.yaml --name kg-$1 --model $M --conditions C > $L/eval-$1.log 2>&1
  echo $? > $L/eval-$1.rc
}
echo "=== $(date +%T) agent dev C: kgbase + ftA_s100 together, then ftA_s200"
run kgbase & sleep 90; run ftA_s100 & wait
run ftA_s200
for t in kgbase ftA_s100 ftA_s200; do
  [ "$(cat $L/eval-$t.rc)" = 0 ] || { echo "=== re-run $t alone"; tail -5 $L/eval-$t.log; run $t; }
  echo "=== $t rc $(cat $L/eval-$t.rc)"; grep "dev.C: primary" $L/eval-$t.log | tail -1
done

echo "=== $(date +%T) fine-tuned vs base ASR (paired, same 250 dev samples)"
for t in ftA_s100 ftA_s200; do
  echo "--- $t vs kgbase"
  python -m lifemate.eval.report $A/eval/kg-$t/dev.C/rows.jsonl --vs $A/eval/kg-kgbase/dev.C/rows.jsonl > $L/report-$t.json
  python - <<PY
import json; t = open("$L/report-$t.json").read(); s = json.loads(t[: t.index('{\n "paired_vs_baseline"')] if '"paired_vs_baseline"' in t else t)
p = json.loads(t[t.index('{\n "paired_vs_baseline"'):])["paired_vs_baseline"]
print(f"$t: primary {s['primary']} | known {s['slices']['apps']['known']['primary']} unseen {s['slices']['apps']['unseen']['primary']} | "
      f"paired diff {p['mean_diff']} CI {p['ci95']}")
PY
done
DECISION=$(python - <<PY
import json
best, out = None, "keep base ASR"
for t in ("ftA_s100", "ftA_s200"):
    txt = open(f"$L/report-{t}.json").read()
    p = json.loads(txt[txt.index('{\n "paired_vs_baseline"'):])["paired_vs_baseline"]
    if p["ci95"][0] > 0 and (best is None or p["mean_diff"] > best[1]):
        best = (t, p["mean_diff"])
if best:
    out = {"ftA_s100": "ftB-s157", "ftA_s200": "ftB-s314"}[best[0]]
print(out)
PY
)
echo "=== decision (paired 95% CI above 0): $DECISION"
echo "$DECISION" > $L/decision.txt

if [ "${AUTO_PUBLIC:-0}" = 1 ] && [ "$DECISION" != "keep base ASR" ]; then
  T=${DECISION//-/_}
  CACHE=$(python -c "from lifemate.config import load_config; from lifemate.asr.base import backend_name; print('$A/asr/' + backend_name(load_config('configs/kaggle_${T}_names.yaml')) + '/public_test.jsonl')")
  echo "=== $(date +%T) public submission with $DECISION transcripts ($CACHE)"
  python -m lifemate.infer.run --config configs/kaggle_${T}_names.yaml --data /content/data/public_test --model $M \
      --asr_cache $CACHE --out $A/submissions/kaggle-$DECISION/submission.jsonl > $L/public.log 2>&1
  echo "public rc $?"; grep "^\[infer\|validation\|schema" $L/public.log | tail -4
fi
echo "=== $(date +%T) kaggle eval done"
