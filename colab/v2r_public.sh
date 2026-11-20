#!/usr/bin/env bash
# On the Colab VM (queued by colab/run_v2r_public.sh): the PUBLIC submission with the v2r agent (molab, 6 Nov,
# epoch 2: v2 data + 50% renaming) and the ASR round-2 Model B2-s941 transcripts (the 38.63 transcripts), then
# (DEV=1, default) v2r on dev C with the round-2 Model A2-s600 transcripts, paired against v1 with the same transcripts
# (dev C 38.21, website 38.63). No ASR runs here; inference config = configs/v2_* (date table on, as v2r was trained).
set -uo pipefail
cd /content/zalo
export HF_HOME=/content/hf LIFEMATE_DATA_ROOT=/content/data LIFEMATE_ARTIFACTS=/content/artifacts PYTHONUNBUFFERED=1 UV_CACHE_DIR=/content/uv-cache
A=/content/artifacts; M=$A/models/v2r-epoch2; L=/content/logs; K=/content/kaggle_out
mkdir -p $A/splits $A/gold $A/asr $A/eval/kg-ftA2_s600 $L
cp -n artifacts/splits/v1.json $A/splits/ && cp -n artifacts/gold/train_hashes.json $A/gold/ || exit 1
cp -rn $K/asr/* $A/asr/ || exit 1
cp -rn artifacts/vm/asr2_eval/eval/kg-ftA2_s600/dev.C $A/eval/kg-ftA2_s600/ || exit 1
BASE=$A/eval/kg-ftA2_s600/dev.C/rows.jsonl
echo "baseline rows: $(wc -l < $BASE) (v1 + round-2 A2-s600, dev C $(python3 -c "import json; print(json.load(open('$A/eval/kg-ftA2_s600/dev.C/summary.json'))['primary'])"))"

echo "=== $(date +%T) train venv + MiniCPM5 (no ASR needed)"
[ -d /content/venv ] || uv venv -q /content/venv --python "$(which python3)" --system-site-packages || exit 1
uv pip install -q --python /content/venv/bin/python pyyaml "huggingface_hub[hf_xet]" "torchao>=0.16" || exit 1
source /content/venv/bin/activate
python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| gpu', torch.cuda.get_device_name(0))" || exit 1
hf download openbmb/MiniCPM5-2B --revision f97400052a43d642bbc6e9975e2397e3ae6a6b52 --quiet > /dev/null || exit 1

echo "=== $(date +%T) merge the v2r epoch-2 agent"
[ -f $M/config.json ] || python -m lifemate.train.merge --config configs/v2r.yaml --ckpt artifacts/vm/adapters/v2r/ckpt-epoch2 --out $M || exit 1
python -c "import json; c = json.load(open('$M/config.json')); print('rope_theta', c.get('rope_theta')); assert c.get('rope_theta') == 5000000" || exit 1
python -c "import yaml, sys; sys.path.insert(0, '.'); from lifemate.config import load_config; c = load_config('configs/v2_ftB2_s941.yaml'); print('prompt', c.get('prompt'), '| harness', c.get('harness')); assert c['prompt'].get('date_table')" || exit 1

cache() {  # $1 = config, $2 = split -> transcript cache path (asserts it exists with the expected line count)
  python - <<PY
import sys; sys.path.insert(0, "."); from pathlib import Path
from lifemate.config import load_config; from lifemate.asr.base import backend_name
p = Path("$A/asr") / backend_name(load_config("configs/$1.yaml")) / "$2.jsonl"
n = sum(1 for _ in open(p)) if p.exists() else 0
assert n == {"train": 887, "public_test": 1933}["$2"], f"{p}: {n} lines"
print(p)
PY
}
PUB=$(cache v2_ftB2_s941 public_test) || { echo "missing the B2-s941 public transcripts"; exit 1; }
echo "public cache: $PUB"

echo "=== $(date +%T) public submission: v2r agent + ftB2-s941 transcripts"
SUB=$A/submissions/v2r-ftB2-s941/submission.jsonl
python -m lifemate.infer.run --config configs/v2_ftB2_s941.yaml --data /content/data/public_test --model $M \
    --asr_cache $PUB --out $SUB > $L/public.log 2>&1
echo "public rc $?"; grep "^\[infer\|validation\|schema" $L/public.log | tail -4
python -I scripts/check_submission.py $SUB /content/data 2>&1 | tail -3
sha256sum $SUB
echo "=== $(date +%T) PUBLIC DONE"

if [ "${DEV:-1}" = 1 ]; then
  DEVC=$(cache v2_ftA2_s600 train) || { echo "missing the A2-s600 dev transcripts"; exit 1; }
  echo "=== $(date +%T) dev C: v2r + ftA2-s600 transcripts ($DEVC)"
  python -m lifemate.eval.run_eval --config configs/v2_ftA2_s600.yaml --name v2r-ftA2_s600 --model $M --conditions C \
      > $L/eval-v2r.log 2>&1
  echo "eval rc $?"; grep "dev.C: primary" $L/eval-v2r.log | tail -1
  python -m lifemate.eval.report $A/eval/v2r-ftA2_s600/dev.C/rows.jsonl --vs $BASE > $L/report-v2r.json
  python - <<PY
import json; t = open("$L/report-v2r.json").read(); s = json.loads(t[: t.index('{\n "paired_vs_baseline"')])
p = json.loads(t[t.index('{\n "paired_vs_baseline"'):])["paired_vs_baseline"]
print(f"v2r + ftA2-s600: dev C {s['primary']} | known {s['slices']['apps']['known']['primary']} unseen {s['slices']['apps']['unseen']['primary']} "
      f"| paired vs v1 + ftA2-s600 (38.21; known 51.15, unseen 20.92): {p['mean_diff']:+} CI {p['ci95']}")
PY
fi
echo "=== $(date +%T) v2r public done"
