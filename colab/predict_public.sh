#!/usr/bin/env bash
# Public-test submission from the trained v1 adapter on a fresh Colab VM: setup (once), merge epoch 2, parameter
# inventory, then the deployment path (online ASR, one sample at a time) and the split's own zaic_tools validator.
# Expects the code at /content/zalo (adapter under artifacts/vm/adapters/) and the data at /content/data/public_test.
set -uo pipefail
cd /content/zalo
export HF_HOME=/content/hf LIFEMATE_DATA_ROOT=/content/data LIFEMATE_ARTIFACTS=/content/artifacts PYTHONUNBUFFERED=1 UV_CACHE_DIR=/content/uv-cache
A=/content/artifacts; CKPT=${CKPT:-/content/zalo/artifacts/vm/adapters/v1/ckpt-epoch2}; NAME=${NAME:-v1-epoch2}
[ -d /content/venv-asr ] || bash colab/setup_vm.sh || exit 1
source /content/venv/bin/activate
echo "=== merge env must match the versions the adapter was trained with (Colab's own torch, not a venv copy)"
python - <<'PY' || exit 1
import peft, torch, transformers
v = (torch.__version__, transformers.__version__, peft.__version__)
print("torch %s | transformers %s | peft %s" % v)
assert torch.__file__.startswith("/usr/"), f"venv has its own torch at {torch.__file__}"
assert v[1:] == ("5.22.0", "0.21.2"), f"expected transformers 5.22.0 + peft 0.21.2, got {v[1:]}"
PY
echo "=== merge $CKPT"
[ -f $A/models/$NAME/config.json ] || python -m lifemate.train.merge --ckpt $CKPT --out $A/models/$NAME || exit 1
python scripts/param_inventory.py --llm $A/models/$NAME
deactivate; source /content/venv-asr/bin/activate
echo "=== public submission with $NAME (online ASR, one sample at a time)"
python -m lifemate.infer.run --data /content/data/public_test --model $A/models/$NAME --out $A/submissions/$NAME/submission.jsonl
rc=$?
echo "=== predict done (exit $rc)"
exit $rc
