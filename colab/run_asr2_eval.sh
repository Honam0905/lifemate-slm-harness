#!/usr/bin/env bash
# From any computer with the Colab CLI logged in: the v1 agent's dev score with the ASR ROUND-2 transcripts
# (Kaggle notebook kaggle/lifemate_asr_round2.ipynb), then the public submission if round 2 beats round 1.
#
#     bash colab/run_asr2_eval.sh /path/to/train+public_test/train+public_test /path/to/kaggle/round2/out
#     AUTO_PUBLIC=0 bash colab/run_asr2_eval.sh ...      # dev eval only, no public submission
#
# Uploads only the code, the v1 epoch-2 agent adapter, this morning's round-1 dev rows (the paired baseline), the
# Kaggle round-2 transcripts and the train/public json + environments (no audio, ~35 MB), starts the Colab A100
# (6.73 units/h; ~30 min dev + ~46 min public ≈ 9 units), queues colab/asr2_eval.sh and returns.
# Follow:  python3 colab/remote.py -s lifemate fetch asr2eval -n 30
# When the log ends with "asr2 eval done":  bash colab/fetch_asr2_eval.sh   (downloads results, stops the VM)
set -euo pipefail
DATA=${1:?usage: bash colab/run_asr2_eval.sh DATA_DIR KAGGLE_ROUND2_OUT_DIR}
KOUT=${2:?usage: bash colab/run_asr2_eval.sh DATA_DIR KAGGLE_ROUND2_OUT_DIR}
[ -f "$DATA/train/train.jsonl" ] && [ -d "$DATA/train/train_environment" ] || { echo "no train/train.jsonl + train_environment/ under $DATA"; exit 1; }
[ -f "$KOUT/results.json" ] && [ -d "$KOUT/asr" ] || { echo "no results.json + asr/ under $KOUT (the Kaggle out/ folder)"; exit 1; }
cd "$(dirname "$0")/.."
[ -f artifacts/vm/adapters/v1/ckpt-epoch2/adapter_model.safetensors ] || { echo "missing artifacts/vm/adapters/v1/ckpt-epoch2"; exit 1; }
for t in ftA2_s400 ftA2_s600 ftB2_s627 ftB2_s941; do [ -f configs/kaggle_${t}_names.yaml ] || { echo "missing configs/kaggle_${t}_names.yaml"; exit 1; }; done
[ -f artifacts/vm/kaggle_eval/eval/kg-ftA_s100/dev.C/rows.jsonl ] || { echo "missing the round-1 baseline rows artifacts/vm/kaggle_eval/eval/kg-ftA_s100/dev.C"; exit 1; }
ls -d "$KOUT"/asr/*ftA2-s400* "$KOUT"/asr/*ftA2-s600* > /dev/null || { echo "no ftA2-s400/s600 dev transcripts in $KOUT/asr (round-2 run not finished?)"; exit 1; }
echo "Kaggle round-2 dev WER: $(python3 -c "import json; r = json.load(open('$KOUT/results.json')); print(r.get('wer_dev'), '| public:', r.get('public_b'))")"
SESSION=${SESSION:-lifemate}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
COLAB=$(command -v colab) || { echo "colab CLI not found (install: uv tool install google-colab-cli)"; exit 1; }
[ -f "$KEY" ] || ssh-keygen -t ed25519 -N "" -f "$KEY" -C lifemate
WORK=${WORK:-$PWD/.asr2_work}; mkdir -p "$WORK"
cat > "$WORK/ssh_config" <<EOF2
Host colab
  ProxyCommand $COLAB ssh --proxy-mode -s $SESSION -i $KEY
  User root
  IdentityFile $KEY
  IdentitiesOnly yes
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
  ServerAliveInterval 30
EOF2
echo "== $(date +%T) packing (no audio)"
tar czf "$WORK/code.tgz" --exclude='__pycache__' lifemate configs colab scripts tests requirements*.txt \
  artifacts/splits artifacts/gold artifacts/vm/adapters/v1/ckpt-epoch2 artifacts/vm/kaggle_eval/eval/kg-ftA_s100
tar czf "$WORK/data.tgz" -C "$DATA" --exclude='audio' --exclude='__MACOSX' --exclude='.DS_Store' train public_test
tar czf "$WORK/kaggle_out.tgz" -C "$KOUT" asr results.json
ls -la "$WORK"
echo "== $(date +%T) starting Colab session '$SESSION' (A100) — billing starts now"
"$COLAB" new -s "$SESSION" --gpu A100
nohup python3 colab/remote.py -s "$SESSION" runner --hours 3 < /dev/null > "$WORK/runner.log" 2>&1 &
echo $! > "$WORK/runner.pid"
echo "== $(date +%T) uploading"
rsync -a --partial -e "ssh -F $WORK/ssh_config" --rsync-path="mkdir -p /content/upload && rsync" \
  "$WORK/code.tgz" "$WORK/data.tgz" "$WORK/kaggle_out.tgz" colab:/content/upload/
ssh -F "$WORK/ssh_config" colab 'set -e; mkdir -p /content/zalo /content/data /content/kaggle_out
  tar xzf /content/upload/code.tgz -C /content/zalo; tar xzf /content/upload/data.tgz -C /content/data
  tar xzf /content/upload/kaggle_out.tgz -C /content/kaggle_out
  echo "train samples on VM: $(wc -l < /content/data/train/train.jsonl), environments: $(ls /content/data/train/train_environment | wc -l)"'
python3 colab/remote.py -s "$SESSION" submit asr2eval -- env AUTO_PUBLIC="${AUTO_PUBLIC:-1}" bash /content/zalo/colab/asr2_eval.sh
echo "== $(date +%T) queued. Follow:  python3 colab/remote.py -s $SESSION fetch asr2eval -n 30"
echo "   When the log ends with 'asr2 eval done':  bash colab/fetch_asr2_eval.sh   (downloads results, stops the VM)"
