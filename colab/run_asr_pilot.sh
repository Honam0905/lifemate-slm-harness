#!/usr/bin/env bash
# Run the ASR LoRA pilot from any Linux / macOS machine that has the Colab CLI logged in (`colab usage` works).
#
#     bash colab/run_asr_pilot.sh /path/to/train+public_test/train+public_test
#
# Packs the code and the train data, starts a Colab A100 session (6.73 units/hour), keeps it alive with the job
# runner, uploads over `colab ssh`, queues colab/asr_pilot.sh and returns. Follow the job with
#     python3 colab/remote.py -s lifemate fetch pilot -n 40
# and when the log ends with "asr pilot done", download the results and STOP the VM with
#     bash colab/fetch_asr_pilot.sh
set -euo pipefail
DATA=${1:?usage: bash colab/run_asr_pilot.sh /path/to/folder/holding/train}
[ -f "$DATA/train/train.jsonl" ] || { echo "no train/train.jsonl under $DATA"; exit 1; }
cd "$(dirname "$0")/.."
SESSION=${SESSION:-lifemate}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
COLAB=$(command -v colab) || { echo "colab CLI not found (install: uv tool install google-colab-cli)"; exit 1; }
[ -f "$KEY" ] || ssh-keygen -t ed25519 -N "" -f "$KEY" -C lifemate
WORK=${WORK:-$PWD/.pilot_work}; mkdir -p "$WORK"
cat > "$WORK/ssh_config" <<EOF
Host colab
  ProxyCommand $COLAB ssh --proxy-mode -s $SESSION -i $KEY
  User root
  IdentityFile $KEY
  IdentitiesOnly yes
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
  ServerAliveInterval 30
EOF
echo "== $(date +%T) packing code and train data into $WORK"
tar czf "$WORK/code.tgz" --exclude='.venv*' --exclude='__pycache__' --exclude='.pilot_work' \
  --exclude='artifacts/vm/adapters/v1/ckpt-epoch1' \
  lifemate configs colab scripts tests requirements*.txt artifacts/splits artifacts/gold artifacts/vm/adapters
tar cf "$WORK/train.tar" -C "$DATA" train
ls -la "$WORK"
echo "== $(date +%T) starting Colab session '$SESSION' (A100) — billing starts now"
"$COLAB" new -s "$SESSION" --gpu A100
nohup python3 colab/remote.py -s "$SESSION" runner --hours 3 < /dev/null > "$WORK/runner.log" 2>&1 &
echo $! > "$WORK/runner.pid"
echo "== $(date +%T) uploading (~1 GB; took ~13 min from the office)"
rsync -a --partial -e "ssh -F $WORK/ssh_config" --rsync-path="mkdir -p /content/upload && rsync" \
  "$WORK/code.tgz" "$WORK/train.tar" colab:/content/upload/
ssh -F "$WORK/ssh_config" colab 'set -e; mkdir -p /content/zalo /content/data
  tar xzf /content/upload/code.tgz -C /content/zalo; tar xf /content/upload/train.tar -C /content/data
  echo "train clips on VM: $(find /content/data/train/audio -name "*.wav" | wc -l)"'
python3 colab/remote.py -s "$SESSION" submit pilot -- bash /content/zalo/colab/asr_pilot.sh
echo "== $(date +%T) pilot queued. Follow:  python3 colab/remote.py -s $SESSION fetch pilot -n 40"
echo "   When the log ends with 'asr pilot done':  bash colab/fetch_asr_pilot.sh   (downloads results, stops the VM)"
