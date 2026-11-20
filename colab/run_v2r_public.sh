#!/usr/bin/env bash
# From any computer with the Colab CLI logged in: public submission with the v2r agent (molab 6 Nov) + ASR round-2
# B2-s941 transcripts, then v2r dev C with A2-s600 (DEV=0 to skip). ~1.3 h on the Colab A100 (6.73 units/h, ~9 units).
#
#     bash colab/run_v2r_public.sh /path/to/train+public_test/train+public_test /path/to/kaggle/round2/out
# Follow:  python3 colab/remote.py -s lifemate fetch v2rpub -n 30
# When the log shows "PUBLIC DONE":        KEEP_VM=1 bash colab/fetch_v2r_public.sh   (submission; VM keeps running)
# When the log ends with "v2r public done": bash colab/fetch_v2r_public.sh            (everything, stops the VM)
set -euo pipefail
DATA=${1:?usage: bash colab/run_v2r_public.sh DATA_DIR KAGGLE_ROUND2_OUT_DIR}
KOUT=${2:?usage: bash colab/run_v2r_public.sh DATA_DIR KAGGLE_ROUND2_OUT_DIR}
[ -f "$DATA/public_test/public_test.jsonl" ] && [ -d "$DATA/train/train_environment" ] || { echo "no train/ + public_test/ under $DATA"; exit 1; }
cd "$(dirname "$0")/.."
[ -f artifacts/vm/adapters/v2r/ckpt-epoch2/adapter_model.safetensors ] || { echo "missing artifacts/vm/adapters/v2r/ckpt-epoch2"; exit 1; }
[ -f artifacts/vm/asr2_eval/eval/kg-ftA2_s600/dev.C/rows.jsonl ] || { echo "missing the v1 baseline rows artifacts/vm/asr2_eval/eval/kg-ftA2_s600/dev.C"; exit 1; }
ls -d "$KOUT"/asr/*ftA2-s600* "$KOUT"/asr/*ftB2-s941* > /dev/null || { echo "no ftA2-s600 / ftB2-s941 transcripts in $KOUT/asr"; exit 1; }
SESSION=${SESSION:-lifemate}
WORK=${WORK:-$PWD/.v2r_work}; mkdir -p "$WORK"
KEY=${KEY:-$WORK/id_ed25519}
COLAB=$(command -v colab) || { echo "colab CLI not found (install: uv tool install google-colab-cli)"; exit 1; }
[ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -f "$KEY" -C lifemate
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
tar czf "$WORK/code.tgz" --exclude='__pycache__' lifemate configs colab scripts requirements*.txt \
  artifacts/splits artifacts/gold artifacts/vm/adapters/v2r/ckpt-epoch2 artifacts/vm/asr2_eval/eval/kg-ftA2_s600
tar czf "$WORK/data.tgz" -C "$DATA" --exclude='audio' --exclude='__MACOSX' --exclude='.DS_Store' train public_test
tar czf "$WORK/kaggle_out.tgz" -C "$KOUT" $(cd "$KOUT" && ls -d asr/*ftA2-s600* asr/*ftB2-s941*)
ls -la "$WORK"
echo "== $(date +%T) starting Colab session '$SESSION' (A100) — billing starts now"
"$COLAB" new -s "$SESSION" --gpu A100
nohup python3 colab/remote.py -s "$SESSION" runner --hours 2.5 < /dev/null > "$WORK/runner.log" 2>&1 &
echo $! > "$WORK/runner.pid"
echo "== $(date +%T) uploading"
rsync -a --partial -e "ssh -F $WORK/ssh_config" --rsync-path="mkdir -p /content/upload && rsync" \
  "$WORK/code.tgz" "$WORK/data.tgz" "$WORK/kaggle_out.tgz" colab:/content/upload/
ssh -F "$WORK/ssh_config" colab 'set -e; mkdir -p /content/zalo /content/data /content/kaggle_out
  tar xzf /content/upload/code.tgz -C /content/zalo; tar xzf /content/upload/data.tgz -C /content/data
  tar xzf /content/upload/kaggle_out.tgz -C /content/kaggle_out
  echo "public samples on VM: $(wc -l < /content/data/public_test/public_test.jsonl), transcripts: $(ls /content/kaggle_out/asr)"'
python3 colab/remote.py -s "$SESSION" submit v2rpub -- env DEV="${DEV:-1}" bash /content/zalo/colab/v2r_public.sh
echo "== $(date +%T) queued. Follow:  python3 colab/remote.py -s $SESSION fetch v2rpub -n 30"
