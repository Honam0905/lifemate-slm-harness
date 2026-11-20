#!/usr/bin/env bash
# Push the code (not the data, not artifacts) to the VM at /content/zalo.  Usage: SSH_CONFIG=... colab/sync.sh
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -a --delete --exclude 'train+public_test/' --exclude 'artifacts/' --exclude '__pycache__/' --exclude '.git/' \
  -e "ssh -F ${SSH_CONFIG:?set SSH_CONFIG}" ./ colab:/content/zalo/
mkdir -p artifacts/splits
for d in splits gold; do [ -d artifacts/$d ] && rsync -a -e "ssh -F $SSH_CONFIG" artifacts/$d/ colab:/content/artifacts/$d/; done
echo "[sync] code -> colab:/content/zalo"
