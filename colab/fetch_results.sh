#!/usr/bin/env bash
# Copy our own small outputs back from the VM (eval rows/summaries, submissions, run logs) into artifacts/vm/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/vm/logs
rsync -a -e "ssh -F $SSH_CONFIG" colab:/content/artifacts/eval/ artifacts/vm/eval/ || true
rsync -a -e "ssh -F $SSH_CONFIG" colab:/content/artifacts/submissions/ artifacts/vm/submissions/ || true
rsync -a -e "ssh -F $SSH_CONFIG" --include '*.log' --include '*.rc' --exclude '*' colab:/content/logs/ artifacts/vm/logs/ || true
du -sh artifacts/vm
