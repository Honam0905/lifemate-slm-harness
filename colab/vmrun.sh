#!/usr/bin/env bash
# Run a command detached on the VM inside the project env; log at /content/logs/NAME.log, exit code at NAME.rc.
#   SSH_CONFIG=... colab/vmrun.sh NAME [train|asr] -- python scripts/x.py ...
set -euo pipefail
name=$1; venv=$2; shift 2; [ "${1:-}" = "--" ] && shift
cmd=$(printf '%q ' "$@")
ssh -F "${SSH_CONFIG:?set SSH_CONFIG}" colab "mkdir -p /content/logs; rm -f /content/logs/$name.rc; cd /content/zalo || exit 1; \
  nohup setsid bash -c 'source /content/env.sh 2>/dev/null; export HF_HOME=/content/hf LIFEMATE_DATA_ROOT=/content/data LIFEMATE_ARTIFACTS=/content/artifacts PYTHONUNBUFFERED=1; \
  source /content/$( [ "$venv" = asr ] && echo venv-asr || echo venv )/bin/activate; $cmd; echo \$? > /content/logs/$name.rc' \
  > /content/logs/$name.log 2>&1 < /dev/null & echo started $name"
