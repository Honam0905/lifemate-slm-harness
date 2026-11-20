#!/usr/bin/env bash
# One-time VM setup: two venvs (train / asr) and model downloads, all under /content.
set -euo pipefail
export HF_HOME=/content/hf UV_CACHE_DIR=/content/uv-cache
mkdir -p /content/logs /content/hf /content/artifacts

echo "=== train venv (reuses Colab's CUDA torch via system site-packages)"
[ -d /content/venv ] || uv venv -q /content/venv --python "$(which python3)" --system-site-packages
uv pip install -q --python /content/venv/bin/python pyyaml "huggingface_hub[hf_xet]" "torchao>=0.16"   # Colab ships torchao 0.10, which peft 0.21 rejects
/content/venv/bin/python -c "import torch, transformers, peft, accelerate; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| accelerate', accelerate.__version__)"

echo "=== asr venv (qwen-asr pins transformers 4.57.6, so it gets its own env)"
[ -d /content/venv-asr ] || uv venv -q /content/venv-asr --python 3.12
uv pip install -q --python /content/venv-asr/bin/python "qwen-asr==0.0.6" pyyaml
/content/venv-asr/bin/python -c "import torch, transformers, qwen_asr; print('torch', torch.__version__, torch.version.cuda, '| transformers', transformers.__version__)"

echo "=== models"
for m in openbmb/MiniCPM5-2B Qwen/Qwen3-ASR-1.7B; do
  /content/venv/bin/hf download "$m" --quiet > /dev/null && echo "downloaded $m"
done
du -sh /content/hf
echo "=== setup done"
