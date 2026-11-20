#!/usr/bin/env bash
# Build the small code dataset for Kaggle: dist/lifemate-code-kaggle.zip (+ a copy of the notebook to import).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 kaggle/make_notebook.py
OUT=${1:-../../dist}
mkdir -p "$OUT"
rm -f "$OUT/lifemate-code-kaggle.zip"
zip -qr "$OUT/lifemate-code-kaggle.zip" lifemate scripts configs kaggle tests colab artifacts/splits artifacts/asr_ft artifacts/gold \
    README.md requirements-infer.txt requirements-train.txt \
    -x '*/__pycache__/*' '*.pyc'
cp kaggle/lifemate_asr_overnight.ipynb "$OUT/"
ls -la "$OUT/lifemate-code-kaggle.zip" "$OUT/lifemate_asr_overnight.ipynb"
