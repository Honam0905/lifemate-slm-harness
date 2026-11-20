#!/usr/bin/env bash
# Build the molab bundle (code + data without audio + training trajectories + round-2 transcripts + baselines + the
# v2r adapter for the same-GPU baseline).
#     bash cloud/make_molab_bundle.sh DATA_DIR KAGGLE_ROUND2_OUT_DIR OUT_ZIP
# DATA_DIR = train+public_test/train+public_test (audio is skipped); KAGGLE_ROUND2_OUT_DIR = the Kaggle round-2 out/.
set -euo pipefail
DATA=${1:?usage: DATA_DIR KAGGLE_ROUND2_OUT_DIR OUT_ZIP}; KOUT=${2:?}; OUT=$(realpath -m "${3:?}")
cd "$(dirname "$0")/.."
need=(artifacts/splits/v1.json artifacts/gold/train_hashes.json artifacts/trajectories/v1/train.allsynth.jsonl
      artifacts/vm/adapters/v2r/ckpt-epoch2/adapter_model.safetensors
      artifacts/vm/asr2_eval/eval/kg-ftA2_s600/dev.C/rows.jsonl artifacts/vm/v2r_public/eval/v2r-ftA2_s600/dev.C/rows.jsonl
      synth/rename/out/rename_part1.json "$DATA/train/train.jsonl" "$DATA/public_test/public_test.jsonl"
      "$KOUT"/asr/qwen3_asr-Qwen3-ASR-1.7B-7278e1e-ftA2-s600-ctxnames/train.jsonl
      "$KOUT"/asr/qwen3_asr-Qwen3-ASR-1.7B-7278e1e-ftB2-s941-ctxnames/public_test.jsonl)
for f in "${need[@]}"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/molab_bundle.XXXX"); trap 'rm -rf "$STAGE"' EXIT
rsync -a --exclude='__pycache__' --exclude='_molab_*' lifemate scripts cloud configs requirements-train.txt "$STAGE/"
mkdir -p "$STAGE/synth/rename" && rsync -a synth/rename/out "$STAGE/synth/rename/"
mkdir -p "$STAGE/data" && rsync -a --exclude='audio' --exclude='__MACOSX' --exclude='.DS_Store' "$DATA/train" "$DATA/public_test" "$STAGE/data/"
rsync -aR artifacts/splits artifacts/gold artifacts/trajectories/v1 artifacts/vm/adapters/v2r/ckpt-epoch2 \
  artifacts/vm/asr2_eval/eval/kg-ftA2_s600/dev.C artifacts/vm/v2r_public/eval/v2r-ftA2_s600/dev.C "$STAGE/"
mkdir -p "$STAGE/artifacts/asr" && rsync -a "$KOUT"/asr/*ftA2-s600-ctxnames "$KOUT"/asr/*ftB2-s941-ctxnames "$STAGE/artifacts/asr/"
find "$STAGE" -name 'state.json' -path '*adapters*' -delete
rm -f "$OUT"; (cd "$STAGE" && zip -qr -X "$OUT" .)
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1)); $(unzip -l "$OUT" | tail -1 | awk '{print $2}') files; sha256 $(sha256sum "$OUT" | cut -c1-16)"
# molab uploads are capped at 100 MB: split into equal parts named <zip>.part<i>of<n>; cell 1 of cloud/molab_cells.md
# joins them (in order) and checks the sha256 written into the part names' companion file
SIZE=$(stat -c %s "$OUT"); LIM=$((95 * 1024 * 1024))
if [ "$SIZE" -gt "$LIM" ]; then
  N=$(( (SIZE + LIM - 1) / LIM ))
  rm -f "$OUT".part*
  split -n "$N" -d -a 1 "$OUT" "$OUT.tmp"
  for i in $(seq 0 $((N - 1))); do mv "$OUT.tmp$i" "$OUT.part$((i + 1))of$N"; done
  sha256sum "$OUT" | cut -d' ' -f1 > "$OUT.sha256"
  ls -l "$OUT".part* | awk '{print $5, $9}'
  echo "upload these $N parts to molab (each < 100 MB); full-zip sha256 $(cat "$OUT.sha256")"
fi
unzip -l "$OUT" | awk '{print $4}' | grep -E '^[^/]+/[^/]+/?$' | cut -d/ -f1-2 | sort -u | tr '\n' ' '; echo
