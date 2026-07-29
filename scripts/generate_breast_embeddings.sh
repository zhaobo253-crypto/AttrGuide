#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${REPO_ROOT}/experiments/breast/vitbase_attrguide"
cd "${EXP_DIR}"

ATTR_CSV="${ATTR_CSV:-${REPO_ROOT}/examples/attributes_breast.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/data/attribute_embeddings_breast.pt}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${REPO_ROOT}/pretrained/clip/ViT-B-32.pt}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

if [ ! -f "${ATTR_CSV}" ]; then
  echo "ATTR_CSV not found: ${ATTR_CSV}" >&2
  echo "Set ATTR_CSV=/path/to/attributes.csv and rerun this script." >&2
  exit 1
fi

extra_args=()
if [ -f "${CLIP_MODEL_PATH}" ]; then
  extra_args+=(--clip_model_path "${CLIP_MODEL_PATH}")
fi

python generate_attribute_embeddings.py \
  --attr_csv "${ATTR_CSV}" \
  --output_path "${OUTPUT_PATH}" \
  --clip_model "ViT-B/32" \
  "${extra_args[@]}" \
  --use_prompt \
  --prompt_template "an ultrasound image showing {attr}" \
  --smart_prompt \
  --overwrite
