#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${REPO_ROOT}/experiments/fetal/resnet50_attrguide"
cd "${EXP_DIR}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/fetal}"
ATTR_CSV="${ATTR_CSV:-${REPO_ROOT}/examples/attributes_fetal.csv}"
ATTR_EMB_PATH="${ATTR_EMB_PATH:-${REPO_ROOT}/data/attribute_embeddings_fetal.pt}"
BACKBONE_WEIGHT="${BACKBONE_WEIGHT:-${REPO_ROOT}/pretrained/resnet50-11ad3fa6.pth}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/outputs/fetal_resnet50_attrguide}"

mkdir -p "${SAVE_DIR}"

if [ ! -d "${DATA_ROOT}" ]; then echo "DATA_ROOT not found: ${DATA_ROOT}" >&2; echo "Set DATA_ROOT=/path/to/dataset and rerun this script." >&2; exit 1; fi
if [ ! -f "${ATTR_CSV}" ]; then echo "ATTR_CSV not found: ${ATTR_CSV}" >&2; echo "Set ATTR_CSV=/path/to/attributes.csv and rerun this script." >&2; exit 1; fi
if [ ! -f "${ATTR_EMB_PATH}" ]; then echo "ATTR_EMB_PATH not found: ${ATTR_EMB_PATH}" >&2; echo "Run the matching generate_*_embeddings.sh script first, or set ATTR_EMB_PATH." >&2; exit 1; fi
if [ ! -f "${BACKBONE_WEIGHT}" ]; then echo "BACKBONE_WEIGHT not found: ${BACKBONE_WEIGHT}" >&2; echo "Set BACKBONE_WEIGHT=/path/to/resnet50-11ad3fa6.pth and rerun this script." >&2; exit 1; fi

python train_fetal_attribute_model.py \
  --data_root "${DATA_ROOT}" \
  --attr_csv "${ATTR_CSV}" \
  --precomputed_attr_emb "${ATTR_EMB_PATH}" \
  --backbone "resnet50" \
  --resnet_path "${BACKBONE_WEIGHT}" \
  --epochs "${EPOCHS:-100}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --lr "${LR:-0.00005}" \
  --weight_decay "${WEIGHT_DECAY:-1e-3}" \
  --lambda_fus "${LAMBDA_FUS:-0.5}" \
  --lambda_reg "${LAMBDA_REG:-0.3}" \
  --lambda_attr_pred "${LAMBDA_ATTR_PRED:-0.2}" \
  --train_mode "${TRAIN_MODE:-fus}" \
  --fusion_weight "${FUSION_WEIGHT:-0.45}" \
  --temperature "${TEMPERATURE:-0.92}" \
  --save_dir "${SAVE_DIR}" \
  --save_best_only \
  --no_wandb \
  --num_workers "${NUM_WORKERS:-4}"
