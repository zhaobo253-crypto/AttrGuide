#!/bin/bash
#SBATCH -J breast_resnet50_attrguide
#SBATCH -o logs/breast_resnet50_attrguide_%j.out
#SBATCH -e logs/breast_resnet50_attrguide_%j.err
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1

module purge
module add nvidia/cuda/12.2
source ~/.bashrc
conda activate attrguide-breast

REPO_ROOT="${REPO_ROOT:-$PWD}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/breastdata}"
ATTR_CSV="${ATTR_CSV:-${REPO_ROOT}/data/attributes_breast.csv}"
ATTR_EMB="${ATTR_EMB:-${REPO_ROOT}/data/attribute_embeddings_3cls_breast.pt}"
WEIGHT_PATH="${WEIGHT_PATH:-${REPO_ROOT}/checkpoints/pretrained/resnet50-11ad3fa6.pth}"

cd "${REPO_ROOT}/Scripts/resnet50_attrguide" || exit 1
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/checkpoints/resnet50_attrguide"

python train_attrguide.py \
  --data_root "${DATA_ROOT}" \
  --attr_csv "${ATTR_CSV}" \
  --precomputed_attr_emb "${ATTR_EMB}" \
  --backbone resnet50 \
  --resnet_path "${WEIGHT_PATH}" \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.0001 \
  --weight_decay 3e-4 \
  --lambda_attr 0.45 \
  --lambda_cls 0.30 \
  --lambda_fus 0.40 \
  --lambda_reg 0.004 \
  --lambda_attr_pred 0.15 \
  --fusion_weight 0.35 \
  --temperature 0.95 \
  --save_dir "${REPO_ROOT}/checkpoints/resnet50_attrguide" \
  --save_best_only \
  --num_workers 4 \
  --no_wandb

