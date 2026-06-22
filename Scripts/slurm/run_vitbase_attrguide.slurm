#!/bin/bash
#SBATCH -J breast_vitbase_attrguide
#SBATCH -o logs/breast_vitbase_attrguide_%j.out
#SBATCH -e logs/breast_vitbase_attrguide_%j.err
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1

module purge
module add nvidia/cuda/12.2
source ~/.bashrc
conda activate attrguide

REPO_ROOT="${REPO_ROOT:-$PWD}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/breastdata}"
ATTR_CSV="${ATTR_CSV:-${REPO_ROOT}/data/attributes_breast.csv}"
ATTR_EMB="${ATTR_EMB:-${REPO_ROOT}/data/attribute_embeddings_3cls_breast.pt}"
WEIGHT_PATH="${WEIGHT_PATH:-${REPO_ROOT}/checkpoints/pretrained/vit_b_16-c867db91.pth}"

cd "${REPO_ROOT}/Scripts/vitbase_attrguide" || exit 1
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/checkpoints/vitbase_attrguide"

python train_attrguide.py \
  --data_root "${DATA_ROOT}" \
  --attr_csv "${ATTR_CSV}" \
  --precomputed_attr_emb "${ATTR_EMB}" \
  --backbone vitbase \
  --resnet_path "${WEIGHT_PATH}" \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.00007 \
  --weight_decay 6e-4 \
  --dropout 0.25 \
  --lambda_attr 0.55 \
  --lambda_cls 0.25 \
  --lambda_fus 0.45 \
  --lambda_reg 0.003 \
  --lambda_attr_pred 0.15 \
  --fusion_weight 0.45 \
  --temperature 0.92 \
  --save_dir "${REPO_ROOT}/checkpoints/vitbase_attrguide" \
  --save_best_only \
  --num_workers 4 \
  --no_wandb
