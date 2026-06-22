#!/bin/bash
#SBATCH -J breast_vitbase_baseline
#SBATCH -o logs/breast_vitbase_baseline_%j.out
#SBATCH -e logs/breast_vitbase_baseline_%j.err
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
WEIGHT_PATH="${WEIGHT_PATH:-${REPO_ROOT}/checkpoints/pretrained/vit_b_16-c867db91.pth}"

cd "${REPO_ROOT}/Scripts/vitbase_baseline" || exit 1
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/checkpoints/vitbase_baseline"

python train_baseline.py \
  --data_root "${DATA_ROOT}" \
  --backbone vitbase \
  --resnet_path "${WEIGHT_PATH}" \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.00007 \
  --weight_decay 6e-4 \
  --dropout 0.25 \
  --save_dir "${REPO_ROOT}/checkpoints/vitbase_baseline" \
  --save_best_only \
  --num_workers 4 \
  --no_wandb
