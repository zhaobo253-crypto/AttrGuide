# AttrGuide for Breast Ultrasound Classification

This repository contains the breast-ultrasound classification code used for the AttrGuide experiments. It follows a compact research-code layout: runnable scripts are under `Scripts/`, generated data and outputs are excluded from version control, and dataset/model paths are provided through command-line arguments.

## Repository layout

```text
AttrGuide-Breast/
├── README.md
├── LICENSE
├── MODEL_ZOO.md
├── environment.yml
├── requirements.txt
├── data/
│   └── .gitkeep
└── Scripts/
    ├── resnet50_attrguide/
    ├── vitbase_attrguide/
    ├── resnet50_baseline/
    ├── vitbase_baseline/
    └── slurm/
```

## Data layout

The default dataset path is `./data/breastdata`. The expected folder structure is:

```text
data/breastdata/
├── train/
│   ├── benign/
│   ├── malignant/
│   └── normal/
├── val/                # optional; if absent, train/ is split into train/val
│   ├── benign/
│   ├── malignant/
│   └── normal/
└── test/
    ├── benign/
    ├── malignant/
    └── normal/
```

Files whose names contain `_mask` are ignored automatically.

The attribute table is expected at:

```text
data/attributes_breast.csv
```

It should contain a `name` or `folder` column matching the class folder names and one or more attribute columns.

## Pretrained weights and checkpoints

Large files are intentionally excluded from git. Put pretrained backbones, generated attribute embeddings, and trained checkpoints in the paths described in [MODEL_ZOO.md](MODEL_ZOO.md).

For a paper-code release, the recommended pattern is:

- Keep code, scripts, and documentation in this GitHub repository.
- Upload large model files through GitHub Releases, Zenodo, Hugging Face, or an institutional file service.
- Add the download links and checksums to `MODEL_ZOO.md`.
- Do not commit raw datasets, logs, wandb runs, or `.pth/.pt/.npy` files.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate attrguide-breast
```

Or using pip:

```bash
pip install -r requirements.txt
```

## Generate attribute embeddings

Run this once before training AttrGuide:

```bash
cd Scripts/resnet50_attrguide
python generate_attribute_embeddings.py \
  --attr_csv ../../data/attributes_breast.csv \
  --output_path ../../data/attribute_embeddings_3cls_breast.pt \
  --clip_model ViT-B/32 \
  --prompt_template "an ultrasound image showing {attr}" \
  --overwrite
```

The ViT-Base AttrGuide folder contains the same embedding script; the generated embedding file can be reused by both backbones.

## Train AttrGuide

ResNet50:

```bash
cd Scripts/resnet50_attrguide
python train_attrguide.py \
  --data_root ../../data/breastdata \
  --attr_csv ../../data/attributes_breast.csv \
  --precomputed_attr_emb ../../data/attribute_embeddings_3cls_breast.pt \
  --backbone resnet50 \
  --resnet_path ../../checkpoints/pretrained/resnet50-11ad3fa6.pth \
  --save_dir ../../checkpoints/resnet50_attrguide \
  --no_wandb
```

ViT-Base:

```bash
cd Scripts/vitbase_attrguide
python train_attrguide.py \
  --data_root ../../data/breastdata \
  --attr_csv ../../data/attributes_breast.csv \
  --precomputed_attr_emb ../../data/attribute_embeddings_3cls_breast.pt \
  --backbone vitbase \
  --resnet_path ../../checkpoints/pretrained/vit_b_16-c867db91.pth \
  --save_dir ../../checkpoints/vitbase_attrguide \
  --no_wandb
```

## Train baselines

ResNet50 baseline:

```bash
cd Scripts/resnet50_baseline
python train_baseline.py \
  --data_root ../../data/breastdata \
  --backbone resnet50 \
  --resnet_path ../../checkpoints/pretrained/resnet50-11ad3fa6.pth \
  --save_dir ../../checkpoints/resnet50_baseline \
  --no_wandb
```

ViT-Base baseline:

```bash
cd Scripts/vitbase_baseline
python train_baseline.py \
  --data_root ../../data/breastdata \
  --backbone vitbase \
  --resnet_path ../../checkpoints/pretrained/vit_b_16-c867db91.pth \
  --save_dir ../../checkpoints/vitbase_baseline \
  --no_wandb
```

## SLURM jobs

Example SLURM scripts are provided in `Scripts/slurm/`. Update the dataset, checkpoint, conda environment, and repository paths before submitting on a cluster.

## Notes

- `data/`, `checkpoints/`, `outputs/`, `logs/`, and `wandb/` are ignored by git.
- Large datasets, pretrained weights, generated embeddings, and trained checkpoints should be released separately.
- The code supports offline Weights & Biases logging through `--wandb_offline`; by default, wandb is disabled unless `--use_wandb` is provided.

## Citation

If you use this code, please cite the corresponding MICCAI 2026 paper.
