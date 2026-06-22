# Model Zoo

Large files are not tracked by git. Please place pretrained backbones, generated attribute embeddings, and trained checkpoints under the following paths.

## Expected local paths

```text
checkpoints/
└── pretrained/
    ├── resnet50-11ad3fa6.pth
    └── vit_b_16-c867db91.pth

data/
├── breastdata/
├── attributes_breast.csv
└── attribute_embeddings_3cls_breast.pt
```

## Download Sources

- ResNet50 ImageNet weights: TorchVision model zoo.
- ViT-B/16 ImageNet weights: TorchVision model zoo.
- BUSI dataset: <https://scholar.cu.edu.eg/?q=afahmy/pages/dataset>
- BUSI Kaggle mirror: <https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset>
- Thyroid ultrasound data: provide the public TCIA/Cancer Imaging Archive or institutional download link used in the final release.

## What to Release

For a public paper-code release, we recommend:

- Release source code on GitHub.
- Release trained weights separately through GitHub Releases, Zenodo, Hugging Face, or an institutional file service.
- Do not commit raw datasets, generated logs, wandb runs, or checkpoint files to git.
- Provide checksums for released weights when possible.
- For private datasets, release only trained weights and anonymized visualization examples when permitted.

## Release Checklist

Update this table after uploading the files.

| File | Description | Download |
| --- | --- | --- |
| `resnet50-11ad3fa6.pth` | ImageNet-pretrained ResNet50 backbone | TBD |
| `vit_b_16-c867db91.pth` | ImageNet-pretrained ViT-B/16 backbone | TBD |
| `attribute_embeddings_3cls_breast.pt` | CLIP-encoded breast attribute embeddings | TBD |
| `resnet50_attrguide_best.pth` | Trained BUSI ResNet50+AttrGuide checkpoint | TBD |
| `vitbase_attrguide_best.pth` | Trained BUSI ViT-B+AttrGuide checkpoint | TBD |
| `thyroid_resnet50_attrguide_best.pth` | Trained thyroid ResNet50+AttrGuide checkpoint, if releasable | TBD |
| `thyroid_vitbase_attrguide_best.pth` | Trained thyroid ViT-B+AttrGuide checkpoint, if releasable | TBD |

## Integrity

After uploading a file, record its checksum:

```bash
sha256sum checkpoints/path/to/file.pth
```
