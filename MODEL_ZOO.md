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

## What to release

For a public paper-code release, we recommend:

- Release source code on GitHub.
- Release trained weights separately through GitHub Releases, Zenodo, Hugging Face, or an institutional file service.
- Do not commit raw datasets, generated logs, wandb runs, or checkpoint files to git.
- Provide checksums for released weights when possible.

## Placeholder download table

Update this table after uploading the files.

| File | Description | Download |
| --- | --- | --- |
| `resnet50-11ad3fa6.pth` | ImageNet-pretrained ResNet50 backbone | TBD |
| `vit_b_16-c867db91.pth` | ImageNet-pretrained ViT-B/16 backbone | TBD |
| `attribute_embeddings_3cls_breast.pt` | CLIP-encoded breast attribute embeddings | TBD |
| `best_model.pth` | Trained AttrGuide checkpoint | TBD |

