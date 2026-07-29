# Model Zoo

Large files are not tracked by git. Please place pretrained backbones, generated attribute embeddings, and trained checkpoints under the following local paths.

## Expected Local Paths

```text
checkpoints/
`-- pretrained/
    |-- resnet50-11ad3fa6.pth
    `-- vit_b_16-c867db91.pth

data/
|-- breastdata/
|-- thyroid/
|-- attributes_breast.csv
|-- attributes_thyroid.csv
|-- attribute_embeddings_3cls_breast.pt
`-- attribute_embeddings_2cls_thyroid.pt
```

## Download Sources

- ResNet50 ImageNet weights: TorchVision model zoo.
- ViT-B/16 ImageNet weights: TorchVision model zoo.
- BUSI dataset: <https://scholar.cu.edu.eg/?q=afahmy/pages/dataset>
- DDTI thyroid ultrasound dataset: <https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images?resource=download>

## Release Status

| Dataset | Logs | Checkpoints |
| --- | --- | --- |
| BUSI breast ViT-Base | Sanitized logs in `results/public_logs/breast/` | Release separately because files are large |
| DDTI thyroid | Sanitized logs in `results/public_logs/thyroid/` | Not included; logs only |

## Integrity

When releasing model files separately, include checksums when possible:

```bash
sha256sum checkpoints/path/to/file.pth
```
