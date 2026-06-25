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
- AttrGuide released checkpoints: <https://huggingface.co/chameleon111/AttrGuide/tree/main/checkpoints>
- BUSI dataset: <https://scholar.cu.edu.eg/?q=afahmy/pages/dataset>
- DDTI thyroid ultrasound dataset: <https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images?resource=download>

## What to Release

For a public paper-code release, we recommend:

- Release source code on GitHub.
- Release trained weights separately through GitHub Releases, Zenodo, Hugging Face, or an institutional file service.
- Do not commit raw datasets, raw logs, wandb runs, or checkpoint files to git.
- Provide checksums for released weights when possible.
- Do not release private fetal ultrasound data, logs, or checkpoints.

Sanitized public logs are tracked in `results/public_logs/`. Raw logs should not be committed directly.

## Release Checklist

Update this table after uploading the files.

| File | Description | Download |
| --- | --- | --- |
| `resnet50-11ad3fa6.pth` | ImageNet-pretrained ResNet50 backbone | TBD |
| `vit_b_16-c867db91.pth` | ImageNet-pretrained ViT-B/16 backbone | TBD |
| `attribute_embeddings_3cls_breast.pt` | CLIP-encoded BUSI breast attribute embeddings | `rebuttal_q4/breast/attribute_embeddings_3cls_breast.pt` |
| `attribute_embeddings_2cls_thyroid.pt` | CLIP-encoded DDTI thyroid attribute embeddings | TBD |
| `resnet50_attrguide_best.pth` | Trained BUSI ResNet50+AttrGuide checkpoint | <https://huggingface.co/chameleon111/AttrGuide/tree/main/checkpoints> |
| `vitbase_attrguide_best.pth` | Trained BUSI ViT-B+AttrGuide checkpoint | <https://huggingface.co/chameleon111/AttrGuide/tree/main/checkpoints> |

Current release status:

| Dataset | Logs | Checkpoints |
| --- | --- | --- |
| BUSI breast | Sanitized logs in `results/public_logs/breast/`; Q4 qualitative CSV in `rebuttal_q4/breast/` | Released through Hugging Face checkpoints link above |
| DDTI thyroid | Sanitized logs in `results/public_logs/thyroid/` | Not included; logs only |
| Private fetal ultrasound | Not included | Not included |

## Integrity

After uploading a file, record its checksum:

```bash
sha256sum checkpoints/path/to/file.pth
```
