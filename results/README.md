# Public Result Logs

This folder contains sanitized training logs for the public BUSI breast ultrasound experiments and the DDTI thyroid ultrasound experiments.

## Data Sources

| Dataset | Task | Source |
| --- | --- | --- |
| BUSI | 3-class breast ultrasound classification: benign / malignant / normal | <https://scholar.cu.edu.eg/?q=afahmy/pages/dataset> |
| DDTI thyroid | 2-class thyroid ultrasound classification: benign / malignant | <https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images?resource=download> |

Raw medical images are not redistributed in this repository. Please download the public datasets from their original sources and arrange them following the instructions in the main README.

## Logs

```text
results/public_logs/
|-- breast/
|   |-- resnet50_baseline.txt
|   |-- resnet50_attrguide.txt
|   |-- vitbase_baseline.txt
|   `-- vitbase_attrguide.txt
`-- thyroid/
    |-- resnet50_baseline.txt
    |-- resnet50_attrguide.txt
    |-- vitbase_baseline.txt
    `-- vitbase_attrguide.txt
```

The logs have been sanitized before release. Local server paths, usernames, scratch/project directories, and offline WandB run identifiers have been replaced with placeholders such as `<PROJECT_DIR>`, `<THYROID_PROJECT_DIR>`, `<BREAST_DATA_ROOT>`, and `<PRETRAINED_WEIGHT>`.

## Weights

Breast ultrasound checkpoints are not committed to the git repository because several model files are close to or above GitHub's regular file-size limit. Release them through GitHub Releases, Zenodo, Hugging Face, or an institutional file service, then record the final download links in `MODEL_ZOO.md`.

The thyroid experiments currently provide logs only; no thyroid checkpoints are included in this public repository.
