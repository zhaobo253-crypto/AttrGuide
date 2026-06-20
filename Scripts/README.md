# Scripts

The code is organized by backbone and training mode:

- `resnet50_attrguide/`: ResNet50 with the attribute-guided branch and adaptive fusion.
- `vitbase_attrguide/`: ViT-Base with the attribute-guided branch and adaptive fusion.
- `resnet50_baseline/`: ResNet50 baseline classifier without attribute guidance.
- `vitbase_baseline/`: ViT-Base baseline classifier without attribute guidance.
- `slurm/`: cluster job examples.

Each experiment folder is self-contained and includes its dataset loader, model definition, training entrypoint, and utilities.

