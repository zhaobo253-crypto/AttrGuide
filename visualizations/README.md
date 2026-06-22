# Visualization Examples

This folder is reserved for public attribute or CAM visualization examples.

Recommended layout:

```text
visualizations/
├── successful_cases/
│   ├── benign_001.png
│   ├── malignant_001.png
│   └── normal_001.png
└── failure_cases/
    ├── benign_001.png
    ├── malignant_001.png
    └── normal_001.png
```

For public release, include a small balanced set of successful and failed examples, such as three to five images per group. Failed examples are useful because they honestly show when attribute localization remains imperfect.

For private clinical images, remove all patient names, dates, hospital identifiers, accession numbers, and other protected information before release. If an image contains embedded metadata or burned-in text, crop or mask those regions before adding the image here.
