# Rebuttal Q4 Materials

This folder contains the public reproducibility and qualitative-analysis artifacts prepared for Q4: interpretability, qualitative cases, clinical use, and reproducibility.

## Contents

```text
rebuttal_q4/
`-- breast/
    |-- attributes_breast_3cls.csv
    |-- attribute_embeddings_3cls_breast.pt
    `-- test_multibranch_reviewer.csv
```

## Qualitative Analysis Summary

The file `breast/test_multibranch_reviewer.csv` reports one row per BUSI breast test sample with:

- ground-truth class;
- visual-branch prediction;
- attribute-branch prediction;
- fusion-branch prediction;
- reviewer-oriented case tags;
- top-5 predicted semantic attributes and their probabilities.

On the 275-sample breast test split, the visual branch correctly predicts 229 samples, the attribute branch correctly predicts 229 samples, and the fusion branch correctly predicts 230 samples. The attribute prediction accuracy is 83.28% after rounding, providing reliable decision clues for the downstream fusion module. The CSV includes five cases where the visual branch is wrong while the attribute branch is correct, and three malignant cases where the visual branch predicts benign but the fusion branch correctly predicts malignant.

Representative correction cases:

| Sample | Ground truth | Visual | Attribute | Fusion | Top-5 attributes |
| --- | --- | --- | --- | --- | --- |
| `malignant (106).png` | malignant | benign | malignant | malignant | poorly circumscribed boundary; circumscribed margin; spiculated margin; well defined boundary; taller than wide |
| `malignant (133).png` | malignant | benign | malignant | malignant | spiculated margin; poorly circumscribed boundary; circumscribed margin; indistinct margin; taller than wide |
| `malignant (179).png` | malignant | benign | malignant | malignant | spiculated margin; taller than wide; architectural distortion present; indistinct margin; irregular shape |

These cases support the rebuttal claim that semantic attributes provide clinically meaningful decision clues and that adaptive fusion can correct visual-head errors, especially when a malignant case is visually misclassified as benign.

## Reproducibility

The breast attribute table and generated CLIP attribute embeddings are tracked here so other researchers can reproduce the attribute-guided branch inputs:

- `breast/attributes_breast_3cls.csv`
- `breast/attribute_embeddings_3cls_breast.pt`
- `breast/test_multibranch_reviewer.csv`

Best model weights are released separately at <https://huggingface.co/chameleon111/AttrGuide/tree/main/checkpoints>.

The private fetal ultrasound images, fetal CAM maps, and fetal embeddings are not committed to this public repository because they are private clinical-data-derived artifacts. For paper release, they should be shared only through an approved controlled-access route after data-use and ethics review.

## Clinical Utility

The qualitative CSV is designed for physician review: senior clinicians can inspect whether the top-ranked attributes align with diagnostic cues such as lesion shape, margin, orientation, posterior shadowing, calcification, vascularity, and tissue distortion. This supports a future clinical-acceptance study measuring both diagnostic accuracy and trust in the model explanations.
