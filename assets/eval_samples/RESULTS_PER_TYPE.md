# Hazelnut Per-Defect Evaluation

## Reproduction

- Input scores: `assets/eval_samples/eval_scores.csv`
- Input SHA-256: `f3a63b688efd895d47295d0b334ab09077e5824d996048e2dd3efdf9f5586eb5`
- Input bytes: 6009
- B10 source report: `assets/eval_samples/RESULTS.md` (SHA-256 `a1e62c81170ab6673454f492fba52f81f93ce42c37fbdadaa465336d5707e997`)
- Label counts: good=40, crack=18, cut=17, hole=18, print=17
- Source images currently present: 110/110
- Verify command: `python scripts/eval_per_type.py --source-root ../FactoryLens`
- Test command: `python -m pytest -q tests/test_eval_per_type.py`
- Current threshold: `0.3884`
- Threshold sweep: `0.30` to `0.50` with step `0.01`
- This command recomputes metrics from measured B10 scores; it does not rerun model inference.
- To regenerate scores from images, rerun the B10 eval with the same model and memory-bank settings first.
- Model B19 eval command: `python -m factorylens.vision.eval_baseline --root data/mvtec --category hazelnut --out-dir assets/eval_samples --memory-bank data/memory_bank_eval.npz --max-train-good 128 --max-patches-per-image 32 --image-size 512 --region-threshold 0.35`
- Model B19 eval memory bank: `data/memory_bank_eval.npz`, shape `(1024, 384)`.

## Overall Threshold Check

### Before (model cũ)

| Threshold | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3884 | 0.682 | 1.000 | 0.500 | 0.667 | 35 | 0 | 40 | 35 |
| 0.3768 | 0.855 | 0.875 | 0.900 | 0.887 | 63 | 9 | 31 | 7 |

### After (model B19)

| Threshold | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3884 | 0.364 | 0.000 | 0.000 | 0.000 | 0 | 0 | 40 | 70 |
| 0.3133 | 0.936 | 0.944 | 0.957 | 0.950 | 67 | 4 | 36 | 3 |

Đề xuất `anomaly_threshold = 0.3133` (model B19) cho Bao.
Recommendation method: maximize F1 over every observed score boundary inside `0.30`–`0.50`, then balanced accuracy, accuracy, and proximity to the current threshold.

## Per-Defect Results

### Before (model cũ)

| Defect | Defect samples | AUROC | Accuracy @ current | Recall @ current | Best F1 threshold | Best F1 |
|---|---:|---:|---:|---:|---:|---:|
| crack | 18 | 0.938 | 0.914 | 0.722 | 0.3838 | 0.919 |
| cut | 17 | 0.800 | 0.737 | 0.118 | 0.3747 | 0.682 |
| hole | 18 | 0.885 | 0.810 | 0.389 | 0.3784 | 0.780 |
| print | 17 | 0.943 | 0.930 | 0.765 | 0.3820 | 0.882 |

### After (model B19)

| Defect | Defect samples | AUROC | Accuracy @ current | Recall @ current | Best F1 threshold | Best F1 |
|---|---:|---:|---:|---:|---:|---:|
| crack | 18 | 0.981 | 0.690 | 0.000 | 0.3133 | 0.900 |
| cut | 17 | 0.918 | 0.702 | 0.000 | 0.3133 | 0.833 |
| hole | 18 | 0.965 | 0.690 | 0.000 | 0.3143 | 0.919 |
| print | 17 | 0.991 | 0.702 | 0.000 | 0.3149 | 0.944 |

### Before/After AUROC Comparison

| Defect | Before (model cũ) | After (model B19) | Delta | Kết quả |
|---|---:|---:|---:|---|
| crack | 0.938 | 0.981 | +0.043 | Tăng |
| cut | 0.800 | 0.918 | +0.118 | Tăng |
| hole | 0.885 | 0.965 | +0.080 | Tăng |
| print | 0.943 | 0.991 | +0.048 | Tăng |

## Plots

![Threshold sensitivity](threshold_sensitivity.png)

![ROC per defect type](roc_per_type.png)

![Confusion matrix](confusion_matrix.png)

## Engineering Reading

- AUROC tăng ở cả bốn loại lỗi sau B19; không có loại nào giảm trong lần đo này.
- Mức tăng lớn nhất là `cut` (`+0.118`), nhưng `cut` vẫn là nhóm yếu nhất với AUROC `0.918`.
- Per-type score ranges overlap with good samples, so a single threshold remains a demo compromise.
- Keep human review for heatmaps and borderline scores; this is not a calibrated defect probability.
- If the memory bank or extractor changes, regenerate the score CSV before reusing this report.
