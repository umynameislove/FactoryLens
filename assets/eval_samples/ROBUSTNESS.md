# Hazelnut Robustness Study — Model B19

## Reproduction

- Detector: model B19, ResNet18 layer2+layer3 fusion with greedy coreset, 384-dimensional patch embeddings.
- Dataset: `MVTec/hazelnut`
- Selected images: 6
- Selected-image content fingerprint: `97d9f8d70b317c37469622758526b86604d9cc83cc2f03615932416d4f937b57`
- Memory bank: `memory_bank_eval.npz`
- Memory-bank shape: `(1024, 384)`
- Memory-bank SHA-256: `b4512845c32d9513df3bdabb1a436d0d7551f85292c9ea71882802f7e982aca3`
- Model checkpoint: `resnet18-f37072fd.pth` (SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`)
- Image size: `512`
- Verdict threshold: `0.3133` (`score >= threshold` means defect)
- Gaussian-noise seed: `13`
- Baseline accuracy on selected subset: `0.833`
- Raw measurements: [`robustness_metrics.csv`](robustness_metrics.csv)
- Verify command: `python scripts/robustness_study.py --dataset-root data/mvtec/hazelnut --memory-bank data/memory_bank_eval.npz --torch-cache data/torch --good-count 2 --defect-per-type 1 --image-size 512 --threshold 0.3133 --seed 13`
- Test command: `python -m pytest -q tests/test_robustness_study.py`

Selected images:

- `test/good/000.png`
- `test/good/001.png`
- `test/crack/000.png`
- `test/cut/000.png`
- `test/hole/000.png`
- `test/print/000.png`

## Aggregate Results

| Perturbation | Level | Mean score | Mean drift | Mean abs drift | Max abs drift | Flip rate |
|---|---|---:|---:|---:|---:|---:|
| brightness | 0 | 0.3217 | +0.0000 | 0.0000 | 0.0000 | 0.0% |
| brightness | -30% | 0.3388 | +0.0170 | 0.0170 | 0.0239 | 16.7% |
| brightness | -15% | 0.3365 | +0.0148 | 0.0148 | 0.0210 | 16.7% |
| brightness | +15% | 0.3221 | +0.0004 | 0.0014 | 0.0028 | 0.0% |
| brightness | +30% | 0.3248 | +0.0031 | 0.0039 | 0.0084 | 16.7% |
| contrast | 0 | 0.3217 | +0.0000 | 0.0000 | 0.0000 | 0.0% |
| contrast | -30% | 0.3197 | -0.0021 | 0.0021 | 0.0042 | 16.7% |
| contrast | -15% | 0.3205 | -0.0012 | 0.0012 | 0.0029 | 0.0% |
| contrast | +15% | 0.3235 | +0.0018 | 0.0018 | 0.0029 | 0.0% |
| contrast | +30% | 0.3245 | +0.0028 | 0.0028 | 0.0036 | 0.0% |
| gaussian_blur | 0 | 0.3217 | +0.0000 | 0.0000 | 0.0000 | 0.0% |
| gaussian_blur | sigma 1.0 | 0.3216 | -0.0001 | 0.0021 | 0.0030 | 0.0% |
| gaussian_blur | sigma 2.0 | 0.3238 | +0.0021 | 0.0031 | 0.0061 | 0.0% |
| gaussian_blur | sigma 3.0 | 0.3280 | +0.0062 | 0.0062 | 0.0097 | 16.7% |
| gaussian_noise | 0 | 0.3217 | +0.0000 | 0.0000 | 0.0000 | 0.0% |
| gaussian_noise | std 2% | 0.3209 | -0.0009 | 0.0012 | 0.0027 | 16.7% |
| gaussian_noise | std 5% | 0.3221 | +0.0004 | 0.0029 | 0.0062 | 0.0% |
| gaussian_noise | std 10% | 0.3282 | +0.0064 | 0.0074 | 0.0124 | 16.7% |
| jpeg | 0 | 0.3217 | +0.0000 | 0.0000 | 0.0000 | 0.0% |
| jpeg | quality 90 | 0.3224 | +0.0006 | 0.0006 | 0.0014 | 0.0% |
| jpeg | quality 70 | 0.3236 | +0.0019 | 0.0020 | 0.0043 | 0.0% |
| jpeg | quality 50 | 0.3253 | +0.0035 | 0.0040 | 0.0075 | 16.7% |
| rotation | 0 | 0.3217 | +0.0000 | 0.0000 | 0.0000 | 0.0% |
| rotation | -15 deg | 0.3219 | +0.0002 | 0.0044 | 0.0093 | 0.0% |
| rotation | -10 deg | 0.3227 | +0.0009 | 0.0025 | 0.0053 | 0.0% |
| rotation | -5 deg | 0.3211 | -0.0007 | 0.0027 | 0.0060 | 0.0% |
| rotation | +5 deg | 0.3223 | +0.0005 | 0.0020 | 0.0036 | 0.0% |
| rotation | +10 deg | 0.3216 | -0.0001 | 0.0017 | 0.0038 | 0.0% |
| rotation | +15 deg | 0.3214 | -0.0003 | 0.0020 | 0.0041 | 16.7% |

## Plots

### Brightness

![Brightness](robustness_brightness.png)

### Contrast

![Contrast](robustness_contrast.png)

### Gaussian Blur

![Gaussian Blur](robustness_gaussian_blur.png)

### Gaussian Noise

![Gaussian Noise](robustness_gaussian_noise.png)

### Jpeg

![Jpeg](robustness_jpeg.png)

### Rotation

![Rotation](robustness_rotation.png)

## Engineering Conclusion

- Highest observed flip rate: `brightness` at `-30%` (16.7%).
- Largest mean absolute score drift: `brightness` at `-30%` (0.0170).
- Treat scores close to the threshold as unstable and require human review.
- Standardize camera angle, exposure, focus, and image encoding before inference.
- Do not change the detector or threshold from this study alone; rerun on a larger production-representative set first.

This model B19 experiment measures robustness only. It does not train or modify the model.
