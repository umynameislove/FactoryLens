# Vision Baseline

This folder contains the first hazelnut vision baseline for FactoryLens. It is a
small PatchCore-style pipeline: build a memory bank from known-good images, score
a test image by nearest patch distance, then turn the score map into a heatmap
and simple bounding boxes.

It is meant to be good enough for the local demo and easy to plug into the
backend. It is not a trained detector and it is not a claim of production
accuracy.

## Files

| File | Purpose |
|---|---|
| `embeddings.py` | ResNet18 layer2 feature extraction and patch embeddings |
| `anomaly.py` | memory bank build/load plus `score_image()` |
| `heatmap.py` | heatmap overlay and connected-component bbox extraction |
| `eval_baseline.py` | mini-eval over `test/good` vs defect folders |

## Local data expected

The MVTec hazelnut folder should be present locally:

```text
data/mvtec/hazelnut/
  train/good/
  test/good/
  test/crack/
  test/cut/
  test/hole/
  test/print/
  ground_truth/
```

Raw data, memory banks, and downloaded model weights stay under `data/` and are
ignored by git.

## Build a memory bank

```bash
python - <<'PY'
from pathlib import Path
from factorylens.vision.anomaly import build_memory_bank
from factorylens.vision.embeddings import PatchEmbeddingExtractor

good_images = sorted(str(p) for p in Path("data/mvtec/hazelnut/train/good").glob("*.png"))[:24]
extractor = PatchEmbeddingExtractor(image_size=96, pretrained=True)
build_memory_bank(
    good_images,
    out_path="data/memory_bank.npz",
    extractor=extractor,
    max_patches_per_image=24,
)
PY
```

The first run downloads ResNet18 weights into `data/torch/checkpoints/`. That is
intentional so the project does not write model files into the repo.

## Score one image

```bash
python - <<'PY'
from factorylens.vision.anomaly import score_image
from factorylens.vision.embeddings import PatchEmbeddingExtractor

extractor = PatchEmbeddingExtractor(image_size=96, pretrained=True)
anomaly_score, score_map = score_image(
    "data/mvtec/hazelnut/test/crack/000.png",
    memory_bank_path="data/memory_bank.npz",
    extractor=extractor,
)
print(anomaly_score)
print(score_map.shape)
PY
```

Output:

- `anomaly_score`: float in `[0, 1]`
- `score_map`: 2D numpy array, low resolution feature grid

Backend handoff: use `anomaly_score` directly for `AnalysisResponse.anomaly_score`.

## Heatmap and regions

```bash
python - <<'PY'
from factorylens.vision.anomaly import score_image
from factorylens.vision.embeddings import PatchEmbeddingExtractor
from factorylens.vision.heatmap import extract_regions, make_heatmap

image_path = "data/mvtec/hazelnut/test/crack/000.png"
extractor = PatchEmbeddingExtractor(image_size=96, pretrained=True)
_, score_map = score_image(image_path, memory_bank_path="data/memory_bank.npz", extractor=extractor)

heatmap_path = make_heatmap(image_path, score_map, "heatmaps")
regions = extract_regions(
    score_map,
    threshold=0.35,
    min_area=2,
    image_shape=(1024, 1024),
)

print(heatmap_path)
print(regions)
PY
```

Region output matches the API contract shape:

```python
{"bbox": [x1, y1, x2, y2], "score": 0.82}
```

Pass `image_shape=(height, width)` when the API needs pixel bboxes for
`DefectRegion`. If `image_shape` is omitted, bbox coordinates stay in score-map
grid coordinates.

## Mini-eval

Run:

```bash
python -m factorylens.vision.eval_baseline \
  --max-train-good 24 \
  --max-patches-per-image 24 \
  --image-size 96 \
  --region-threshold 0.35
```

Current result on the local hazelnut dataset:

```text
AUROC: 0.943
threshold: 0.3884
accuracy: 0.900
precision: 0.940
recall: 0.900
```

Use `0.3884` as the default image-level threshold for the demo. Eval artifacts
are in `assets/eval_samples/RESULTS.md`.

## Known limits

- This is an anomaly baseline, not a classifier trained for hazelnut defects.
- Scores are close together, so do not oversell the threshold.
- Heatmaps are useful for review, but localization still needs human checking.
- The score map is low resolution because it comes from ResNet feature patches.
- Memory-bank quality matters. If the good-image set changes, rerun eval and
  update the threshold.

## Tests

```bash
pytest tests/test_anomaly.py tests/test_heatmap.py tests/test_eval_baseline.py
pytest
```

The full suite is expected to pass. Slow model/eval runs are kept out of unit
tests; the test files use small fake inputs where possible.
