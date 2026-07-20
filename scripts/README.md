# Scripts

Small local utilities for the data and vision lane.

## Build the hazelnut memory bank

Install the vision dependencies first:

```bash
source .venv/bin/activate
pip install -e ".[vision]"
```

For the reproducible v2 handoff, build and calibrate the bank with the same
extractor settings and record its manifest/provenance:

```bash
mkdir -p /tmp/vision_bundle_v2
python scripts/build_memory_bank.py \
  --train-dir data/mvtec/hazelnut/train/good \
  --out /tmp/vision_bundle_v2/memory_bank.npz \
  --image-size 512 \
  --max-patches-per-image 32 \
  --coreset-size 1024 \
  --manifest-out /tmp/vision_bundle_v2/MANIFEST.txt \
  --manifest-md5-out /tmp/vision_bundle_v2/MANIFEST_md5.txt \
  --build-metadata-out /tmp/vision_bundle_v2/build_metadata.json

python scripts/eval_threshold_full.py \
  --root data/mvtec/hazelnut \
  --train-dir data/mvtec/hazelnut/train/good \
  --memory-bank /tmp/vision_bundle_v2/memory_bank.npz \
  --image-size 512 \
  --selection youden \
  --scores-out /tmp/vision_bundle_v2/eval_scores_v2.csv \
  --output /tmp/vision_bundle_v2/eval_output_v2.txt \
  --metadata-out /tmp/vision_bundle_v2/METADATA.md \
  --build-metadata /tmp/vision_bundle_v2/build_metadata.json
```

`eval_threshold_full.py` rejects a non-391-image train set, checks every one
of the 110 test images, verifies no train/test MD5 overlap, and records the
bank SHA-256 with the recommended threshold. Do not reuse a threshold with a
different memory-bank hash. The generated `memory_bank.npz` and `train_good/`
are delivery artifacts: keep them out of Git and distribute the complete
bundle privately.

If `python3` still points outside the virtualenv on your machine, call the venv
interpreter directly:

```bash
.venv/bin/python scripts/build_memory_bank.py
```

By default this reads:

```text
data/mvtec/hazelnut/train/good/
```

and writes:

```text
data/memory_bank.npz
```

That output file is intentionally ignored by git. It is the local artifact used
by `analyze_image_defect` when scoring real images.

Useful overrides:

```bash
python scripts/build_memory_bank.py \
  --train-dir data/mvtec/hazelnut/train/good \
  --out data/memory_bank.npz
```

The script prints how many training images were used and the saved memory-bank
shape plus `distance_scale`, so it is easy to confirm the build finished
cleanly.
