# Scripts

Small local utilities for the data and vision lane.

## Build the hazelnut memory bank

Install the vision dependencies first:

```bash
source .venv/bin/activate
pip install -e ".[vision]"
```

Then build the memory bank from the local MVTec hazelnut `train/good` images:

```bash
python scripts/build_memory_bank.py
```

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
shape, so it is easy to confirm the build finished cleanly.
