# B15 Demo Scenarios

These scenarios were re-measured with the real `score_image` pipeline using the
model B19 eval settings: `memory_bank_path=data/memory_bank_eval.npz`,
`image_size=512`, pretrained ResNet18 layer2+layer3 fusion, and threshold
`0.3133`.

Important scope note: `score_image` is an anomaly detector, not a defect-type
classifier. The defect type in this pack comes from the MVTec ground-truth
folder in each source path, for example `data/mvtec/hazelnut/test/crack/008.png`.

## Summary

| Scenario | Defect type | Passing MVTec image numbers | Primary score | Known issue | Primary FAIL evidence |
|---|---|---|---:|---|---|
| `crack_compression_sizing` | crack | `008`, `009`, `007` | 0.343104 | KI-002 | `compression_force_n` above spec at `sizing_station_01` |
| `cut_trimming_nick` | cut | `015`, `001` | 0.320435 | KI-010 | `blade_clearance_mm` below spec at `trimming_station_01` |
| `hole_impact_chip` | hole | `014`, `010`, `012` | 0.328586 | KI-006 | `impact_peak_g` above spec at `bulk_transfer_02` |
| `print_belt_transfer` | print | `002`, `010`, `006` | 0.325660 | KI-007 | `residue_index` above spec at `belt_contact_01` |

Full per-image evidence is in `vision_scores.csv`. All 11 selected scenario
images remain above the model B19 threshold. A new full-folder candidate scan
was not part of this optional B20 scenario refresh.

## Score Reproduction

The score table can be reproduced from the model B19 B10 eval pipeline. The
threshold `0.3133` comes from the B20 eval, so the scenario scores use the same
eval memory-bank recipe rather than a full-size memory bank.

Recreate the B10-style memory bank and eval artifacts:

```bash
python -m factorylens.vision.eval_baseline \
  --root data/mvtec \
  --category hazelnut \
  --out-dir /tmp/factorylens_b15_eval_repro \
  --memory-bank data/memory_bank_eval.npz \
  --max-train-good 128 \
  --max-patches-per-image 32 \
  --image-size 512 \
  --region-threshold 0.35
```

Then rerun scenario scoring:

```bash
python - <<'PY'
from pathlib import Path
from factorylens.vision.anomaly import score_image
from factorylens.vision.embeddings import PatchEmbeddingExtractor

root = Path("assets/demo/scenarios")
threshold = 0.3133
extractor = PatchEmbeddingExtractor(
    image_size=512,
    pretrained=True,
    allow_untrained_fallback=False,
)

for image_path in sorted(root.glob("*/*.png")):
    if "heatmap" in image_path.name:
        continue
    score, _ = score_image(
        str(image_path),
        memory_bank_path="data/memory_bank_eval.npz",
        extractor=extractor,
    )
    verdict = "PASS" if score >= threshold else "FAIL"
    print(f"{image_path},{score:.6f},{verdict}")
PY
```

Do not mix these numbers with a different memory bank. A full memory bank built
from all `train/good` images changes the score distribution.

## Required checks before marking B15 done

1. Run `python scripts/check_scenarios.py assets/demo/scenarios`.
2. Reproduce or confirm the B10-style eval memory bank described above.
3. Rerun the B15 vision scoring command if any image changes.
4. Keep only scenarios whose measured score is at least `0.3133`.
5. Regenerate `image_heatmap.png` for each kept scenario after every image change.

If any candidate image scores below threshold, replace that image with another
image from the same MVTec defect folder and rerun the same check. Do not hide
failed candidates in the final scenario writeup.
