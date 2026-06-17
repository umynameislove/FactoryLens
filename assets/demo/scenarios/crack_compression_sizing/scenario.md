# Crack - Compression at sizing

Status: verified with the B10 eval vision settings.

## Image evidence

- Primary scenario image: `image.png`
- Primary original source: `data/mvtec/hazelnut/test/crack/008.png`
- MVTec label: `crack`
- Defect-type evidence: ground-truth folder `test/crack`; `score_image` is only the anomaly gate.
- Vision pipeline: `score_image`
- Vision settings: `memory_bank_path=data/memory_bank_eval.npz`, `image_size=512`, pretrained ResNet18
- Threshold: `0.3884`
- Measured anomaly score: `0.421390`
- Model verdict: `PASS` (`0.421390 >= 0.3884`)
- Heatmap: `image_heatmap.png`

| Demo file | MVTec source | Image number | Score | Verdict | Heatmap |
|---|---|---:|---:|---|---|
| `image.png` | `data/mvtec/hazelnut/test/crack/008.png` | 008 | 0.421390 | PASS | `image_heatmap.png` |
| `crack_009.png` | `data/mvtec/hazelnut/test/crack/009.png` | 009 | 0.416618 | PASS | `crack_009_heatmap.png` |
| `crack_007.png` | `data/mvtec/hazelnut/test/crack/007.png` | 007 | 0.415622 | PASS | `crack_007_heatmap.png` |

## Known issue

- Related known issue: `KI-002 - Compression crack during sizing`
- Source file: `assets/known_issues/KI-002-crack-compression.md`

## Log evidence

- Log file: `logs.csv`
- Smoking-gun line: `sizing_station_01,compression_force_n,128.0,70.0,110.0,FAIL`
- Why it fits: KI-002 points to excessive compression, narrow guide spacing, or a jammed feed path during sizing/feeding. This log shows force above the allowed range at the sizing station, plus a narrow roller gap, so the production evidence matches a compression crack mechanism.

## Final B15 note

The score and heatmap above were generated from this exact committed demo image.
