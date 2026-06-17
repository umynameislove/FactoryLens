# Hole - Impact chip

Status: verified with the B10 eval vision settings.

## Image evidence

- Primary scenario image: `image.png`
- Primary original source: `data/mvtec/hazelnut/test/hole/014.png`
- MVTec label: `hole`
- Defect-type evidence: ground-truth folder `test/hole`; `score_image` is only the anomaly gate.
- Vision pipeline: `score_image`
- Vision settings: `memory_bank_path=data/memory_bank_eval.npz`, `image_size=512`, pretrained ResNet18
- Threshold: `0.3884`
- Measured anomaly score: `0.395427`
- Model verdict: `PASS` (`0.395427 >= 0.3884`)
- Heatmap: `image_heatmap.png`

| Demo file | MVTec source | Image number | Score | Verdict | Heatmap |
|---|---|---:|---:|---|---|
| `image.png` | `data/mvtec/hazelnut/test/hole/014.png` | 014 | 0.395427 | PASS | `image_heatmap.png` |
| `hole_010.png` | `data/mvtec/hazelnut/test/hole/010.png` | 010 | 0.398885 | PASS | `hole_010_heatmap.png` |
| `hole_012.png` | `data/mvtec/hazelnut/test/hole/012.png` | 012 | 0.395313 | PASS | `hole_012_heatmap.png` |

## Known issue

- Related known issue: `KI-006 - Impact chip mistaken as hole`
- Source file: `assets/known_issues/KI-006-hole-impact-chip.md`

## Log evidence

- Log file: `logs.csv`
- Smoking-gun line: `bulk_transfer_02,impact_peak_g,6.8,0.0,4.5,FAIL`
- Why it fits: KI-006 says drop impact, bulk handling collision, or aggressive agitation can remove a shell chip and appear as a hole. This log shows impact above spec at the transfer step, with elevated drop height and low weight, so the evidence supports an impact-chip mechanism.

## Final B15 note

The score and heatmap above were generated from this exact committed demo image.
