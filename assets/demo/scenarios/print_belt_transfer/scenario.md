# Print - Belt transfer mark

Status: verified with the B10 eval vision settings.

## Image evidence

- Primary scenario image: `image.png`
- Primary original source: `data/mvtec/hazelnut/test/print/002.png`
- MVTec label: `print`
- Defect-type evidence: ground-truth folder `test/print`; `score_image` is only the anomaly gate.
- Vision pipeline: `score_image`
- Vision settings: `memory_bank_path=data/memory_bank_eval.npz`, `image_size=512`, pretrained ResNet18
- Threshold: `0.3884`
- Measured anomaly score: `0.412831`
- Model verdict: `PASS` (`0.412831 >= 0.3884`)
- Heatmap: `image_heatmap.png`

| Demo file | MVTec source | Image number | Score | Verdict | Heatmap |
|---|---|---:|---:|---|---|
| `image.png` | `data/mvtec/hazelnut/test/print/002.png` | 002 | 0.412831 | PASS | `image_heatmap.png` |
| `print_010.png` | `data/mvtec/hazelnut/test/print/010.png` | 010 | 0.409735 | PASS | `print_010_heatmap.png` |
| `print_006.png` | `data/mvtec/hazelnut/test/print/006.png` | 006 | 0.409551 | PASS | `print_006_heatmap.png` |

## Known issue

- Related known issue: `KI-007 - Ink or belt transfer mark`
- Source file: `assets/known_issues/KI-007-print-transfer-mark.md`

## Log evidence

- Log file: `logs.csv`
- Smoking-gun line: `belt_contact_01,residue_index,0.76,0.00,0.25,FAIL`
- Why it fits: KI-007 points to ink, belt residue, marker material, or packaging debris transferring a visible surface mark. This log shows belt residue above spec and a stale sanitation interval, while surface defect area stays within spec, so it reads as a transfer mark rather than physical shell damage.

## Final B15 note

The score and heatmap above were generated from this exact committed demo image.
