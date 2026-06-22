# Cut - Trimming station nick

Status: re-verified with the model B19 eval vision settings.

## Image evidence

- Primary scenario image: `image.png`
- Primary original source: `data/mvtec/hazelnut/test/cut/015.png`
- MVTec label: `cut`
- Defect-type evidence: ground-truth folder `test/cut`; `score_image` is only the anomaly gate.
- Vision pipeline: `score_image`
- Vision settings: `memory_bank_path=data/memory_bank_eval.npz`, `image_size=512`, pretrained ResNet18 layer2+layer3 fusion
- Threshold: `0.3133`
- Measured anomaly score: `0.320435`
- Model verdict: `PASS` (`0.320435 >= 0.3133`)
- Heatmap: `image_heatmap.png`

| Demo file | MVTec source | Image number | Score | Verdict | Heatmap |
|---|---|---:|---:|---|---|
| `image.png` | `data/mvtec/hazelnut/test/cut/015.png` | 015 | 0.320435 | PASS | `image_heatmap.png` |
| `cut_001.png` | `data/mvtec/hazelnut/test/cut/001.png` | 001 | 0.319574 | PASS | `cut_001_heatmap.png` |

Note: these two selected B15 cut images remain above the model B19 threshold
`0.3133`. A new full-folder candidate scan was not part of this optional B20
scenario refresh.

## Known issue

- Related known issue: `KI-010 - Knife nick from trimming station`
- Source file: `assets/known_issues/KI-010-cut-trimming-nick.md`

## Log evidence

- Log file: `logs.csv`
- Smoking-gun line: `trimming_station_01,blade_clearance_mm,0.18,0.30,0.80,FAIL`
- Why it fits: KI-010 describes a short straight nick from a trimming blade, guide edge, or misaligned metal lip. This log shows the trimming blade clearance below spec and a high guide burr score, so the production evidence supports a cut caused by trimming contact.

## Final B15 note

The model B19 score was re-measured from the listed local MVTec source image.
Scenario images and heatmaps remain uncommitted in this text-only bundle.
