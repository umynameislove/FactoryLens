# Heatmap Samples

These PNGs are small review artifacts generated from the local MVTec hazelnut
dataset with the B5/B6 baseline.

Generation settings:

- Feature extractor: ResNet18 layer2 features, image size 96
- Memory bank: first 12 `train/good` images, 16 patches per image
- Region smoke threshold: 0.35
- Output heatmaps: copied from `heatmaps/` into this directory for PR review

| File | Source label | Smoke regions at threshold 0.35 |
|---|---:|---:|
| hazelnut_crack_heatmap.png | crack | 1 |
| hazelnut_cut_heatmap.png | cut | 3 |
| hazelnut_hole_heatmap.png | hole | 1 |
| hazelnut_print_heatmap.png | print | 1 |

These samples are not the final evaluation threshold. B7 will choose and report
the threshold using a broader good-vs-defect mini-eval.
