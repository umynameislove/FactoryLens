from __future__ import annotations

import cv2
import numpy as np
import pytest

from factorylens.vision.heatmap import extract_regions, make_heatmap


def test_extract_regions_returns_valid_bbox():
    score_map = np.zeros((10, 12), dtype=np.float32)
    score_map[2:5, 3:8] = 0.92
    score_map[8, 10] = 0.3

    regions = extract_regions(score_map, threshold=0.7)

    assert len(regions) == 1
    assert regions[0]["bbox"] == [3, 2, 8, 5]
    assert regions[0]["score"] == pytest.approx(0.92)


def test_extract_regions_filters_small_components():
    score_map = np.zeros((8, 8), dtype=np.float32)
    score_map[1, 1] = 0.95
    score_map[3:6, 3:6] = 0.8

    regions = extract_regions(score_map, threshold=0.7, min_area=4)

    assert len(regions) == 1
    assert regions[0]["bbox"] == [3, 3, 6, 6]


def test_make_heatmap_writes_overlay_png(tmp_path):
    image_path = tmp_path / "sample.png"
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 120
    cv2.imwrite(str(image_path), image)

    score_map = np.zeros((6, 8), dtype=np.float32)
    score_map[2:4, 3:5] = 1.0

    out_path = make_heatmap(str(image_path), score_map, str(tmp_path / "heatmaps"))
    overlay = cv2.imread(out_path, cv2.IMREAD_COLOR)

    assert out_path.endswith("sample_heatmap.png")
    assert overlay is not None
    assert overlay.shape == image.shape
