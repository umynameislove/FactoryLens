"""Heatmap rendering and region extraction for anomaly score maps."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_heatmap(
    image_path: str,
    score_map: np.ndarray,
    out_dir: str,
    alpha: float = 0.45,
) -> str:
    """Upsample a score map, overlay it on the image, and save a PNG."""

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if score_map.ndim != 2:
        raise ValueError("score_map must have shape (H, W)")

    height, width = image.shape[:2]
    normalized = _normalize_score_map(score_map)
    upsampled = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.clip(upsampled * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0)

    out_path = Path(out_dir) / f"{Path(image_path).stem}_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), overlay):
        raise ValueError(f"Could not write heatmap: {out_path}")

    return out_path.as_posix()


def extract_regions(
    score_map: np.ndarray,
    threshold: float,
    min_area: int = 1,
) -> list[dict[str, object]]:
    """Extract connected high-score regions from a normalized score map."""

    if score_map.ndim != 2:
        raise ValueError("score_map must have shape (H, W)")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if min_area < 1:
        raise ValueError("min_area must be >= 1")

    normalized = _normalize_score_map(score_map)
    mask = (normalized >= threshold).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    regions: list[dict[str, object]] = []
    for component_id in range(1, component_count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component_scores = normalized[labels == component_id]
        score = float(np.clip(component_scores.max(initial=0.0), 0.0, 1.0))
        regions.append({"bbox": [x, y, x + width, y + height], "score": score})

    return sorted(regions, key=lambda region: float(region["score"]), reverse=True)


def _normalize_score_map(score_map: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(score_map.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    min_value = float(finite.min(initial=0.0))
    max_value = float(finite.max(initial=0.0))
    if min_value < 0.0 or max_value > 1.0:
        span = max(max_value - min_value, 1e-12)
        finite = (finite - min_value) / span

    return np.clip(finite, 0.0, 1.0)
