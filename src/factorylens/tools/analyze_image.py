"""Safe adapter around the standalone vision anomaly baseline."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from factorylens.config import Settings, get_settings
from factorylens.schemas import DefectRegion, ImageDefectResult

Scorer = Callable[[str], tuple[float, Any]]
Heatmapper = Callable[[str, Any, str], str]
RegionExtractor = Callable[[Any, float], list[dict[str, object]]]

_IMAGE_UNAVAILABLE_WARNING = "image not found/unreadable"
_MEMORY_BANK_UNAVAILABLE_WARNING = (
    "anomaly memory bank not available; run the build step"
)
_ANALYSIS_FAILED_WARNING = "image analysis failed"


def analyze_image_defect(
    image_path: str,
    category: str | None = None,
    settings: Settings | None = None,
    *,
    scorer: Scorer | None = None,
    heatmapper: Heatmapper | None = None,
    region_extractor: RegionExtractor | None = None,
) -> ImageDefectResult:
    """Score one image and return a contract-valid, path-safe result.

    The real vision functions are imported only when an injected implementation
    is absent, so callers and unit tests can import this module without Torch or
    OpenCV installed.
    """

    resolved_settings = settings or get_settings()
    try:
        source_path = Path(image_path).expanduser().resolve()
        image_available = source_path.is_file()
    except (OSError, RuntimeError):
        image_available = False
    if not image_available:
        return _warning_result(_IMAGE_UNAVAILABLE_WARNING)

    try:
        memory_bank_path = (
            Path(resolved_settings.vision_memory_bank_path).expanduser().resolve()
        )
        memory_bank_available = memory_bank_path.is_file()
    except (OSError, RuntimeError):
        memory_bank_available = False
    if not memory_bank_available:
        return _warning_result(_MEMORY_BANK_UNAVAILABLE_WARNING)

    try:
        if scorer is None:
            from factorylens.vision.anomaly import score_image

            def configured_scorer(path: str) -> tuple[float, Any]:
                return score_image(
                    path,
                    memory_bank_path=str(memory_bank_path),
                )

            scorer = configured_scorer

        if heatmapper is None:
            from factorylens.vision.heatmap import make_heatmap

            heatmapper = make_heatmap

        if region_extractor is None:
            from PIL import Image

            from factorylens.vision.heatmap import extract_regions

            with Image.open(source_path) as image:
                image_shape = (image.height, image.width)

            def configured_region_extractor(
                score_map: Any,
                threshold: float,
            ) -> list[dict[str, object]]:
                return extract_regions(
                    score_map,
                    threshold,
                    image_shape=image_shape,
                )

            region_extractor = configured_region_extractor

        raw_score, score_map = scorer(str(source_path))
        anomaly_score = _clamp_score(raw_score)
        state = (
            "defect" if anomaly_score >= resolved_settings.anomaly_threshold else "ok"
        )
        defect_label = f"{category}:{state}" if category else state

        raw_heatmap_path = heatmapper(
            str(source_path),
            score_map,
            resolved_settings.heatmap_dir,
        )
        heatmap_path = _safe_relative_heatmap_path(
            raw_heatmap_path,
            resolved_settings.heatmap_dir,
        )
        defect_regions = _map_regions(
            region_extractor(
                score_map,
                resolved_settings.anomaly_threshold,
            )
        )
    except Exception:
        return _warning_result(_ANALYSIS_FAILED_WARNING)

    return ImageDefectResult(
        anomaly_score=anomaly_score,
        defect_label=defect_label,
        defect_regions=defect_regions,
        heatmap_path=heatmap_path,
    )


def _clamp_score(value: float) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("anomaly score must be finite")
    return min(max(score, 0.0), 1.0)


def _map_regions(raw_regions: list[dict[str, object]]) -> list[DefectRegion]:
    regions: list[DefectRegion] = []
    for raw_region in raw_regions:
        region = DefectRegion.model_validate(raw_region)
        if len(region.bbox) != 4:
            raise ValueError("defect region bbox must contain four coordinates")
        x1, y1, x2, y2 = region.bbox
        if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
            raise ValueError("defect region bbox is invalid")
        regions.append(region)
    return regions


def _safe_relative_heatmap_path(output_path: str, configured_dir: str) -> str:
    heatmap_root = Path(configured_dir).expanduser().resolve()
    if heatmap_root.parent == heatmap_root:
        raise ValueError("heatmap root is not allowed")

    candidate = Path(output_path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    try:
        relative_to_root = candidate.relative_to(heatmap_root)
    except ValueError as exc:
        raise ValueError("heatmap output escaped configured directory") from exc

    try:
        return candidate.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return relative_to_root.as_posix()


def _warning_result(warning: str) -> ImageDefectResult:
    return ImageDefectResult(
        anomaly_score=0.0,
        defect_label=None,
        defect_regions=[],
        heatmap_path=None,
        warnings=[warning],
    )
