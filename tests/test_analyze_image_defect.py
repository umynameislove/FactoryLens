"""Unit tests for the dependency-injected image-defect tool."""

from __future__ import annotations

import sys
from types import ModuleType
from pathlib import Path

import pytest
from PIL import Image

from factorylens.config import Settings
from factorylens.schemas import ImageDefectResult
from factorylens.tools import analyze_image_defect


def _settings(tmp_path: Path, *, threshold: float = 0.5) -> Settings:
    memory_bank_path = tmp_path / "memory_bank.npz"
    memory_bank_path.write_bytes(b"test-memory-bank")
    return Settings(
        database_url="sqlite://",
        vision_memory_bank_path=str(memory_bank_path),
        anomaly_threshold=threshold,
        heatmap_dir=str(tmp_path / "heatmaps"),
        _env_file=None,
    )


def _image(tmp_path: Path) -> Path:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"test-image")
    return image_path


def _heatmapper(
    image_path: str,
    score_map: object,
    out_dir: str,
) -> str:
    del image_path, score_map
    heatmap_path = Path(out_dir) / "sample_heatmap.png"
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path.write_bytes(b"test-heatmap")
    return str(heatmap_path)


def test_high_score_returns_defect_and_contract_valid_regions(
    tmp_path: Path,
) -> None:
    score_map = [[0.0, 0.8], [0.7, 0.1]]

    result = analyze_image_defect(
        str(_image(tmp_path)),
        category="hazelnut",
        settings=_settings(tmp_path),
        scorer=lambda image_path: (0.8, score_map),
        heatmapper=_heatmapper,
        region_extractor=lambda score_map, threshold: [
            {"bbox": [1, 2, 8, 9], "score": 0.9},
            {"bbox": [10, 12, 18, 19], "score": 0.7},
        ],
    )

    validated = ImageDefectResult.model_validate(result.model_dump())
    assert validated.anomaly_score == pytest.approx(0.8)
    assert validated.defect_label == "hazelnut:defect"
    assert len(validated.defect_regions) == 2
    assert all(
        region.bbox[0] < region.bbox[2] and region.bbox[1] < region.bbox[3]
        for region in validated.defect_regions
    )
    assert validated.heatmap_path == "sample_heatmap.png"
    assert validated.warnings == []


def test_low_score_returns_ok(tmp_path: Path) -> None:
    result = analyze_image_defect(
        str(_image(tmp_path)),
        settings=_settings(tmp_path, threshold=0.5),
        scorer=lambda image_path: (0.1, [[0.1]]),
        heatmapper=_heatmapper,
        region_extractor=lambda score_map, threshold: [],
    )

    assert result.anomaly_score == pytest.approx(0.1)
    assert result.defect_label == "ok"
    assert result.defect_regions == []


@pytest.mark.parametrize(
    ("raw_score", "expected"),
    [(-0.25, 0.0), (1.25, 1.0)],
)
def test_anomaly_score_is_clamped(
    tmp_path: Path,
    raw_score: float,
    expected: float,
) -> None:
    result = analyze_image_defect(
        str(_image(tmp_path)),
        settings=_settings(tmp_path),
        scorer=lambda image_path: (raw_score, [[raw_score]]),
        heatmapper=_heatmapper,
        region_extractor=lambda score_map, threshold: [],
    )

    assert result.anomaly_score == expected


def test_default_adapters_use_configured_bank_and_pixel_image_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "real.png"
    Image.new("RGB", (8, 6), color=(120, 80, 40)).save(image_path)
    settings = _settings(tmp_path)
    calls: dict[str, object] = {}
    score_map = object()

    vision_package = ModuleType("factorylens.vision")
    vision_package.__path__ = []  # type: ignore[attr-defined]
    anomaly_module = ModuleType("factorylens.vision.anomaly")
    heatmap_module = ModuleType("factorylens.vision.heatmap")

    def fake_score_image(
        path: str,
        *,
        memory_bank_path: str,
    ) -> tuple[float, object]:
        calls["scorer"] = (path, memory_bank_path)
        return 0.8, score_map

    def fake_make_heatmap(
        path: str,
        received_score_map: object,
        out_dir: str,
    ) -> str:
        calls["heatmapper"] = (path, received_score_map, out_dir)
        output_path = Path(out_dir) / "default_heatmap.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"heatmap")
        return str(output_path)

    def fake_extract_regions(
        received_score_map: object,
        threshold: float,
        min_area: int = 1,
        image_shape: tuple[int, int] | None = None,
    ) -> list[dict[str, object]]:
        calls["extractor"] = (
            received_score_map,
            threshold,
            min_area,
            image_shape,
        )
        return [{"bbox": [0, 0, 4, 3], "score": 0.8}]

    anomaly_module.score_image = fake_score_image  # type: ignore[attr-defined]
    heatmap_module.make_heatmap = fake_make_heatmap  # type: ignore[attr-defined]
    heatmap_module.extract_regions = fake_extract_regions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "factorylens.vision", vision_package)
    monkeypatch.setitem(
        sys.modules,
        "factorylens.vision.anomaly",
        anomaly_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "factorylens.vision.heatmap",
        heatmap_module,
    )

    result = analyze_image_defect(
        str(image_path),
        settings=settings,
    )

    scorer_path, bank_path = calls["scorer"]
    assert Path(scorer_path) == image_path.resolve()
    assert Path(bank_path) == Path(settings.vision_memory_bank_path).resolve()
    assert calls["heatmapper"] == (
        str(image_path.resolve()),
        score_map,
        settings.heatmap_dir,
    )
    assert calls["extractor"] == (
        score_map,
        settings.anomaly_threshold,
        1,
        (6, 8),
    )
    assert result.defect_regions[0].bbox == [0, 0, 4, 3]
    assert result.heatmap_path == "default_heatmap.png"


def test_missing_image_returns_safe_warning(tmp_path: Path) -> None:
    result = analyze_image_defect(
        str(tmp_path / "missing.png"),
        settings=_settings(tmp_path),
    )

    assert ImageDefectResult.model_validate(result.model_dump()) == result
    assert result.anomaly_score == 0.0
    assert result.defect_label is None
    assert result.heatmap_path is None
    assert result.warnings == ["image not found/unreadable"]


def test_missing_memory_bank_returns_safe_warning(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    Path(settings.vision_memory_bank_path).unlink()

    result = analyze_image_defect(
        str(_image(tmp_path)),
        settings=settings,
    )

    assert result.warnings == ["anomaly memory bank not available; run the build step"]
    assert result.defect_regions == []


def test_heatmap_path_cannot_escape_configured_directory(
    tmp_path: Path,
) -> None:
    private_path = tmp_path / "outside" / "leaked.png"
    private_path.parent.mkdir()
    private_path.write_bytes(b"private")

    result = analyze_image_defect(
        str(_image(tmp_path)),
        settings=_settings(tmp_path),
        scorer=lambda image_path: (0.8, [[0.8]]),
        heatmapper=lambda image_path, score_map, out_dir: str(private_path),
        region_extractor=lambda score_map, threshold: [],
    )

    assert result.heatmap_path is None
    assert result.warnings == ["image analysis failed"]
    assert str(tmp_path) not in " ".join(result.warnings)


def test_vision_exception_does_not_leak_machine_path(tmp_path: Path) -> None:
    image_path = _image(tmp_path)

    def failing_scorer(path: str) -> tuple[float, object]:
        raise RuntimeError(f"could not read {path}")

    result = analyze_image_defect(
        str(image_path),
        settings=_settings(tmp_path),
        scorer=failing_scorer,
        heatmapper=_heatmapper,
        region_extractor=lambda score_map, threshold: [],
    )

    assert result.warnings == ["image analysis failed"]
    assert str(tmp_path) not in " ".join(result.warnings)
