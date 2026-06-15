from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("torchvision")

from factorylens.vision.defect_type import (  # noqa: E402
    DEFECT_TYPES,
    build_defect_type_prototypes,
    classify_defect_type,
    extract_image_embedding,
)


class FakeExtractor:
    def __init__(self) -> None:
        self.embeddings = {
            "crack-a.png": _patches([1.0, 0.0, 0.0, 0.0]),
            "crack-b.png": _patches([0.9, 0.1, 0.0, 0.0]),
            "cut-a.png": _patches([0.0, 1.0, 0.0, 0.0]),
            "cut-b.png": _patches([0.1, 0.9, 0.0, 0.0]),
            "hole-a.png": _patches([0.0, 0.0, 1.0, 0.0]),
            "hole-b.png": _patches([0.0, 0.1, 0.9, 0.0]),
            "print-a.png": _patches([0.0, 0.0, 0.0, 1.0]),
            "print-b.png": _patches([0.1, 0.0, 0.0, 0.9]),
            "query-crack.png": _patches([0.95, 0.05, 0.0, 0.0]),
        }

    def extract_patch_embeddings(self, image_path: str) -> np.ndarray:
        return self.embeddings[image_path]


def test_classify_defect_type_returns_nearest_label():
    extractor = FakeExtractor()
    result = classify_defect_type(
        "query-crack.png",
        _reference_paths(),
        extractor=extractor,
    )

    assert result.label == "crack"
    assert set(result.distances) == set(DEFECT_TYPES)
    assert result.distances["crack"] < result.distances["cut"]
    assert 0.3 < result.confidence < 0.5


def test_classify_defect_type_accepts_prebuilt_prototypes():
    extractor = FakeExtractor()
    prototypes = build_defect_type_prototypes(
        _reference_paths(),
        extractor=extractor,
    )

    result = classify_defect_type(
        "query-crack.png",
        prototypes=prototypes,
        extractor=extractor,
    )

    assert result.label == "crack"
    assert [prototype.source_count for prototype in prototypes] == [2, 2, 2, 2]
    assert all(prototype.intra_class_radius for prototype in prototypes)
    assert result.confidence < 1.0


def test_single_reference_exact_match_stays_conservative():
    extractor = FakeExtractor()
    result = classify_defect_type(
        "crack-a.png",
        {
            "crack": ["crack-a.png"],
            "cut": ["cut-a.png"],
            "hole": ["hole-a.png"],
            "print": ["print-a.png"],
        },
        extractor=extractor,
    )

    assert result.label == "crack"
    assert result.confidence < 0.3


def test_build_defect_type_prototypes_requires_all_labels():
    extractor = FakeExtractor()

    with pytest.raises(ValueError, match="missing=print"):
        build_defect_type_prototypes(
            {
                "crack": ["crack-a.png"],
                "cut": ["cut-a.png"],
                "hole": ["hole-a.png"],
            },
            extractor=extractor,
        )


def test_extract_image_embedding_rejects_empty_patch_map():
    class EmptyExtractor:
        def extract_patch_embeddings(self, image_path: str) -> np.ndarray:
            return np.empty((0, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="N > 0"):
        extract_image_embedding("empty.png", EmptyExtractor())


def _reference_paths() -> dict[str, list[str]]:
    return {
        "crack": ["crack-a.png", "crack-b.png"],
        "cut": ["cut-a.png", "cut-b.png"],
        "hole": ["hole-a.png", "hole-b.png"],
        "print": ["print-a.png", "print-b.png"],
    }


def _patches(center: list[float]) -> np.ndarray:
    center_array = np.array(center, dtype=np.float32)
    return np.vstack(
        [
            center_array,
            center_array + np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float32),
        ]
    )
