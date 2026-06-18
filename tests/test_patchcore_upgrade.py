from __future__ import annotations

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from factorylens.vision.anomaly import (  # noqa: E402
    build_memory_bank,
    greedy_coreset,
    load_memory_bank,
    save_memory_bank,
    score_image,
)
from factorylens.vision.embeddings import PatchEmbeddingExtractor  # noqa: E402


def test_multilayer_extractor_fuses_layer2_and_layer3(tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(120, 80, 40)).save(image_path)
    extractor = PatchEmbeddingExtractor(
        image_size=64,
        pretrained=False,
        device="cpu",
    )

    feature_map = extractor.extract_feature_map(str(image_path))
    embeddings = extractor.extract_patch_embeddings(str(image_path))

    assert feature_map.shape == (384, 4, 4)
    assert embeddings.shape == (16, 384)
    assert np.linalg.norm(embeddings, axis=1) == pytest.approx(
        np.ones(16),
        abs=1e-5,
    )


def test_greedy_coreset_is_deterministic_and_reduces_rows() -> None:
    rng = np.random.default_rng(5)
    clusters = np.vstack(
        [
            rng.normal((1.0, 0.0, 0.0), 0.02, size=(100, 3)),
            rng.normal((0.0, 1.0, 0.0), 0.02, size=(100, 3)),
            rng.normal((0.0, 0.0, 1.0), 0.02, size=(100, 3)),
        ]
    ).astype(np.float32)

    first = greedy_coreset(clusters, 30, seed=13, projection_dim=3)
    second = greedy_coreset(clusters, 30, seed=13, projection_dim=3)

    assert first.shape == (30, 3)
    assert np.array_equal(first, second)
    nearest_centroid = np.argmax(first, axis=1)
    assert set(nearest_centroid) == {0, 1, 2}


def test_memory_bank_npz_contract_is_unchanged(tmp_path) -> None:
    path = tmp_path / "memory_bank.npz"
    memory_bank = np.eye(4, dtype=np.float32)

    save_memory_bank(str(path), memory_bank, 0.25)
    with np.load(path) as data:
        assert set(data.files) == {"memory_bank", "distance_scale"}
    loaded, distance_scale = load_memory_bank(str(path))

    assert np.array_equal(loaded, memory_bank)
    assert distance_scale == pytest.approx(0.25)


def test_public_vision_signatures_remain_compatible() -> None:
    assert list(inspect.signature(build_memory_bank).parameters) == [
        "good_image_paths",
        "out_path",
        "extractor",
        "max_patches_per_image",
        "seed",
    ]
    assert list(inspect.signature(score_image).parameters) == [
        "image_path",
        "memory_bank",
        "memory_bank_path",
        "extractor",
        "distance_scale",
        "anomaly_percentile",
    ]
    assert list(inspect.signature(save_memory_bank).parameters) == [
        "path",
        "memory_bank",
        "distance_scale",
    ]
    assert list(inspect.signature(load_memory_bank).parameters) == ["path"]


def test_score_image_rejects_stale_single_layer_bank() -> None:
    class FusedExtractor:
        def extract_feature_map(self, image_path: str) -> np.ndarray:
            return np.ones((4, 2, 2), dtype=np.float32)

    stale_bank = np.ones((8, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="rebuild"):
        score_image(
            "sample.png",
            memory_bank=stale_bank,
            extractor=FusedExtractor(),
        )
