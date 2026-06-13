from __future__ import annotations

import numpy as np

from factorylens.vision.anomaly import build_memory_bank, load_memory_bank, score_image


class FakeExtractor:
    def __init__(self):
        self.patch_embeddings = {
            "good-a.png": np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "good-b.png": np.array(
                [
                    [1.0, 0.1],
                    [0.1, 1.0],
                    [0.9, 1.0],
                ],
                dtype=np.float32,
            ),
        }
        self.feature_maps = {
            "test.png": np.array(
                [
                    [[1.0, 0.0], [4.0, 1.0]],
                    [[0.0, 1.0], [4.0, 1.0]],
                ],
                dtype=np.float32,
            )
        }

    def extract_patch_embeddings(self, image_path: str) -> np.ndarray:
        return self.patch_embeddings[image_path]

    def extract_feature_map(self, image_path: str) -> np.ndarray:
        return self.feature_maps[image_path]


def test_build_memory_bank_saves_npz(tmp_path):
    out_path = tmp_path / "memory_bank.npz"
    extractor = FakeExtractor()

    memory_bank = build_memory_bank(
        ["good-a.png", "good-b.png"],
        out_path=str(out_path),
        extractor=extractor,
        max_patches_per_image=2,
        seed=1,
    )
    loaded_bank, distance_scale = load_memory_bank(str(out_path))

    assert memory_bank.shape == (4, 2)
    assert loaded_bank.shape == (4, 2)
    assert distance_scale > 0


def test_score_image_returns_normalized_score_map(tmp_path):
    out_path = tmp_path / "memory_bank.npz"
    extractor = FakeExtractor()
    memory_bank = build_memory_bank(
        ["good-a.png", "good-b.png"],
        out_path=str(out_path),
        extractor=extractor,
        max_patches_per_image=3,
        seed=1,
    )

    anomaly_score, score_map = score_image(
        "test.png",
        memory_bank=memory_bank,
        extractor=extractor,
        distance_scale=0.5,
    )

    assert isinstance(anomaly_score, float)
    assert 0.0 <= anomaly_score <= 1.0
    assert score_map.shape == (2, 2)
    assert np.all(score_map >= 0.0)
    assert np.all(score_map <= 1.0)
    assert score_map[1, 0] > score_map[0, 0]
