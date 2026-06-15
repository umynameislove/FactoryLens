from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image


def load_build_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_memory_bank.py"
    spec = importlib.util.spec_from_file_location("build_memory_bank_script", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeExtractor:
    def extract_patch_embeddings(self, image_path: str) -> np.ndarray:
        image_index = int(Path(image_path).stem.removeprefix("good_"))
        return np.array(
            [
                [1.0 + image_index, 0.0],
                [0.0, 1.0 + image_index],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )


def test_collect_training_images_keeps_supported_images_sorted(tmp_path):
    script = load_build_script()
    train_dir = tmp_path / "train" / "good"
    train_dir.mkdir(parents=True)
    (train_dir / "b.png").write_bytes(b"fake")
    (train_dir / "a.jpg").write_bytes(b"fake")
    (train_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    image_paths = script.collect_training_images(train_dir)

    assert [path.name for path in image_paths] == ["a.jpg", "b.png"]


def test_build_memory_bank_from_small_fake_images(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    script = load_build_script()
    train_dir = tmp_path / "mvtec" / "hazelnut" / "train" / "good"
    train_dir.mkdir(parents=True)
    for index in range(3):
        pixels = np.full((8, 8, 3), index * 40, dtype=np.uint8)
        Image.fromarray(pixels).save(train_dir / f"good_{index}.png")

    out_path = tmp_path / "memory_bank.npz"
    result = script.build_memory_bank_from_dir(
        train_dir=train_dir,
        out=out_path,
        extractor=FakeExtractor(),
        max_patches_per_image=2,
        seed=1,
    )

    saved = np.load(out_path)

    assert result.image_count == 3
    assert result.output_path == out_path
    assert result.memory_bank_shape == (6, 2)
    assert saved["memory_bank"].shape == (6, 2)
    assert saved["distance_scale"].shape == (1,)
