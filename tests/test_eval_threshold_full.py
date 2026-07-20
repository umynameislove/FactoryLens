from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_threshold_full.py"
    spec = importlib.util.spec_from_file_location("eval_threshold_full_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _samples(script: ModuleType):
    return [
        script.SampleScore(Path("good-1.png"), "good", False, 0.10),
        script.SampleScore(Path("good-2.png"), "good", False, 0.20),
        script.SampleScore(Path("bad-1.png"), "crack", True, 0.80),
        script.SampleScore(Path("bad-2.png"), "cut", True, 0.90),
    ]


def test_calibration_candidates_stay_within_observed_score_range() -> None:
    script = load_script()
    samples = _samples(script)

    selected = script.choose_youden(samples)

    assert 0.10 <= selected.threshold <= 0.90
    assert selected.false_positive == 0
    assert selected.false_negative == 0
    assert script.compute_auroc(samples) == pytest.approx(1.0)


def test_leakage_check_rejects_matching_train_and_test_bytes(tmp_path: Path) -> None:
    script = load_script()
    train_dir = tmp_path / "train" / "good"
    train_dir.mkdir(parents=True)
    for index in range(391):
        (train_dir / f"{index:03d}.png").write_bytes(f"image-{index}".encode())
    test_path = tmp_path / "test.png"
    test_path.write_bytes(b"image-0")
    samples = [script.SampleScore(test_path, "good", False, 0.1)]

    with pytest.raises(ValueError, match="leakage"):
        script.verify_no_train_test_leakage(train_dir, samples)
