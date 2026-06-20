from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "robustness_study.py"
)


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "robustness_study_script",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_image() -> np.ndarray:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(32, dtype=np.uint8)
    image[:, :, 1] = 100
    image[:, :, 2] = 220
    return image


def test_zero_level_perturbations_are_exact_noops() -> None:
    script = load_script()
    image = sample_image()
    rng = np.random.default_rng(13)

    outputs = [
        script.rotate_image(image, 0),
        script.adjust_brightness(image, 0),
        script.adjust_contrast(image, 0),
        script.gaussian_blur(image, 0),
        script.gaussian_noise(image, 0, rng),
        script.jpeg_compress(image, None),
    ]

    assert all(np.array_equal(output, image) for output in outputs)
    assert all(output is not image for output in outputs)


@pytest.mark.parametrize(
    "transform",
    [
        lambda script, image: script.rotate_image(image, 15),
        lambda script, image: script.adjust_brightness(image, 0.3),
        lambda script, image: script.adjust_contrast(image, -0.3),
        lambda script, image: script.gaussian_blur(image, 2.0),
        lambda script, image: script.gaussian_noise(
            image,
            0.05,
            np.random.default_rng(7),
        ),
        lambda script, image: script.jpeg_compress(image, 70),
    ],
)
def test_perturbations_preserve_shape_and_dtype(transform) -> None:
    script = load_script()
    image = sample_image()

    output = transform(script, image)

    assert output.shape == image.shape
    assert output.dtype == np.uint8


def test_gaussian_noise_is_reproducible_for_same_seed() -> None:
    script = load_script()
    image = sample_image()

    first = script.gaussian_noise(image, 0.1, np.random.default_rng(42))
    second = script.gaussian_noise(image, 0.1, np.random.default_rng(42))

    assert np.array_equal(first, second)
    assert not np.array_equal(first, image)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda script, image: script.gaussian_blur(image, -1), "sigma"),
        (
            lambda script, image: script.gaussian_noise(
                image,
                -0.1,
                np.random.default_rng(1),
            ),
            "standard deviation",
        ),
        (lambda script, image: script.jpeg_compress(image, 0), "quality"),
        (lambda script, image: script.adjust_brightness(image, 1.1), "brightness"),
    ],
)
def test_invalid_perturbation_levels_are_rejected(call, message: str) -> None:
    script = load_script()

    with pytest.raises(ValueError, match=message):
        call(script, sample_image())


def test_evaluate_robustness_measures_score_drift_and_flip(
    tmp_path: Path,
) -> None:
    script = load_script()
    image_path = tmp_path / "good.png"
    cv2.imwrite(str(image_path), sample_image())
    samples = [script.SampleImage(image_path, "good")]
    spec = script.PerturbationSpec(
        name="brightness",
        levels=(0.0, 0.5),
        transform=lambda image, level, rng: script.adjust_brightness(image, level),
        level_label=lambda level: str(level),
    )

    def scorer(path: str) -> float:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        assert image is not None
        return 0.2 if float(image.mean()) < 150 else 0.8

    rows = script.evaluate_robustness(
        samples,
        scorer=scorer,
        threshold=0.5,
        seed=13,
        specs=[spec],
    )

    assert len(rows) == 2
    assert rows[0].absolute_drift == 0
    assert rows[1].score_drift == pytest.approx(0.6)
    assert rows[1].verdict_flipped is True
    aggregate = script.aggregate_rows(rows)
    assert aggregate[1].flip_rate == pytest.approx(1.0)


def test_write_outputs_creates_report_csv_and_small_plots(tmp_path: Path) -> None:
    script = load_script()
    rows = [
        script.RobustnessRow(
            image_path="test/good/000.png",
            label="good",
            perturbation="rotation",
            level="0",
            level_order=0,
            baseline_score=0.2,
            perturbed_score=0.2,
            score_drift=0.0,
            absolute_drift=0.0,
            baseline_defect=False,
            perturbed_defect=False,
            verdict_flipped=False,
        ),
        script.RobustnessRow(
            image_path="test/good/000.png",
            label="good",
            perturbation="rotation",
            level="+15 deg",
            level_order=1,
            baseline_score=0.2,
            perturbed_score=0.7,
            score_drift=0.5,
            absolute_drift=0.5,
            baseline_defect=False,
            perturbed_defect=True,
            verdict_flipped=True,
        ),
    ]
    provenance = script.StudyProvenance(
        dataset_name="MVTec/hazelnut",
        selected_images_sha256="a" * 64,
        memory_bank_name="memory_bank_eval.npz",
        memory_bank_sha256="b" * 64,
        threshold=0.3884,
        image_size=512,
        seed=13,
        selected_images=("test/good/000.png",),
    )

    result = script.write_outputs(
        rows,
        out_dir=tmp_path / "out",
        provenance=provenance,
    )

    report = Path(result["report_path"])
    assert report.is_file()
    assert "Highest observed flip rate" in report.read_text(encoding="utf-8")
    assert Path(result["metrics_csv"]).is_file()
    for plot in result["plots"]:
        assert Path(plot).is_file()
        assert Path(plot).stat().st_size < 150 * 1024
