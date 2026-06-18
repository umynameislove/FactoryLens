"""Measure anomaly-detector robustness under realistic image perturbations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFECT_LABELS = ("crack", "cut", "hole", "print")
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/mvtec/hazelnut"
DEFAULT_MEMORY_BANK = PROJECT_ROOT / "data/memory_bank_eval.npz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "assets/eval_samples"
DEFAULT_THRESHOLD = 0.3884
PLOT_COLORS = {
    "mean score": (40, 105, 205),
    "mean absolute drift": (55, 155, 75),
    "flip rate": (190, 75, 45),
}

ScoreFunction = Callable[[str], float]
TransformFunction = Callable[[np.ndarray, object, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class SampleImage:
    path: Path
    label: str

    @property
    def is_defect(self) -> bool:
        return self.label != "good"


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    levels: tuple[object, ...]
    transform: TransformFunction
    level_label: Callable[[object], str]


@dataclass(frozen=True)
class RobustnessRow:
    image_path: str
    label: str
    perturbation: str
    level: str
    level_order: int
    baseline_score: float
    perturbed_score: float
    score_drift: float
    absolute_drift: float
    baseline_defect: bool
    perturbed_defect: bool
    verdict_flipped: bool


@dataclass(frozen=True)
class AggregateRow:
    perturbation: str
    level: str
    level_order: int
    sample_count: int
    mean_score: float
    mean_score_drift: float
    mean_absolute_drift: float
    max_absolute_drift: float
    flip_rate: float


@dataclass(frozen=True)
class StudyProvenance:
    dataset_name: str
    selected_images_sha256: str
    memory_bank_name: str
    memory_bank_sha256: str
    threshold: float
    image_size: int
    seed: int
    selected_images: tuple[str, ...]
    model_checkpoint_name: str | None = None
    model_checkpoint_sha256: str | None = None


def perturbation_specs() -> tuple[PerturbationSpec, ...]:
    """Return the fixed B18 experiment matrix, including a true zero level."""

    return (
        PerturbationSpec(
            "rotation",
            (0, -15, -10, -5, 5, 10, 15),
            lambda image, level, rng: rotate_image(image, float(level)),
            lambda level: f"{int(level):+d} deg" if level else "0",
        ),
        PerturbationSpec(
            "brightness",
            (0.0, -0.30, -0.15, 0.15, 0.30),
            lambda image, level, rng: adjust_brightness(image, float(level)),
            lambda level: f"{float(level):+.0%}" if level else "0",
        ),
        PerturbationSpec(
            "contrast",
            (0.0, -0.30, -0.15, 0.15, 0.30),
            lambda image, level, rng: adjust_contrast(image, float(level)),
            lambda level: f"{float(level):+.0%}" if level else "0",
        ),
        PerturbationSpec(
            "gaussian_blur",
            (0.0, 1.0, 2.0, 3.0),
            lambda image, level, rng: gaussian_blur(image, float(level)),
            lambda level: f"sigma {float(level):.1f}" if level else "0",
        ),
        PerturbationSpec(
            "gaussian_noise",
            (0.0, 0.02, 0.05, 0.10),
            lambda image, level, rng: gaussian_noise(
                image,
                float(level),
                rng,
            ),
            lambda level: f"std {float(level):.0%}" if level else "0",
        ),
        PerturbationSpec(
            "jpeg",
            (None, 90, 70, 50),
            lambda image, level, rng: jpeg_compress(
                image,
                None if level is None else int(level),
            ),
            lambda level: "0" if level is None else f"quality {int(level)}",
        ),
    )


def collect_samples(
    dataset_root: str | Path,
    *,
    good_count: int,
    defect_per_type: int,
) -> list[SampleImage]:
    """Select a deterministic good+defect subset from MVTec hazelnut."""

    if good_count <= 0 or defect_per_type <= 0:
        raise ValueError("sample counts must be positive")
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    samples: list[SampleImage] = []
    label_counts = {"good": good_count, **dict.fromkeys(DEFECT_LABELS, defect_per_type)}
    for label, count in label_counts.items():
        paths = sorted((root / "test" / label).glob("*.png"))
        if len(paths) < count:
            raise ValueError(
                f"Need {count} test/{label} images under {root}; found {len(paths)}"
            )
        samples.extend(SampleImage(path=path, label=label) for path in paths[:count])
    return samples


def rotate_image(image: np.ndarray, degrees: float) -> np.ndarray:
    _validate_image(image)
    if not math.isfinite(degrees):
        raise ValueError("rotation degrees must be finite")
    if degrees == 0:
        return image.copy()
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def adjust_brightness(image: np.ndarray, delta: float) -> np.ndarray:
    _validate_image(image)
    if not math.isfinite(delta) or not -1.0 <= delta <= 1.0:
        raise ValueError("brightness delta must be finite and in [-1, 1]")
    if delta == 0:
        return image.copy()
    return np.clip(image.astype(np.float32) + delta * 255.0, 0, 255).astype(
        np.uint8
    )


def adjust_contrast(image: np.ndarray, delta: float) -> np.ndarray:
    _validate_image(image)
    if not math.isfinite(delta) or not -1.0 <= delta <= 1.0:
        raise ValueError("contrast delta must be finite and in [-1, 1]")
    if delta == 0:
        return image.copy()
    factor = 1.0 + delta
    mean = image.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(
        np.uint8
    )


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    _validate_image(image)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("blur sigma must be finite and non-negative")
    if sigma == 0:
        return image.copy()
    radius = max(1, math.ceil(3.0 * sigma))
    kernel_size = 2 * radius + 1
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )


def gaussian_noise(
    image: np.ndarray,
    standard_deviation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    _validate_image(image)
    if (
        not math.isfinite(standard_deviation)
        or not 0.0 <= standard_deviation <= 1.0
    ):
        raise ValueError("noise standard deviation must be finite and in [0, 1]")
    if standard_deviation == 0:
        return image.copy()
    noise = rng.normal(
        0.0,
        standard_deviation * 255.0,
        size=image.shape,
    )
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg_compress(image: np.ndarray, quality: int | None) -> np.ndarray:
    _validate_image(image)
    if quality is None:
        return image.copy()
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        raise OSError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise OSError("JPEG decoding failed")
    return decoded


def evaluate_robustness(
    samples: Sequence[SampleImage],
    *,
    scorer: ScoreFunction,
    threshold: float,
    seed: int,
    specs: Sequence[PerturbationSpec] | None = None,
) -> list[RobustnessRow]:
    """Score baseline and perturbed images, returning per-image measurements."""

    if not samples:
        raise ValueError("samples must not be empty")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    resolved_specs = tuple(specs or perturbation_specs())
    if not resolved_specs:
        raise ValueError("perturbation specs must not be empty")

    baseline_scores = {sample.path: _validated_score(scorer(str(sample.path))) for sample in samples}
    rows: list[RobustnessRow] = []
    with tempfile.TemporaryDirectory(prefix="factorylens-b18-") as temporary_dir:
        temp_root = Path(temporary_dir)
        for sample_index, sample in enumerate(samples):
            image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {sample.path}")
            baseline_score = baseline_scores[sample.path]
            baseline_defect = baseline_score >= threshold
            for spec_index, spec in enumerate(resolved_specs):
                for level_order, level in enumerate(spec.levels):
                    if level_order == 0:
                        perturbed_score = baseline_score
                    else:
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [seed, sample_index, spec_index, level_order]
                            )
                        )
                        perturbed = spec.transform(image, level, rng)
                        _validate_transformed_image(image, perturbed)
                        output_path = (
                            temp_root
                            / f"{sample_index:03d}-{spec.name}-{level_order}.png"
                        )
                        if not cv2.imwrite(str(output_path), perturbed):
                            raise OSError(f"Could not write perturbed image: {output_path}")
                        perturbed_score = _validated_score(scorer(str(output_path)))

                    perturbed_defect = perturbed_score >= threshold
                    drift = perturbed_score - baseline_score
                    rows.append(
                        RobustnessRow(
                            image_path=_sample_display_path(sample),
                            label=sample.label,
                            perturbation=spec.name,
                            level=spec.level_label(level),
                            level_order=level_order,
                            baseline_score=baseline_score,
                            perturbed_score=perturbed_score,
                            score_drift=drift,
                            absolute_drift=abs(drift),
                            baseline_defect=baseline_defect,
                            perturbed_defect=perturbed_defect,
                            verdict_flipped=baseline_defect != perturbed_defect,
                        )
                    )
    return rows


def aggregate_rows(rows: Sequence[RobustnessRow]) -> list[AggregateRow]:
    """Aggregate score drift and verdict flips per perturbation level."""

    if not rows:
        raise ValueError("rows must not be empty")
    groups: dict[tuple[str, str, int], list[RobustnessRow]] = {}
    for row in rows:
        groups.setdefault(
            (row.perturbation, row.level, row.level_order),
            [],
        ).append(row)

    aggregates: list[AggregateRow] = []
    for (name, level, level_order), group in groups.items():
        aggregates.append(
            AggregateRow(
                perturbation=name,
                level=level,
                level_order=level_order,
                sample_count=len(group),
                mean_score=_mean(row.perturbed_score for row in group),
                mean_score_drift=_mean(row.score_drift for row in group),
                mean_absolute_drift=_mean(row.absolute_drift for row in group),
                max_absolute_drift=max(row.absolute_drift for row in group),
                flip_rate=_mean(float(row.verdict_flipped) for row in group),
            )
        )
    return sorted(
        aggregates,
        key=lambda row: (row.perturbation, row.level_order),
    )


def write_outputs(
    rows: Sequence[RobustnessRow],
    *,
    out_dir: str | Path,
    provenance: StudyProvenance,
) -> dict[str, object]:
    """Write the measured CSV, report, and one plot per perturbation."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_rows(rows)
    csv_path = output_dir / "robustness_metrics.csv"
    _write_metrics_csv(rows, csv_path)

    plot_paths: list[Path] = []
    for perturbation in sorted({row.perturbation for row in aggregates}):
        plot_path = output_dir / f"robustness_{perturbation}.png"
        _render_perturbation_plot(
            [row for row in aggregates if row.perturbation == perturbation],
            plot_path,
        )
        plot_paths.append(plot_path)

    report_path = output_dir / "ROBUSTNESS.md"
    report_path.write_text(
        _build_report(
            rows=rows,
            aggregates=aggregates,
            provenance=provenance,
            csv_name=csv_path.name,
            plot_paths=plot_paths,
        ),
        encoding="utf-8",
    )
    return {
        "report_path": report_path,
        "metrics_csv": csv_path,
        "plots": plot_paths,
        "aggregates": aggregates,
    }


def build_provenance(
    samples: Sequence[SampleImage],
    *,
    dataset_root: str | Path,
    memory_bank_path: str | Path,
    threshold: float,
    image_size: int,
    seed: int,
    torch_cache: str | Path | None = None,
) -> StudyProvenance:
    dataset_path = Path(dataset_root).expanduser().resolve()
    bank_path = Path(memory_bank_path).expanduser().resolve()
    if not bank_path.is_file():
        raise FileNotFoundError(f"Memory bank does not exist: {bank_path}")
    selected = tuple(_sample_display_path(sample) for sample in samples)
    image_digest = hashlib.sha256()
    for sample, display_path in zip(samples, selected, strict=True):
        image_digest.update(display_path.encode("utf-8"))
        image_digest.update(_sha256(sample.path).encode("ascii"))
    checkpoint = _resnet_checkpoint(torch_cache)
    return StudyProvenance(
        dataset_name=f"MVTec/{dataset_path.name}",
        selected_images_sha256=image_digest.hexdigest(),
        memory_bank_name=bank_path.name,
        memory_bank_sha256=_sha256(bank_path),
        threshold=threshold,
        image_size=image_size,
        seed=seed,
        selected_images=selected,
        model_checkpoint_name=checkpoint.name if checkpoint else None,
        model_checkpoint_sha256=_sha256(checkpoint) if checkpoint else None,
    )


def build_real_scorer(
    *,
    memory_bank_path: str | Path,
    image_size: int,
) -> ScoreFunction:
    """Load the real B10 anomaly detector once and return a reusable scorer."""

    from factorylens.vision.anomaly import load_memory_bank, score_image
    from factorylens.vision.embeddings import PatchEmbeddingExtractor

    memory_bank, distance_scale = load_memory_bank(str(memory_bank_path))
    extractor = PatchEmbeddingExtractor(
        image_size=image_size,
        pretrained=True,
        allow_untrained_fallback=False,
    )

    def scorer(image_path: str) -> float:
        score, _ = score_image(
            image_path,
            memory_bank=memory_bank,
            extractor=extractor,
            distance_scale=distance_scale,
        )
        return score

    return scorer


def _build_report(
    *,
    rows: Sequence[RobustnessRow],
    aggregates: Sequence[AggregateRow],
    provenance: StudyProvenance,
    csv_name: str,
    plot_paths: Sequence[Path],
) -> str:
    nonzero = [row for row in aggregates if row.level_order > 0]
    most_flips = max(
        nonzero,
        key=lambda row: (row.flip_rate, row.mean_absolute_drift),
    )
    most_drift = max(
        nonzero,
        key=lambda row: (row.mean_absolute_drift, row.flip_rate),
    )
    sample_count = len({row.image_path for row in rows})
    baseline_rows = [
        row
        for row in rows
        if row.perturbation == "rotation" and row.level_order == 0
    ]
    baseline_accuracy = _mean(
        float(row.baseline_defect == (row.label != "good"))
        for row in baseline_rows
    )
    lines = [
        "# Hazelnut Robustness Study",
        "",
        "## Reproduction",
        "",
        f"- Dataset: `{provenance.dataset_name}`",
        f"- Selected images: {sample_count}",
        f"- Selected-image content fingerprint: `{provenance.selected_images_sha256}`",
        f"- Memory bank: `{provenance.memory_bank_name}`",
        f"- Memory-bank SHA-256: `{provenance.memory_bank_sha256}`",
        (
            f"- Model checkpoint: `{provenance.model_checkpoint_name}` "
            f"(SHA-256 `{provenance.model_checkpoint_sha256}`)"
            if provenance.model_checkpoint_name
            else "- Model checkpoint: resolved by TorchVision cache; hash not recorded"
        ),
        f"- Image size: `{provenance.image_size}`",
        f"- Verdict threshold: `{provenance.threshold:.4f}` (`score >= threshold` means defect)",
        f"- Gaussian-noise seed: `{provenance.seed}`",
        f"- Baseline accuracy on selected subset: `{baseline_accuracy:.3f}`",
        f"- Raw measurements: [`{csv_name}`]({csv_name})",
        (
            "- Verify command: `python scripts/robustness_study.py "
            "--dataset-root ../FactoryLens/data/mvtec/hazelnut "
            "--memory-bank ../FactoryLens/data/memory_bank_eval.npz "
            "--torch-cache ../FactoryLens/data/torch --good-count 2 "
            "--defect-per-type 1 --image-size 512 --threshold 0.3884 --seed 13`"
        ),
        "- Test command: `python -m pytest -q tests/test_robustness_study.py`",
        "",
        "Selected images:",
        "",
        *[f"- `{path}`" for path in provenance.selected_images],
        "",
        "## Aggregate Results",
        "",
        "| Perturbation | Level | Mean score | Mean drift | Mean abs drift | Max abs drift | Flip rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row.perturbation} | {row.level} | {row.mean_score:.4f} | "
            f"{row.mean_score_drift:+.4f} | {row.mean_absolute_drift:.4f} | "
            f"{row.max_absolute_drift:.4f} | {row.flip_rate:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
        ]
    )
    for plot_path in plot_paths:
        title = plot_path.stem.removeprefix("robustness_").replace("_", " ").title()
        lines.extend([f"### {title}", "", f"![{title}]({plot_path.name})", ""])
    lines.extend(
        [
            "## Engineering Conclusion",
            "",
            (
                f"- Highest observed flip rate: `{most_flips.perturbation}` at "
                f"`{most_flips.level}` ({most_flips.flip_rate:.1%})."
            ),
            (
                f"- Largest mean absolute score drift: `{most_drift.perturbation}` "
                f"at `{most_drift.level}` ({most_drift.mean_absolute_drift:.4f})."
            ),
            "- Treat scores close to the threshold as unstable and require human review.",
            "- Standardize camera angle, exposure, focus, and image encoding before inference.",
            "- Do not change the detector or threshold from this study alone; rerun on a larger production-representative set first.",
            "",
            "This experiment measures robustness only. It does not train or modify the model.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_metrics_csv(rows: Sequence[RobustnessRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "label",
                "perturbation",
                "level",
                "level_order",
                "baseline_score",
                "perturbed_score",
                "score_drift",
                "absolute_drift",
                "baseline_verdict",
                "perturbed_verdict",
                "verdict_flipped",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row.image_path,
                    "label": row.label,
                    "perturbation": row.perturbation,
                    "level": row.level,
                    "level_order": row.level_order,
                    "baseline_score": f"{row.baseline_score:.6f}",
                    "perturbed_score": f"{row.perturbed_score:.6f}",
                    "score_drift": f"{row.score_drift:+.6f}",
                    "absolute_drift": f"{row.absolute_drift:.6f}",
                    "baseline_verdict": _verdict(row.baseline_defect),
                    "perturbed_verdict": _verdict(row.perturbed_defect),
                    "verdict_flipped": int(row.verdict_flipped),
                }
            )


def _render_perturbation_plot(
    rows: Sequence[AggregateRow],
    out_path: Path,
) -> None:
    if not rows:
        raise ValueError("plot rows must not be empty")
    canvas = np.full((560, 1040, 3), 255, dtype=np.uint8)
    left, top, width, height = 85, 70, 700, 370
    cv2.rectangle(canvas, (left, top), (left + width, top + height), (70, 70, 70), 1)
    for tick in range(6):
        y = top + height - int(tick * height / 5)
        cv2.line(canvas, (left, y), (left + width, y), (225, 225, 225), 1)
        _put_text(canvas, f"{tick / 5:.1f}", (38, y + 5), 0.42)

    series = {
        "mean score": [row.mean_score for row in rows],
        "mean absolute drift": [row.mean_absolute_drift for row in rows],
        "flip rate": [row.flip_rate for row in rows],
    }
    x_positions = [
        left + int(index * width / max(len(rows) - 1, 1))
        for index in range(len(rows))
    ]
    for name, values in series.items():
        color = PLOT_COLORS[name]
        points = [
            (x, top + height - int(np.clip(value, 0.0, 1.0) * height))
            for x, value in zip(x_positions, values, strict=True)
        ]
        for start, end in zip(points, points[1:]):
            cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, point, 3, color, -1, cv2.LINE_AA)

    _put_text(
        canvas,
        rows[0].perturbation.replace("_", " ").title(),
        (left, 40),
        0.82,
    )
    for x, row in zip(x_positions, rows, strict=True):
        label = _short_level_label(row.level)
        (label_width, _), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            1,
        )
        _put_text(canvas, label, (x - label_width // 2, 475), 0.38)
    for index, name in enumerate(series):
        y = 105 + index * 40
        color = PLOT_COLORS[name]
        cv2.line(canvas, (820, y), (855, y), color, 3)
        _put_text(canvas, name, (865, y + 5), 0.42)
    _put_text(canvas, "metric / rate", (12, top - 12), 0.42)
    _write_compact_png(out_path, canvas)


def _short_level_label(level: str) -> str:
    return (
        level.replace("sigma ", "s")
        .replace("std ", "")
        .replace("quality ", "Q")
        .replace(" deg", "")
    )


def _write_compact_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise OSError(f"Could not write plot: {path}")
    if path.stat().st_size > 150 * 1024:
        raise OSError(f"Plot exceeds 150 KB limit: {path}")


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )


def _validate_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image dimensions must be positive")


def _validate_transformed_image(
    original: np.ndarray,
    transformed: np.ndarray,
) -> None:
    _validate_image(transformed)
    if transformed.shape != original.shape:
        raise ValueError("perturbation changed image shape")


def _validated_score(score: float) -> float:
    value = float(score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("scorer must return a finite score in [0, 1]")
    return value


def _sample_display_path(sample: SampleImage) -> str:
    label = sample.label
    return f"test/{label}/{sample.path.name}"


def _verdict(is_defect: bool) -> str:
    return "DEFECT" if is_defect else "GOOD"


def _mean(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        raise ValueError("cannot average an empty sequence")
    return float(sum(numbers) / len(numbers))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resnet_checkpoint(torch_cache: str | Path | None) -> Path | None:
    if torch_cache is None:
        configured = os.environ.get("FACTORYLENS_TORCH_CACHE")
        if not configured:
            return None
        torch_cache = configured
    checkpoint = Path(torch_cache).expanduser() / "checkpoints/resnet18-f37072fd.pth"
    return checkpoint.resolve() if checkpoint.is_file() else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure hazelnut anomaly-score robustness and verdict flips."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--memory-bank", default=str(DEFAULT_MEMORY_BANK))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--good-count", type=int, default=2)
    parser.add_argument("--defect-per-type", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--torch-cache",
        default=None,
        help="Torch cache containing checkpoints/resnet18-f37072fd.pth.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.torch_cache:
        os.environ["FACTORYLENS_TORCH_CACHE"] = str(
            Path(args.torch_cache).expanduser().resolve()
        )
    samples = collect_samples(
        args.dataset_root,
        good_count=args.good_count,
        defect_per_type=args.defect_per_type,
    )
    scorer = build_real_scorer(
        memory_bank_path=args.memory_bank,
        image_size=args.image_size,
    )
    rows = evaluate_robustness(
        samples,
        scorer=scorer,
        threshold=args.threshold,
        seed=args.seed,
    )
    provenance = build_provenance(
        samples,
        dataset_root=args.dataset_root,
        memory_bank_path=args.memory_bank,
        threshold=args.threshold,
        image_size=args.image_size,
        seed=args.seed,
        torch_cache=args.torch_cache,
    )
    result = write_outputs(
        rows,
        out_dir=args.out_dir,
        provenance=provenance,
    )
    flip_count = sum(row.verdict_flipped for row in rows)
    print(f"samples={len(samples)}")
    print(f"measurements={len(rows)}")
    print(f"verdict_flips={flip_count}")
    print(result["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
