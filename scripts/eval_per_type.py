"""Per-defect evaluation and threshold sensitivity for the hazelnut baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFECT_LABELS = ("crack", "cut", "hole", "print")
REQUIRED_COLUMNS = ("image_path", "label", "is_defect", "score")
DEFAULT_SCORES_CSV = PROJECT_ROOT / "assets/eval_samples/eval_scores.csv"
DEFAULT_SOURCE_REPORT = PROJECT_ROOT / "assets/eval_samples/RESULTS.md"
DEFAULT_OUT_DIR = PROJECT_ROOT / "assets/eval_samples"
PLOT_COLORS = {
    "accuracy": (40, 120, 210),
    "precision": (60, 150, 70),
    "recall": (200, 90, 45),
    "f1": (145, 70, 180),
    "crack": (50, 90, 210),
    "cut": (40, 155, 90),
    "hole": (210, 115, 35),
    "print": (165, 70, 175),
}


@dataclass(frozen=True)
class SampleScore:
    image_path: str
    label: str
    is_defect: bool
    score: float


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def balanced_accuracy(self) -> float:
        positive_total = self.true_positive + self.false_negative
        negative_total = self.true_negative + self.false_positive
        sensitivity = self.true_positive / max(positive_total, 1)
        specificity = self.true_negative / max(negative_total, 1)
        return (sensitivity + specificity) / 2.0


@dataclass(frozen=True)
class PerTypeResult:
    label: str
    sample_count: int
    auroc: float
    current: BinaryMetrics
    best: BinaryMetrics


@dataclass(frozen=True)
class Provenance:
    scores_csv: str
    scores_sha256: str
    scores_bytes: int
    source_report: str | None
    source_report_sha256: str | None
    sample_count: int
    label_counts: dict[str, int]
    source_paths_present: int


def load_scores_csv(path: str | Path) -> list[SampleScore]:
    """Load B10 score output with strict validation."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Score CSV does not exist: {csv_path}")

    samples: list[SampleScore] = []
    seen_paths: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(
                "Score CSV header must be exactly: " + ",".join(REQUIRED_COLUMNS)
            )
        for line_number, row in enumerate(reader, start=2):
            image_path = row["image_path"].strip()
            label = row["label"].strip()
            if not image_path:
                raise ValueError(f"line {line_number}: image_path must not be blank")
            if image_path in seen_paths:
                raise ValueError(f"line {line_number}: duplicate image_path {image_path}")
            seen_paths.add(image_path)
            if label not in {"good", *DEFECT_LABELS}:
                raise ValueError(f"line {line_number}: unsupported label {label!r}")

            is_defect = _parse_binary(row["is_defect"], line_number)
            expected_is_defect = label != "good"
            if is_defect != expected_is_defect:
                raise ValueError(
                    f"line {line_number}: is_defect conflicts with label {label!r}"
                )
            score = _parse_score(row["score"], line_number)
            samples.append(
                SampleScore(
                    image_path=image_path,
                    label=label,
                    is_defect=is_defect,
                    score=score,
                )
            )

    _validate_label_coverage(samples)
    return samples


def binary_metrics(
    samples: Iterable[SampleScore],
    threshold: float,
) -> BinaryMetrics:
    """Return binary defect metrics at one threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("samples must not be empty")

    tp = fp = tn = fn = 0
    for sample in sample_list:
        predicted_defect = sample.score >= threshold
        if predicted_defect and sample.is_defect:
            tp += 1
        elif predicted_defect:
            fp += 1
        elif sample.is_defect:
            fn += 1
        else:
            tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return BinaryMetrics(
        threshold=threshold,
        accuracy=(tp + tn) / len(sample_list),
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def compute_auroc(samples: Iterable[SampleScore]) -> float:
    """Compute image-level AUROC with average ranks for tied scores."""

    sample_list = list(samples)
    positive_count = sum(sample.is_defect for sample in sample_list)
    negative_count = len(sample_list) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC requires both good and defect samples")

    sorted_samples = sorted(sample_list, key=lambda sample: sample.score)
    positive_rank_sum = 0.0
    index = 0
    while index < len(sorted_samples):
        tie_end = index + 1
        while (
            tie_end < len(sorted_samples)
            and sorted_samples[tie_end].score == sorted_samples[index].score
        ):
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        positive_rank_sum += average_rank * sum(
            sample.is_defect for sample in sorted_samples[index:tie_end]
        )
        index = tie_end

    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def threshold_sweep(
    samples: Iterable[SampleScore],
    *,
    start: float = 0.30,
    end: float = 0.50,
    step: float = 0.01,
) -> list[BinaryMetrics]:
    """Evaluate an inclusive, deterministic threshold grid."""

    if not 0.0 <= start <= end <= 1.0:
        raise ValueError("threshold range must satisfy 0 <= start <= end <= 1")
    if step <= 0.0:
        raise ValueError("threshold step must be positive")
    sample_list = list(samples)
    count = int(math.floor((end - start) / step + 1e-9))
    thresholds = [round(start + index * step, 10) for index in range(count + 1)]
    if thresholds[-1] < end - 1e-9:
        thresholds.append(end)
    return [binary_metrics(sample_list, threshold) for threshold in thresholds]


def exact_threshold_search(
    samples: Iterable[SampleScore],
    *,
    start: float,
    end: float,
    current_threshold: float,
) -> list[BinaryMetrics]:
    """Evaluate every decision boundary in-range, not only the plotting grid."""

    if not 0.0 <= start <= end <= 1.0:
        raise ValueError("threshold range must satisfy 0 <= start <= end <= 1")
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("samples must not be empty")

    scores = sorted({sample.score for sample in sample_list})
    candidates = {start, end}
    if start <= current_threshold <= end:
        candidates.add(current_threshold)
    candidates.update(
        midpoint
        for left, right in zip(scores, scores[1:])
        if start <= (midpoint := (left + right) / 2.0) <= end
    )
    return [
        binary_metrics(sample_list, threshold)
        for threshold in sorted(candidates)
    ]


def choose_balanced_threshold(
    sweep: Iterable[BinaryMetrics],
    *,
    current_threshold: float,
) -> BinaryMetrics:
    """Choose a threshold by F1, then balanced accuracy and stability."""

    rows = list(sweep)
    if not rows:
        raise ValueError("threshold sweep must not be empty")
    return max(
        rows,
        key=lambda row: (
            row.f1,
            row.balanced_accuracy,
            row.accuracy,
            -abs(row.threshold - current_threshold),
        ),
    )


def evaluate_per_type(
    samples: list[SampleScore],
    *,
    current_threshold: float,
    sweep_start: float,
    sweep_end: float,
    sweep_step: float,
) -> list[PerTypeResult]:
    """Compare each defect type with all good samples."""

    good_samples = [sample for sample in samples if not sample.is_defect]
    results: list[PerTypeResult] = []
    for label in DEFECT_LABELS:
        defect_samples = [sample for sample in samples if sample.label == label]
        comparison = [*good_samples, *defect_samples]
        exact_candidates = exact_threshold_search(
            comparison,
            start=sweep_start,
            end=sweep_end,
            current_threshold=current_threshold,
        )
        results.append(
            PerTypeResult(
                label=label,
                sample_count=len(defect_samples),
                auroc=compute_auroc(comparison),
                current=binary_metrics(comparison, current_threshold),
                best=choose_balanced_threshold(
                    exact_candidates,
                    current_threshold=current_threshold,
                ),
            )
        )
    return results


def roc_points(samples: Iterable[SampleScore]) -> list[tuple[float, float]]:
    """Return sorted (false-positive-rate, true-positive-rate) ROC points."""

    sample_list = list(samples)
    positives = sum(sample.is_defect for sample in sample_list)
    negatives = len(sample_list) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC requires both good and defect samples")

    points = [(0.0, 0.0)]
    true_positive = false_positive = 0
    by_score: dict[float, list[SampleScore]] = {}
    for sample in sample_list:
        by_score.setdefault(sample.score, []).append(sample)

    for score in sorted(by_score, reverse=True):
        for sample in by_score[score]:
            if sample.is_defect:
                true_positive += 1
            else:
                false_positive += 1
        points.append(
            (
                false_positive / negatives,
                true_positive / positives,
            )
        )
    return points


def write_outputs(
    samples: list[SampleScore],
    *,
    out_dir: str | Path,
    scores_csv: str | Path,
    source_report: str | Path | None,
    source_root: str | Path | None,
    current_threshold: float,
    sweep_start: float,
    sweep_end: float,
    sweep_step: float,
) -> dict[str, object]:
    """Write report and compact plots, returning measured results."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_sweep = threshold_sweep(
        samples,
        start=sweep_start,
        end=sweep_end,
        step=sweep_step,
    )
    exact_candidates = exact_threshold_search(
        samples,
        start=sweep_start,
        end=sweep_end,
        current_threshold=current_threshold,
    )
    recommended = choose_balanced_threshold(
        exact_candidates,
        current_threshold=current_threshold,
    )
    current = binary_metrics(samples, current_threshold)
    per_type = evaluate_per_type(
        samples,
        current_threshold=current_threshold,
        sweep_start=sweep_start,
        sweep_end=sweep_end,
        sweep_step=sweep_step,
    )

    threshold_plot = output_dir / "threshold_sensitivity.png"
    roc_plot = output_dir / "roc_per_type.png"
    confusion_plot = output_dir / "confusion_matrix.png"
    _render_threshold_plot(overall_sweep, threshold_plot)
    _render_roc_plot(samples, per_type, roc_plot)
    _render_confusion_matrix(recommended, confusion_plot)
    provenance = build_provenance(
        samples,
        scores_csv=scores_csv,
        source_report=source_report,
        source_root=source_root,
    )

    report_path = output_dir / "RESULTS_PER_TYPE.md"
    report_path.write_text(
        _build_report(
            samples=samples,
            provenance=provenance,
            current=current,
            recommended=recommended,
            per_type=per_type,
            threshold_plot=threshold_plot.name,
            roc_plot=roc_plot.name,
            confusion_plot=confusion_plot.name,
            sweep_start=sweep_start,
            sweep_end=sweep_end,
            sweep_step=sweep_step,
        ),
        encoding="utf-8",
    )
    return {
        "current": current,
        "recommended": recommended,
        "per_type": per_type,
        "report_path": report_path,
        "plots": [threshold_plot, roc_plot, confusion_plot],
        "provenance": provenance,
    }


def build_provenance(
    samples: list[SampleScore],
    *,
    scores_csv: str | Path,
    source_report: str | Path | None,
    source_root: str | Path | None,
) -> Provenance:
    """Record immutable input fingerprints and local source-path availability."""

    scores_path = Path(scores_csv).resolve()
    report_path = Path(source_report).resolve() if source_report else None
    resolved_source_root = (
        Path(source_root).expanduser().resolve() if source_root else PROJECT_ROOT
    )
    if report_path is not None and not report_path.is_file():
        raise FileNotFoundError(f"Source report does not exist: {report_path}")

    label_counts = {
        label: sum(sample.label == label for sample in samples)
        for label in ("good", *DEFECT_LABELS)
    }
    return Provenance(
        scores_csv=_display_path(scores_path),
        scores_sha256=_sha256(scores_path),
        scores_bytes=scores_path.stat().st_size,
        source_report=_display_path(report_path) if report_path else None,
        source_report_sha256=_sha256(report_path) if report_path else None,
        sample_count=len(samples),
        label_counts=label_counts,
        source_paths_present=sum(
            _resolve_source_image(
                sample.image_path,
                source_root=resolved_source_root,
            ).is_file()
            for sample in samples
        ),
    )


def _build_report(
    *,
    samples: list[SampleScore],
    provenance: Provenance,
    current: BinaryMetrics,
    recommended: BinaryMetrics,
    per_type: list[PerTypeResult],
    threshold_plot: str,
    roc_plot: str,
    confusion_plot: str,
    sweep_start: float,
    sweep_end: float,
    sweep_step: float,
) -> str:
    weakest = min(per_type, key=lambda result: result.auroc)
    lines = [
        "# Hazelnut Per-Defect Evaluation",
        "",
        "## Reproduction",
        "",
        f"- Input scores: `{provenance.scores_csv}`",
        f"- Input SHA-256: `{provenance.scores_sha256}`",
        f"- Input bytes: {provenance.scores_bytes}",
        (
            f"- B10 source report: `{provenance.source_report}` "
            f"(SHA-256 `{provenance.source_report_sha256}`)"
            if provenance.source_report
            else "- B10 source report: not provided"
        ),
        (
            "- Label counts: "
            + ", ".join(
                f"{label}={count}"
                for label, count in provenance.label_counts.items()
            )
        ),
        (
            f"- Source images currently present: "
            f"{provenance.source_paths_present}/{provenance.sample_count}"
        ),
        "- Verify command: `python scripts/eval_per_type.py --source-root ../FactoryLens`",
        "- Test command: `python -m pytest -q tests/test_eval_per_type.py`",
        f"- Current threshold: `{current.threshold:.4f}`",
        (
            f"- Threshold sweep: `{sweep_start:.2f}` to `{sweep_end:.2f}` "
            f"with step `{sweep_step:.2f}`"
        ),
        "- This command recomputes metrics from measured B10 scores; it does not rerun model inference.",
        "- To regenerate scores from images, rerun the B10 eval with the same model and memory-bank settings first.",
        "",
        "## Overall Threshold Check",
        "",
        "| Threshold | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metrics_row(current),
        _metrics_row(recommended),
        "",
        f"Đề xuất `anomaly_threshold = {recommended.threshold:.4f}` (tổng thể) cho Bao.",
        (
            f"Recommendation method: maximize F1 over every observed score boundary "
            f"inside `{sweep_start:.2f}`–`{sweep_end:.2f}`, then balanced accuracy, "
            "accuracy, and proximity to the current threshold."
        ),
        "",
        "## Per-Defect Results",
        "",
        "| Defect | Defect samples | AUROC | Accuracy @ current | Recall @ current | Best F1 threshold | Best F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in per_type:
        lines.append(
            f"| {result.label} | {result.sample_count} | {result.auroc:.3f} | "
            f"{result.current.accuracy:.3f} | {result.current.recall:.3f} | "
            f"{result.best.threshold:.4f} | {result.best.f1:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            f"![Threshold sensitivity]({threshold_plot})",
            "",
            f"![ROC per defect type]({roc_plot})",
            "",
            f"![Confusion matrix]({confusion_plot})",
            "",
            "## Engineering Reading",
            "",
            (
                f"- Weakest separation is `{weakest.label}` with AUROC "
                f"`{weakest.auroc:.3f}`."
            ),
            "- Per-type score ranges overlap with good samples, so a single threshold remains a demo compromise.",
            "- Keep human review for heatmaps and borderline scores; this is not a calibrated defect probability.",
            "- If the memory bank or extractor changes, regenerate the score CSV before reusing this report.",
            "",
        ]
    )
    return "\n".join(lines)


def _metrics_row(metrics: BinaryMetrics) -> str:
    return (
        f"| {metrics.threshold:.4f} | {metrics.accuracy:.3f} | "
        f"{metrics.precision:.3f} | {metrics.recall:.3f} | {metrics.f1:.3f} | "
        f"{metrics.true_positive} | {metrics.false_positive} | "
        f"{metrics.true_negative} | {metrics.false_negative} |"
    )


def _render_threshold_plot(rows: list[BinaryMetrics], out_path: Path) -> None:
    series = {
        "accuracy": [(row.threshold, row.accuracy) for row in rows],
        "precision": [(row.threshold, row.precision) for row in rows],
        "recall": [(row.threshold, row.recall) for row in rows],
        "f1": [(row.threshold, row.f1) for row in rows],
    }
    _render_line_plot(
        series,
        out_path,
        title="Threshold sensitivity",
        x_label="threshold",
        y_label="metric",
        x_range=(rows[0].threshold, rows[-1].threshold),
    )


def _render_roc_plot(
    samples: list[SampleScore],
    per_type: list[PerTypeResult],
    out_path: Path,
) -> None:
    good_samples = [sample for sample in samples if not sample.is_defect]
    auc_by_label = {result.label: result.auroc for result in per_type}
    series: dict[str, list[tuple[float, float]]] = {}
    for label in DEFECT_LABELS:
        defects = [sample for sample in samples if sample.label == label]
        legend = f"{label} AUC={auc_by_label[label]:.3f}"
        series[legend] = roc_points([*good_samples, *defects])
        PLOT_COLORS[legend] = PLOT_COLORS[label]
    _render_line_plot(
        series,
        out_path,
        title="ROC by defect type",
        x_label="false positive rate",
        y_label="true positive rate",
        x_range=(0.0, 1.0),
    )


def _render_confusion_matrix(metrics: BinaryMetrics, out_path: Path) -> None:
    canvas = np.full((520, 680, 3), 255, dtype=np.uint8)
    _put_text(canvas, f"Confusion matrix @ {metrics.threshold:.4f}", (35, 45), 0.8)
    x0, y0, cell = 180, 115, 150
    values = [
        [metrics.true_negative, metrics.false_positive],
        [metrics.false_negative, metrics.true_positive],
    ]
    max_value = max(max(row) for row in values) or 1
    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            intensity = int(245 - 145 * value / max_value)
            color = (255, intensity, intensity)
            left = x0 + col_index * cell
            top = y0 + row_index * cell
            cv2.rectangle(
                canvas,
                (left, top),
                (left + cell, top + cell),
                color,
                -1,
            )
            cv2.rectangle(
                canvas,
                (left, top),
                (left + cell, top + cell),
                (60, 60, 60),
                1,
            )
            _put_text(
                canvas,
                str(value),
                (left + 58, top + 85),
                1.0,
            )
    _put_text(canvas, "pred good", (205, 95), 0.55)
    _put_text(canvas, "pred defect", (345, 95), 0.55)
    _put_text(canvas, "actual good", (55, 200), 0.55)
    _put_text(canvas, "actual defect", (45, 350), 0.55)
    _write_compact_png(out_path, canvas)


def _render_line_plot(
    series: dict[str, list[tuple[float, float]]],
    out_path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_range: tuple[float, float],
) -> None:
    canvas = np.full((520, 1040, 3), 255, dtype=np.uint8)
    left, top, width, height = 80, 65, 700, 365
    cv2.rectangle(
        canvas,
        (left, top),
        (left + width, top + height),
        (70, 70, 70),
        1,
    )
    for tick in range(6):
        y = top + height - int(tick * height / 5)
        cv2.line(canvas, (left, y), (left + width, y), (225, 225, 225), 1)
        _put_text(canvas, f"{tick / 5:.1f}", (35, y + 5), 0.4)
    _put_text(canvas, title, (left, 38), 0.8)
    _put_text(canvas, x_label, (left + width // 2 - 35, 485), 0.5)
    _put_text(canvas, y_label, (12, top - 10), 0.45)
    x_min, x_max = x_range
    for index, (name, points) in enumerate(series.items()):
        color = PLOT_COLORS.get(name, (40 + index * 35, 80, 180))
        pixel_points = []
        for x_value, y_value in points:
            x_ratio = (x_value - x_min) / max(x_max - x_min, 1e-12)
            x = left + int(np.clip(x_ratio, 0.0, 1.0) * width)
            y = top + height - int(np.clip(y_value, 0.0, 1.0) * height)
            pixel_points.append((x, y))
        for start, end in zip(pixel_points, pixel_points[1:]):
            cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)
        legend_y = 85 + index * 34
        cv2.line(canvas, (815, legend_y), (850, legend_y), color, 3)
        _put_text(canvas, name, (860, legend_y + 5), 0.42)
    _put_text(canvas, f"{x_min:.2f}", (left - 10, 452), 0.4)
    _put_text(canvas, f"{x_max:.2f}", (left + width - 25, 452), 0.4)
    _write_compact_png(out_path, canvas)


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


def _parse_binary(value: str, line_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"line {line_number}: is_defect must be 0/1 or true/false")


def _parse_score(value: str, line_number: int) -> float:
    try:
        score = float(value)
    except ValueError as exc:
        raise ValueError(f"line {line_number}: score must be a float") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"line {line_number}: score must be finite and in [0, 1]")
    return score


def _validate_label_coverage(samples: list[SampleScore]) -> None:
    labels = {sample.label for sample in samples}
    missing = {"good", *DEFECT_LABELS} - labels
    if missing:
        raise ValueError(f"Score CSV is missing labels: {', '.join(sorted(missing))}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_source_image(image_path: str, *, source_root: Path) -> Path:
    path = Path(image_path).expanduser()
    return path if path.is_absolute() else source_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hazelnut per-defect metrics and threshold sensitivity."
    )
    parser.add_argument(
        "--scores-csv",
        default=str(DEFAULT_SCORES_CSV),
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
    )
    parser.add_argument(
        "--source-report",
        default=str(DEFAULT_SOURCE_REPORT),
        help="B10 report whose hash is recorded as score provenance.",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Optional local root used only to verify that CSV image paths exist.",
    )
    parser.add_argument("--current-threshold", type=float, default=0.3884)
    parser.add_argument("--sweep-start", type=float, default=0.30)
    parser.add_argument("--sweep-end", type=float, default=0.50)
    parser.add_argument("--sweep-step", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = load_scores_csv(args.scores_csv)
    result = write_outputs(
        samples,
        out_dir=args.out_dir,
        scores_csv=args.scores_csv,
        source_report=args.source_report,
        source_root=args.source_root,
        current_threshold=args.current_threshold,
        sweep_start=args.sweep_start,
        sweep_end=args.sweep_end,
        sweep_step=args.sweep_step,
    )
    recommended = result["recommended"]
    print(f"samples={len(samples)}")
    print(f"recommended_threshold={recommended.threshold:.4f}")
    print(f"f1={recommended.f1:.3f}")
    print(result["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
