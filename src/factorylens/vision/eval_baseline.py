"""Mini-evaluate the hazelnut anomaly baseline."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from factorylens.vision.anomaly import build_memory_bank, score_image
from factorylens.vision.embeddings import PatchEmbeddingExtractor
from factorylens.vision.heatmap import extract_regions, make_heatmap


DEFECT_LABELS = ("crack", "cut", "hole", "print")


@dataclass(frozen=True)
class SampleScore:
    image_path: str
    label: str
    is_defect: bool
    score: float


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


def run_eval(
    root: str = "data/mvtec",
    category: str = "hazelnut",
    out_dir: str = "assets/eval_samples",
    memory_bank_path: str = "data/memory_bank_eval.npz",
    max_train_good: int = 24,
    max_test_good: int | None = None,
    max_test_defect_per_label: int | None = None,
    max_patches_per_image: int = 24,
    image_size: int = 96,
    region_threshold: float = 0.35,
) -> dict[str, object]:
    category_dir = Path(root) / category
    train_good = sorted((category_dir / "train" / "good").glob("*.png"))[:max_train_good]
    test_good = _limit(sorted((category_dir / "test" / "good").glob("*.png")), max_test_good)
    defect_paths: list[Path] = []
    for label in DEFECT_LABELS:
        label_paths = sorted((category_dir / "test" / label).glob("*.png"))
        defect_paths.extend(_limit(label_paths, max_test_defect_per_label))

    if not train_good:
        raise ValueError(f"No train/good images found under {category_dir}")
    if not test_good or not defect_paths:
        raise ValueError(f"Need both test/good and test defect images under {category_dir}")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = PatchEmbeddingExtractor(
        image_size=image_size,
        pretrained=True,
        allow_untrained_fallback=False,
    )
    memory_bank = build_memory_bank(
        [str(path) for path in train_good],
        out_path=memory_bank_path,
        extractor=extractor,
        max_patches_per_image=max_patches_per_image,
    )

    samples: list[SampleScore] = []
    for image_path in test_good:
        score, _ = score_image(str(image_path), memory_bank=memory_bank, extractor=extractor)
        samples.append(SampleScore(str(image_path), "good", False, score))
    for image_path in defect_paths:
        label = image_path.parent.name
        score, _ = score_image(str(image_path), memory_bank=memory_bank, extractor=extractor)
        samples.append(SampleScore(str(image_path), label, True, score))

    auroc = compute_auroc(samples)
    metrics = choose_threshold(samples)
    scores_csv = output_dir / "eval_scores.csv"
    distribution_png = output_dir / "score_distribution.png"
    results_md = output_dir / "RESULTS.md"
    write_scores_csv(samples, scores_csv)
    render_score_distribution(samples, distribution_png)
    examples = write_heatmap_examples(
        samples=samples,
        memory_bank=memory_bank,
        extractor=extractor,
        out_dir=output_dir,
        threshold=metrics.threshold,
        region_threshold=region_threshold,
    )
    write_results_markdown(
        samples=samples,
        metrics=metrics,
        auroc=auroc,
        out_path=results_md,
        distribution_path=distribution_png.name,
        examples=examples,
        settings={
            "root": root,
            "category": category,
            "max_train_good": max_train_good,
            "max_patches_per_image": max_patches_per_image,
            "image_size": image_size,
            "region_threshold": region_threshold,
        },
    )

    return {
        "samples": samples,
        "auroc": auroc,
        "metrics": metrics,
        "scores_csv": scores_csv,
        "distribution_png": distribution_png,
        "results_md": results_md,
    }


def compute_auroc(samples: list[SampleScore]) -> float:
    positives = [sample for sample in samples if sample.is_defect]
    negatives = [sample for sample in samples if not sample.is_defect]
    if not positives or not negatives:
        return math.nan

    sorted_samples = sorted(samples, key=lambda sample: sample.score)
    ranks: dict[int, float] = {}
    index = 0
    while index < len(sorted_samples):
        tie_end = index + 1
        while (
            tie_end < len(sorted_samples)
            and sorted_samples[tie_end].score == sorted_samples[index].score
        ):
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        for tied_index in range(index, tie_end):
            ranks[tied_index] = average_rank
        index = tie_end

    positive_rank_sum = sum(
        ranks[index] for index, sample in enumerate(sorted_samples) if sample.is_defect
    )
    positive_count = len(positives)
    negative_count = len(negatives)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def choose_threshold(samples: list[SampleScore]) -> ThresholdMetrics:
    scores = sorted({sample.score for sample in samples})
    if not scores:
        raise ValueError("samples must not be empty")

    candidates = [max(0.0, scores[0] - 1e-6)]
    candidates.extend((left + right) / 2.0 for left, right in zip(scores, scores[1:]))
    candidates.append(min(1.0, scores[-1] + 1e-6))

    metrics = [_threshold_metrics(samples, threshold) for threshold in candidates]
    return max(
        metrics,
        key=lambda item: (
            item.accuracy,
            _balanced_accuracy(item),
            item.recall,
            -item.false_positive,
        ),
    )


def write_scores_csv(samples: list[SampleScore], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "is_defect", "score"],
        )
        writer.writeheader()
        for sample in sorted(samples, key=lambda item: (item.label, item.image_path)):
            writer.writerow(
                {
                    "image_path": sample.image_path,
                    "label": sample.label,
                    "is_defect": int(sample.is_defect),
                    "score": f"{sample.score:.6f}",
                }
            )


def render_score_distribution(samples: list[SampleScore], out_path: Path) -> None:
    width, height = 900, 460
    margin_left, margin_right = 70, 30
    margin_top, margin_bottom = 45, 70
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    good_scores = np.array([sample.score for sample in samples if not sample.is_defect])
    defect_scores = np.array([sample.score for sample in samples if sample.is_defect])
    bins = np.linspace(0.0, 1.0, 21)
    good_hist, _ = np.histogram(good_scores, bins=bins)
    defect_hist, _ = np.histogram(defect_scores, bins=bins)
    max_count = max(int(good_hist.max(initial=0)), int(defect_hist.max(initial=0)), 1)
    bar_width = plot_width / (len(bins) - 1)

    _draw_axes(canvas, margin_left, margin_top, plot_width, plot_height)
    for index, (good_count, defect_count) in enumerate(zip(good_hist, defect_hist)):
        x0 = int(margin_left + index * bar_width)
        x_mid = int(x0 + bar_width / 2)
        x1 = int(x0 + bar_width)
        good_height = int((good_count / max_count) * plot_height)
        defect_height = int((defect_count / max_count) * plot_height)
        baseline = margin_top + plot_height
        cv2.rectangle(canvas, (x0 + 2, baseline - good_height), (x_mid - 1, baseline), (50, 145, 70), -1)
        cv2.rectangle(canvas, (x_mid + 1, baseline - defect_height), (x1 - 2, baseline), (45, 80, 190), -1)

    _put_text(canvas, "Anomaly score distribution", (margin_left, 28), scale=0.8)
    _put_text(canvas, "good", (margin_left + 20, height - 28), color=(50, 145, 70))
    _put_text(canvas, "defect", (margin_left + 130, height - 28), color=(45, 80, 190))
    _put_text(canvas, "0.0", (margin_left - 12, height - 43), scale=0.45)
    _put_text(canvas, "1.0", (width - margin_right - 28, height - 43), scale=0.45)
    cv2.imwrite(str(out_path), canvas)


def write_heatmap_examples(
    samples: list[SampleScore],
    memory_bank: np.ndarray,
    extractor: PatchEmbeddingExtractor,
    out_dir: Path,
    threshold: float,
    region_threshold: float,
) -> list[dict[str, object]]:
    high_defect = max((sample for sample in samples if sample.is_defect), key=lambda item: item.score)
    low_defect = min((sample for sample in samples if sample.is_defect), key=lambda item: item.score)
    high_good = max((sample for sample in samples if not sample.is_defect), key=lambda item: item.score)
    chosen = [
        ("highest_defect", high_defect),
        ("lowest_defect", low_defect),
        ("highest_good", high_good),
    ]

    examples: list[dict[str, object]] = []
    temp_dir = Path("heatmaps/eval_baseline")
    for role, sample in chosen:
        _, score_map = score_image(sample.image_path, memory_bank=memory_bank, extractor=extractor)
        heatmap_path = make_heatmap(sample.image_path, score_map, str(temp_dir))
        output_name = f"eval_{role}_{sample.label}_{Path(sample.image_path).stem}_heatmap.png"
        output_path = out_dir / output_name
        _copy_resized_png(Path(heatmap_path), output_path, size=512)
        regions = extract_regions(score_map, threshold=region_threshold, min_area=2)
        examples.append(
            {
                "role": role,
                "file": output_name,
                "label": sample.label,
                "score": sample.score,
                "predicted_defect": sample.score >= threshold,
                "regions": len(regions),
            }
        )
    return examples


def write_results_markdown(
    samples: list[SampleScore],
    metrics: ThresholdMetrics,
    auroc: float,
    out_path: Path,
    distribution_path: str,
    examples: list[dict[str, object]],
    settings: dict[str, object],
) -> None:
    label_stats = _label_stats(samples)
    per_defect_metrics = _per_defect_metrics(samples)
    verdict = _verdict(auroc, metrics.accuracy)
    lines = [
        "# Hazelnut Baseline Eval",
        "",
        f"Verdict: {verdict}",
        "",
        "## Settings",
        "",
        f"- Dataset: `{settings['root']}/{settings['category']}`",
        f"- Train good images in memory bank: {settings['max_train_good']}",
        f"- Patches per train image: {settings['max_patches_per_image']}",
        f"- Eval image size: {settings['image_size']}",
        f"- Region smoke threshold: {settings['region_threshold']}",
        "",
        "## Image-Level Results",
        "",
        f"- Samples: {len(samples)} ({sum(not s.is_defect for s in samples)} good, {sum(s.is_defect for s in samples)} defect)",
        f"- AUROC: {_format_float(auroc)}",
        f"- Chosen threshold: {metrics.threshold:.4f}",
        f"- Accuracy at threshold: {metrics.accuracy:.3f}",
        f"- Precision: {metrics.precision:.3f}",
        f"- Recall: {metrics.recall:.3f}",
        f"- Recommended demo threshold: `anomaly_threshold = {metrics.threshold:.4f}`",
        "- Đề xuất `anomaly_threshold = "
        f"{metrics.threshold:.4f}` cho Bao cập nhật `config.py`.",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| True positive | {metrics.true_positive} |",
        f"| False positive | {metrics.false_positive} |",
        f"| True negative | {metrics.true_negative} |",
        f"| False negative | {metrics.false_negative} |",
        "",
        "## Score Summary",
        "",
        "| Label | Count | Mean | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ["good", *DEFECT_LABELS]:
        stats = label_stats[label]
        lines.append(
            f"| {label} | {stats['count']} | {stats['mean']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Defect Threshold Check",
            "",
            "Each row compares one defect type against all good samples. These "
            "thresholds are diagnostic only; keep the overall threshold as the "
            "single demo recommendation unless Bao decides otherwise.",
            "",
            "| Defect type | Threshold | Accuracy | Precision | Recall | TP | FP | TN | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in DEFECT_LABELS:
        label_metrics = per_defect_metrics[label]
        lines.append(
            f"| {label} | {label_metrics.threshold:.4f} | "
            f"{label_metrics.accuracy:.3f} | {label_metrics.precision:.3f} | "
            f"{label_metrics.recall:.3f} | {label_metrics.true_positive} | "
            f"{label_metrics.false_positive} | {label_metrics.true_negative} | "
            f"{label_metrics.false_negative} |"
        )

    lines.extend(
        [
            "",
            "## Distribution",
            "",
            f"![Score distribution]({distribution_path})",
            "",
            "## Heatmap Examples",
            "",
            "| Role | Label | Score | Pred defect? | Regions | File |",
            "|---|---|---:|---|---:|---|",
        ]
    )
    for example in examples:
        predicted = "yes" if example["predicted_defect"] else "no"
        lines.append(
            f"| {example['role']} | {example['label']} | {example['score']:.4f} | {predicted} | {example['regions']} | {example['file']} |"
        )
        lines.append(f"![{example['role']}]({example['file']})")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- This is a compact CPU-oriented baseline, not a SOTA detector.",
            "- Threshold is chosen from this eval and should be treated as a demo default.",
            "- False positives/false negatives are still expected because score ranges overlap.",
            "- B6 region counts are smoke evidence only; visual localization quality still needs review.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _threshold_metrics(samples: list[SampleScore], threshold: float) -> ThresholdMetrics:
    tp = fp = tn = fn = 0
    for sample in samples:
        predicted = sample.score >= threshold
        if predicted and sample.is_defect:
            tp += 1
        elif predicted and not sample.is_defect:
            fp += 1
        elif not predicted and not sample.is_defect:
            tn += 1
        else:
            fn += 1

    total = max(len(samples), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return ThresholdMetrics(
        threshold=threshold,
        accuracy=(tp + tn) / total,
        precision=precision,
        recall=recall,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def _balanced_accuracy(metrics: ThresholdMetrics) -> float:
    positive_total = metrics.true_positive + metrics.false_negative
    negative_total = metrics.true_negative + metrics.false_positive
    sensitivity = metrics.true_positive / max(positive_total, 1)
    specificity = metrics.true_negative / max(negative_total, 1)
    return (sensitivity + specificity) / 2.0


def _label_stats(samples: list[SampleScore]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for label in ["good", *DEFECT_LABELS]:
        scores = [sample.score for sample in samples if sample.label == label]
        if not scores:
            stats[label] = {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
            continue
        stats[label] = {
            "count": len(scores),
            "mean": float(np.mean(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }
    return stats


def _per_defect_metrics(samples: list[SampleScore]) -> dict[str, ThresholdMetrics]:
    metrics: dict[str, ThresholdMetrics] = {}
    good_samples = [sample for sample in samples if not sample.is_defect]
    for label in DEFECT_LABELS:
        label_samples = [sample for sample in samples if sample.label == label]
        metrics[label] = choose_threshold([*good_samples, *label_samples])
    return metrics


def _verdict(auroc: float, accuracy: float) -> str:
    if auroc >= 0.80 and accuracy >= 0.75:
        return "PASS"
    if auroc >= 0.60 and accuracy >= 0.60:
        return "PASS WITH GAP"
    return "FAIL"


def _format_float(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.3f}"


def _limit(paths: list[Path], limit: int | None) -> list[Path]:
    return paths if limit is None else paths[:limit]


def _draw_axes(canvas: np.ndarray, left: int, top: int, width: int, height: int) -> None:
    bottom = top + height
    right = left + width
    cv2.line(canvas, (left, top), (left, bottom), (30, 30, 30), 1)
    cv2.line(canvas, (left, bottom), (right, bottom), (30, 30, 30), 1)


def _put_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (30, 30, 30),
    scale: float = 0.55,
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _copy_resized_png(source: Path, destination: Path, size: int) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read generated heatmap: {source}")
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(destination), resized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hazelnut anomaly mini-eval.")
    parser.add_argument("--root", default="data/mvtec")
    parser.add_argument("--category", default="hazelnut")
    parser.add_argument("--out-dir", default="assets/eval_samples")
    parser.add_argument("--memory-bank", default="data/memory_bank_eval.npz")
    parser.add_argument("--max-train-good", type=int, default=24)
    parser.add_argument("--max-test-good", type=int, default=None)
    parser.add_argument("--max-test-defect-per-label", type=int, default=None)
    parser.add_argument("--max-patches-per-image", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--region-threshold", type=float, default=0.35)
    args = parser.parse_args()

    result = run_eval(
        root=args.root,
        category=args.category,
        out_dir=args.out_dir,
        memory_bank_path=args.memory_bank,
        max_train_good=args.max_train_good,
        max_test_good=args.max_test_good,
        max_test_defect_per_label=args.max_test_defect_per_label,
        max_patches_per_image=args.max_patches_per_image,
        image_size=args.image_size,
        region_threshold=args.region_threshold,
    )
    metrics = result["metrics"]
    print(f"AUROC={_format_float(result['auroc'])}")
    print(f"threshold={metrics.threshold:.4f}")
    print(f"accuracy={metrics.accuracy:.3f}")
    print(result["results_md"])


if __name__ == "__main__":
    main()
