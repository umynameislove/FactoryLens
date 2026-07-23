"""Evaluate and calibrate one hazelnut memory bank on the complete test split.

The script deliberately receives the bank path as an argument and records its
SHA-256.  This makes a proposed threshold auditable and prevents reusing a
number calibrated against another memory bank.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFECT_LABELS = ("crack", "cut", "hole", "print")


@dataclass(frozen=True)
class SampleScore:
    image_path: Path
    label: str
    is_defect: bool
    score: float


@dataclass(frozen=True)
class Metrics:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def precision(self) -> float:
        return self.true_positive / max(self.true_positive + self.false_positive, 1)

    @property
    def recall(self) -> float:
        return self.true_positive / max(self.true_positive + self.false_negative, 1)

    @property
    def f1(self) -> float:
        return 2.0 * self.precision * self.recall / max(self.precision + self.recall, 1e-12)

    @property
    def specificity(self) -> float:
        return self.true_negative / max(self.true_negative + self.false_positive, 1)

    @property
    def youden_j(self) -> float:
        return self.recall + self.specificity - 1.0


def collect_test_paths(root: str | Path) -> list[tuple[Path, str, bool]]:
    """Return every good and defect PNG in the canonical hazelnut test split."""

    test_dir = Path(root) / "test"
    groups = [("good", False), *((label, True) for label in DEFECT_LABELS)]
    paths: list[tuple[Path, str, bool]] = []
    for label, is_defect in groups:
        label_paths = sorted((test_dir / label).glob("*.png"))
        if not label_paths:
            raise FileNotFoundError(f"No test images found in {test_dir / label}")
        paths.extend((path, label, is_defect) for path in label_paths)
    return paths


def score_dataset(
    root: str | Path,
    memory_bank_path: str | Path,
    *,
    image_size: int,
) -> tuple[list[SampleScore], tuple[int, int], float]:
    """Score the whole test split with exactly the supplied memory bank."""

    from factorylens.vision.anomaly import load_memory_bank, score_image
    from factorylens.vision.embeddings import PatchEmbeddingExtractor

    memory_bank, distance_scale = load_memory_bank(str(memory_bank_path))
    if memory_bank.ndim != 2 or memory_bank.shape[1] != 384:
        raise ValueError(
            "Expected a two-dimensional 384-d memory bank; rebuild with the "
            "ResNet18 layer2+layer3 extractor."
        )
    extractor = PatchEmbeddingExtractor(
        image_size=image_size,
        pretrained=True,
        allow_untrained_fallback=False,
    )
    samples = [
        SampleScore(
            image_path=path,
            label=label,
            is_defect=is_defect,
            score=score_image(
                str(path),
                memory_bank=memory_bank,
                distance_scale=distance_scale,
                extractor=extractor,
            )[0],
        )
        for path, label, is_defect in collect_test_paths(root)
    ]
    return samples, tuple(int(value) for value in memory_bank.shape), distance_scale


def metrics_at(samples: Iterable[SampleScore], threshold: float) -> Metrics:
    tp = fp = tn = fn = 0
    for sample in samples:
        predicted_defect = sample.score >= threshold
        if predicted_defect and sample.is_defect:
            tp += 1
        elif predicted_defect:
            fp += 1
        elif sample.is_defect:
            fn += 1
        else:
            tn += 1
    return Metrics(threshold, tp, fp, tn, fn)


def candidate_metrics(samples: Sequence[SampleScore]) -> list[Metrics]:
    scores = sorted({sample.score for sample in samples})
    if not scores:
        raise ValueError("Cannot calibrate an empty score list")
    # All candidates lie in the observed range, as required for the handoff.
    candidates = set(scores)
    candidates.update((left + right) / 2.0 for left, right in zip(scores, scores[1:]))
    return [metrics_at(samples, threshold) for threshold in sorted(candidates)]


def choose_f1(samples: Sequence[SampleScore]) -> Metrics:
    return max(
        candidate_metrics(samples),
        key=lambda row: (row.f1, row.recall, row.youden_j, -row.false_positive),
    )


def choose_youden(samples: Sequence[SampleScore]) -> Metrics:
    return max(
        candidate_metrics(samples),
        key=lambda row: (row.youden_j, row.f1, -row.false_positive, row.recall),
    )


def compute_auroc(samples: Sequence[SampleScore]) -> float:
    positives = sum(sample.is_defect for sample in samples)
    negatives = len(samples) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both good and defect samples")
    ordered = sorted(samples, key=lambda sample: sample.score)
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        tie_end = index + 1
        while tie_end < len(ordered) and ordered[tie_end].score == ordered[index].score:
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        positive_rank_sum += average_rank * sum(
            sample.is_defect for sample in ordered[index:tie_end]
        )
        index = tie_end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def verify_no_train_test_leakage(
    train_dir: str | Path,
    samples: Sequence[SampleScore],
) -> int:
    """Reject a train set that shares bytes with any image in the test split."""

    train_paths = sorted(Path(train_dir).glob("*.png"))
    if len(train_paths) != 391:
        raise ValueError(f"Expected 391 train/good images, found {len(train_paths)} in {train_dir}")
    train_hashes = {_md5(path) for path in train_paths}
    overlapping = [sample.image_path for sample in samples if _md5(sample.image_path) in train_hashes]
    if overlapping:
        first = overlapping[0].as_posix()
        raise ValueError(f"Train/test data leakage detected; matching image: {first}")
    return len(train_paths)


def write_scores(samples: Sequence[SampleScore], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "label", "is_defect", "score"))
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "image_path": sample.image_path.as_posix(),
                    "label": sample.label,
                    "is_defect": int(sample.is_defect),
                    "score": f"{sample.score:.8f}",
                }
            )


def format_output(
    samples: Sequence[SampleScore],
    *,
    bank_path: Path,
    bank_shape: tuple[int, int],
    distance_scale: float,
    auroc: float,
    f1: Metrics,
    youden: Metrics,
    selected: Metrics,
    selection: str,
    train_count: int,
) -> str:
    scores = [sample.score for sample in samples]
    lines = [
        "FactoryLens full threshold evaluation",
        f"memory_bank={bank_path.resolve().as_posix()}",
        f"memory_bank_sha256={_sha256(bank_path)}",
        f"memory_bank_shape={bank_shape}",
        f"distance_scale={distance_scale:.8f}",
        f"train_good_images={train_count}; md5_train_test_overlap=0",
        f"samples={len(samples)} (good={sum(not sample.is_defect for sample in samples)}, defect={sum(sample.is_defect for sample in samples)})",
        "",
        "Score distribution by label:",
    ]
    for label in ("good", *DEFECT_LABELS):
        values = [sample.score for sample in samples if sample.label == label]
        lines.append(
            f"  {label}: n={len(values)} min={min(values):.8f} max={max(values):.8f} mean={np.mean(values):.8f}"
        )
    lines.extend(
        [
            "",
            f"AUROC={auroc:.8f}",
            f"score_range={min(scores):.8f}..{max(scores):.8f} width={max(scores) - min(scores):.8f}",
            _format_metrics("best_f1", f1),
            _format_metrics("best_youden", youden),
            _format_metrics(f"selected_{selection}", selected),
            "acceptance=" + ("PASS" if _acceptance(auroc, selected, scores) else "FAIL"),
        ]
    )
    return "\n".join(lines) + "\n"


def write_metadata(
    out_path: str | Path,
    *,
    build_metadata_path: str | Path,
    samples: Sequence[SampleScore],
    bank_path: Path,
    bank_shape: tuple[int, int],
    distance_scale: float,
    auroc: float,
    selected: Metrics,
    selection: str,
    train_count: int,
) -> None:
    build = json.loads(Path(build_metadata_path).read_text(encoding="utf-8"))
    scores = [sample.score for sample in samples]
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Vision Bundle v2 — Metadata",
        "",
        "## Nguồn gốc",
        "",
        f"- Ngày chạy: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"- Commit hash của repo lúc dựng bank: {build['git_commit']}",
        f"- Máy / OS / Python version: {build['machine']} / {build['os']} / {build['python_version']}",
        "",
        "## Dữ liệu train",
        "",
        f"- Số ảnh train/good đã dùng: {train_count}",
        "- Nguồn: MVTec AD hazelnut train/good",
        "- Có lẫn ảnh test không: KHÔNG (đã kiểm bằng MD5 toàn bộ train/test)",
        "",
        "## Tham số dựng bank",
        "",
        f"- Backbone: {build['backbone']}",
        f"- Layer: {' + '.join(build['layers'])}",
        f"- Embedding dim: {build['embedding_dim']}",
        f"- Coreset size: {build['coreset_size_requested']}",
        f"- Resize ảnh đầu vào: {build['input_resize']} × {build['input_resize']}",
        f"- Normalize: {build['normalize']}",
        "",
        "## Kết quả bank",
        "",
        f"- memory_bank shape: {bank_shape}",
        f"- memory_bank SHA-256: {_sha256(bank_path)}",
        f"- distance_scale: {distance_scale:.8f}",
        "",
        "## Ngưỡng",
        "",
        f"- anomaly_threshold ĐỀ XUẤT: {selected.threshold:.8f}",
        "- Hiệu chỉnh trên bank nào: chính bank trong bundle này (SHA-256 ở trên)",
        f"- Tiêu chí chọn: {selection} (đánh đổi cân bằng sensitivity/specificity cho demo)",
        f"- Ngưỡng này nằm trong dải điểm: min {min(scores):.8f} ≤ threshold ≤ max {max(scores):.8f}",
        "",
        "## Số đo trên 110 ảnh test",
        "",
        f"- AUROC: {auroc:.8f}",
        f"- Dải điểm (min – max): {min(scores):.8f} – {max(scores):.8f}",
        f"- FP trên good: {selected.false_positive}/40",
        f"- FN trên defect: {selected.false_negative}/70",
        f"- Precision / Recall / F1: {selected.precision:.8f} / {selected.recall:.8f} / {selected.f1:.8f}",
        "",
        "## So với baseline v1",
        "",
        f"- AUROC v1 = 0.8304 → v2 = {auroc:.8f}",
        f"- FP v1 (tốt nhất ép được) = 9/40 → v2 = {selected.false_positive}/40",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_metrics(name: str, metrics: Metrics) -> str:
    return (
        f"{name}: threshold={metrics.threshold:.8f} FP={metrics.false_positive}/40 "
        f"FN={metrics.false_negative}/70 precision={metrics.precision:.8f} "
        f"recall={metrics.recall:.8f} f1={metrics.f1:.8f} youden_j={metrics.youden_j:.8f}"
    )


def _acceptance(auroc: float, selected: Metrics, scores: Sequence[float]) -> bool:
    return auroc > 0.8304 and selected.false_positive < 9 and max(scores) - min(scores) > 0.038


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required interoperability checksum.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one memory bank on all 110 hazelnut test images.")
    parser.add_argument("--root", default="data/mvtec/hazelnut")
    parser.add_argument("--train-dir", default="data/mvtec/hazelnut/train/good")
    parser.add_argument("--memory-bank", default="data/memory_bank.npz")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--selection", choices=("f1", "youden"), default="youden")
    parser.add_argument("--threshold", type=float, default=None, help="Evaluate this proposed threshold in addition to both calibrated candidates.")
    parser.add_argument("--scores-out", default=None)
    parser.add_argument("--output", default=None, help="Write the verbatim CLI report to this text file.")
    parser.add_argument("--metadata-out", default=None)
    parser.add_argument("--build-metadata", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bank_path = Path(args.memory_bank)
    samples, bank_shape, distance_scale = score_dataset(
        args.root, bank_path, image_size=args.image_size
    )
    if len(samples) != 110 or sum(not sample.is_defect for sample in samples) != 40:
        raise ValueError("Expected the complete MVTec hazelnut test set: 40 good and 70 defect images")
    train_count = verify_no_train_test_leakage(args.train_dir, samples)
    auroc = compute_auroc(samples)
    f1 = choose_f1(samples)
    youden = choose_youden(samples)
    selected = f1 if args.selection == "f1" else youden
    if args.threshold is not None:
        scores = [sample.score for sample in samples]
        if not min(scores) <= args.threshold <= max(scores):
            raise ValueError("--threshold must lie within the observed score range")
        selected = metrics_at(samples, args.threshold)
    report = format_output(
        samples,
        bank_path=bank_path,
        bank_shape=bank_shape,
        distance_scale=distance_scale,
        auroc=auroc,
        f1=f1,
        youden=youden,
        selected=selected,
        selection="provided" if args.threshold is not None else args.selection,
        train_count=train_count,
    )
    print(report, end="")
    if args.scores_out:
        write_scores(samples, args.scores_out)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
    if args.metadata_out:
        if not args.build_metadata:
            raise ValueError("--metadata-out requires --build-metadata")
        write_metadata(
            args.metadata_out,
            build_metadata_path=args.build_metadata,
            samples=samples,
            bank_path=bank_path,
            bank_shape=bank_shape,
            distance_scale=distance_scale,
            auroc=auroc,
            selected=selected,
            selection="provided threshold" if args.threshold is not None else args.selection,
            train_count=train_count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
