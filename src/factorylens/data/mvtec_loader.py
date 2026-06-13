"""Small MVTec AD loader used by the data/vision lane.

The public functions in this module intentionally match the handoff contract in
``../_tracking/JOBS.md`` so backend code can plug them in later without rename
churn.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFECT_LABELS = ("crack", "cut", "hole", "print")
SPLITS = {"train_good", "test_good", "test_defect"}


@dataclass(frozen=True)
class ImageItem:
    image_path: str
    label: str
    mask_path: str | None = None


def list_split(root: str, category: str, split: str) -> list[ImageItem]:
    """List images for an MVTec category split.

    Args:
        root: Directory containing category folders, for example ``data/mvtec``.
        category: MVTec category name, for example ``hazelnut``.
        split: One of ``train_good``, ``test_good``, or ``test_defect``.
    """

    if split not in SPLITS:
        allowed = ", ".join(sorted(SPLITS))
        raise ValueError(f"split must be one of: {allowed}")

    category_dir = Path(root) / category
    if split == "train_good":
        return _items_for_label(category_dir, "train", "good")
    if split == "test_good":
        return _items_for_label(category_dir, "test", "good")

    items: list[ImageItem] = []
    for label in DEFECT_LABELS:
        items.extend(_items_for_label(category_dir, "test", label))
    return sorted(items, key=lambda item: (item.label, item.image_path))


def build_manifest(root: str, category: str, out_csv: str) -> int:
    """Write a CSV manifest and return the number of image rows written."""

    rows: list[dict[str, str]] = []
    for split in ("train_good", "test_good", "test_defect"):
        for item in list_split(root, category, split):
            rows.append(
                {
                    "image_path": _relative_path(item.image_path),
                    "label": item.label,
                    "split": split,
                    "mask_path": _relative_path(item.mask_path) if item.mask_path else "",
                }
            )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "split", "mask_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def _items_for_label(category_dir: Path, split_dir: str, label: str) -> list[ImageItem]:
    image_dir = category_dir / split_dir / label
    if not image_dir.exists():
        return []

    items: list[ImageItem] = []
    for image_path in _iter_images(image_dir):
        mask_path = _mask_for_image(category_dir, label, image_path)
        items.append(
            ImageItem(
                image_path=_path_string(image_path),
                label=label,
                mask_path=_path_string(mask_path) if mask_path else None,
            )
        )
    return items


def _iter_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _mask_for_image(category_dir: Path, label: str, image_path: Path) -> Path | None:
    if label == "good":
        return None

    mask_path = category_dir / "ground_truth" / label / f"{image_path.stem}_mask.png"
    return mask_path if mask_path.exists() else None


def _relative_path(path: str | None) -> str:
    if not path:
        return ""

    return os.path.relpath(path, start=Path.cwd()).replace(os.sep, "/")


def _path_string(path: Path) -> str:
    return path.as_posix()
