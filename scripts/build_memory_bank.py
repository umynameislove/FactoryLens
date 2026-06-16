"""Build the local hazelnut memory bank used by image anomaly scoring."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_TRAIN_DIR = Path("data/mvtec/hazelnut/train/good")
DEFAULT_OUT = Path("data/memory_bank.npz")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

MemoryBankBuilder = Callable[..., Any]


@dataclass(frozen=True)
class BuildMemoryBankResult:
    image_count: int
    output_path: Path
    memory_bank_shape: tuple[int, ...]


def collect_training_images(train_dir: str | Path) -> list[Path]:
    """Return sorted image files from the MVTec train/good directory."""

    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory does not exist: {train_path}")
    if not train_path.is_dir():
        raise NotADirectoryError(f"Training path is not a directory: {train_path}")

    image_paths = sorted(
        path
        for path in train_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No training images found in {train_path}")

    return image_paths


def build_memory_bank_from_dir(
    train_dir: str | Path = DEFAULT_TRAIN_DIR,
    out: str | Path = DEFAULT_OUT,
    *,
    max_patches_per_image: int = 128,
    image_size: int = 224,
    seed: int = 13,
    extractor: object | None = None,
    builder: MemoryBankBuilder | None = None,
) -> BuildMemoryBankResult:
    """Build a memory bank from a train/good folder and write it to disk."""

    image_paths = collect_training_images(train_dir)

    if builder is None:
        from factorylens.vision.anomaly import build_memory_bank

        builder = build_memory_bank

    if extractor is None:
        from factorylens.vision.embeddings import PatchEmbeddingExtractor

        extractor = PatchEmbeddingExtractor(image_size=image_size, pretrained=True)

    output_path = Path(out)
    memory_bank = builder(
        [path.as_posix() for path in image_paths],
        out_path=str(output_path),
        extractor=extractor,
        max_patches_per_image=max_patches_per_image,
        seed=seed,
    )

    return BuildMemoryBankResult(
        image_count=len(image_paths),
        output_path=output_path,
        memory_bank_shape=tuple(int(value) for value in memory_bank.shape),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/memory_bank.npz from MVTec hazelnut train/good images.",
    )
    parser.add_argument(
        "--train-dir",
        default=str(DEFAULT_TRAIN_DIR),
        help="Folder containing normal training images.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output .npz path used by analyze_image_defect.",
    )
    parser.add_argument(
        "--max-patches-per-image",
        type=int,
        default=128,
        help="Maximum patch embeddings sampled from each training image.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square input size used by the ResNet18 patch extractor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for deterministic patch subsampling.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_memory_bank_from_dir(
        train_dir=args.train_dir,
        out=args.out,
        max_patches_per_image=args.max_patches_per_image,
        image_size=args.image_size,
        seed=args.seed,
    )

    print(f"Built memory bank from {result.image_count} images.")
    print(f"Saved to {result.output_path.as_posix()}.")
    print(f"Memory bank shape: {result.memory_bank_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
