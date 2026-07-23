"""Build the local hazelnut memory bank used by image anomaly scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_TRAIN_DIR = Path("data/mvtec/hazelnut/train/good")
DEFAULT_OUT = Path("data/memory_bank.npz")
DEFAULT_CORESET_SIZE = 1024
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

MemoryBankBuilder = Callable[..., Any]


@dataclass(frozen=True)
class BuildMemoryBankResult:
    image_count: int
    output_path: Path
    memory_bank_shape: tuple[int, ...]
    distance_scale: float


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
    coreset_size: int | None = DEFAULT_CORESET_SIZE,
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
        coreset_size=coreset_size,
    )

    with np.load(output_path) as data:
        distance_scale = float(data["distance_scale"][0])

    return BuildMemoryBankResult(
        image_count=len(image_paths),
        output_path=output_path,
        memory_bank_shape=tuple(int(value) for value in memory_bank.shape),
        distance_scale=distance_scale,
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
        default=512,
        help="Square input size used by the ResNet18 patch extractor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for deterministic patch subsampling.",
    )
    parser.add_argument(
        "--coreset-size",
        type=int,
        default=DEFAULT_CORESET_SIZE,
        help="Exact maximum number of representative patch embeddings to retain.",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional one-image-per-line manifest written from the input train directory.",
    )
    parser.add_argument(
        "--manifest-md5-out",
        default=None,
        help="Optional MD5 manifest (digest and relative filename) for the train images.",
    )
    parser.add_argument(
        "--build-metadata-out",
        default=None,
        help="Optional JSON provenance file consumed by eval_threshold_full.py.",
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
        coreset_size=args.coreset_size,
    )

    image_paths = collect_training_images(args.train_dir)
    if args.manifest_out:
        _write_manifest(image_paths, Path(args.train_dir), Path(args.manifest_out))
    if args.manifest_md5_out:
        _write_md5_manifest(image_paths, Path(args.train_dir), Path(args.manifest_md5_out))
    if args.build_metadata_out:
        _write_build_metadata(
            out_path=Path(args.build_metadata_out),
            train_dir=Path(args.train_dir),
            image_count=result.image_count,
            memory_bank_shape=result.memory_bank_shape,
            distance_scale=result.distance_scale,
            image_size=args.image_size,
            max_patches_per_image=args.max_patches_per_image,
            coreset_size=args.coreset_size,
            seed=args.seed,
        )

    print(f"Built memory bank from {result.image_count} images.")
    print(f"Saved to {result.output_path.as_posix()}.")
    print(f"Memory bank shape: {result.memory_bank_shape}")
    print(f"distance_scale: {result.distance_scale:.8f}")
    return 0


def _write_manifest(image_paths: list[Path], train_dir: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(f"{path.relative_to(train_dir).as_posix()}\n" for path in image_paths),
        encoding="utf-8",
    )


def _write_md5_manifest(image_paths: list[Path], train_dir: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{_md5(path)}  {path.relative_to(train_dir).as_posix()}\n"
        for path in image_paths
    ]
    out_path.write_text("".join(lines), encoding="utf-8")


def _write_build_metadata(
    *,
    out_path: Path,
    train_dir: Path,
    image_count: int,
    memory_bank_shape: tuple[int, ...],
    distance_scale: float,
    image_size: int,
    max_patches_per_image: int,
    coreset_size: int | None,
    seed: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "backbone": "ResNet18",
        "layers": ["layer2", "layer3"],
        "embedding_dim": memory_bank_shape[1],
        "image_count": image_count,
        "input_resize": image_size,
        "normalize": "ImageNet mean/std; L2-normalized patch embeddings",
        "max_patches_per_image": max_patches_per_image,
        "coreset_size_requested": coreset_size,
        "memory_bank_shape": list(memory_bank_shape),
        "distance_scale": distance_scale,
        "seed": seed,
        "train_dir": train_dir.resolve().as_posix(),
        "python_version": platform.python_version(),
        "machine": platform.node(),
        "os": platform.platform(),
        "git_commit": _git_commit(),
    }
    out_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required interoperability checksum.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
