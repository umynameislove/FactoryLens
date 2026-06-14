"""Bounded, content-validated storage for uploaded product images."""

from __future__ import annotations

import warnings
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from factorylens.config import Settings

_CHUNK_SIZE = 64 * 1024
_BYTES_PER_MB = 1024 * 1024
_IMAGE_TYPES = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
}


class UploadValidationError(ValueError):
    """Raised when an uploaded image violates the public input contract."""


class ImageStorageError(RuntimeError):
    """Raised when a validated image cannot be stored safely."""


def save_upload_image(
    file: UploadFile,
    settings: Settings,
) -> tuple[str, str, str, int]:
    """Validate and persist one PNG/JPEG without trusting client path metadata."""

    declared_type = file.content_type
    allowed_types = {content_type for content_type, _ in _IMAGE_TYPES.values()}
    if declared_type not in allowed_types:
        raise UploadValidationError("Only PNG and JPEG images are accepted.")

    upload_root = _prepare_upload_root(settings.upload_dir)
    image_id = uuid4().hex
    temporary_path = (upload_root / f".{image_id}.upload").resolve()
    _ensure_inside(temporary_path, upload_root)

    size_bytes = 0
    max_bytes = settings.max_image_mb * _BYTES_PER_MB
    try:
        with temporary_path.open("xb") as destination:
            while chunk := file.file.read(_CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise UploadValidationError(
                        f"Image exceeds the {settings.max_image_mb} MB limit."
                    )
                destination.write(chunk)

        validated_type, extension = _validate_image_contents(temporary_path)
        if validated_type != declared_type:
            raise UploadValidationError(
                "Declared image type does not match the uploaded content."
            )

        final_path = (upload_root / f"{image_id}{extension}").resolve()
        _ensure_inside(final_path, upload_root)
        temporary_path.replace(final_path)
    except UploadValidationError:
        raise
    except OSError as exc:
        raise ImageStorageError("Image could not be stored.") from exc
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

    stored_path = _relative_stored_path(final_path, upload_root)
    return image_id, stored_path, validated_type, size_bytes


def _prepare_upload_root(configured_path: str) -> Path:
    try:
        upload_root = Path(configured_path).expanduser().resolve()
        if upload_root.parent == upload_root:
            raise ImageStorageError("Image storage root is not allowed.")
        upload_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ImageStorageError("Image storage is unavailable.") from exc
    return upload_root


def _validate_image_contents(path: Path) -> tuple[str, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image_format = image.format
                image.verify()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
    ) as exc:
        raise UploadValidationError(
            "Uploaded file is not a valid PNG or JPEG image."
        ) from exc

    image_type = _IMAGE_TYPES.get(image_format or "")
    if image_type is None:
        raise UploadValidationError("Uploaded file is not a valid PNG or JPEG image.")
    return image_type


def _ensure_inside(path: Path, upload_root: Path) -> None:
    try:
        path.relative_to(upload_root)
    except ValueError as exc:
        raise ImageStorageError("Unsafe image storage path.") from exc


def _relative_stored_path(path: Path, upload_root: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.relative_to(upload_root).as_posix()
