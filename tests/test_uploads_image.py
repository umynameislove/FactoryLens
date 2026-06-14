"""Security and contract tests for product image uploads."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from factorylens.config import Settings, get_settings
from factorylens.main import app

client = TestClient(app)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=(120, 80, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=(120, 80, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _settings(tmp_path: Path, *, max_image_mb: int = 10) -> Settings:
    return Settings(
        database_url="sqlite://",
        upload_dir=str(tmp_path / "uploads"),
        max_image_mb=max_image_mb,
        _env_file=None,
    )


def _post_image(
    settings: Settings,
    *,
    filename: str,
    content: bytes,
    content_type: str,
):
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        return client.post(
            "/uploads/image",
            files={"file": (filename, content, content_type)},
        )
    finally:
        app.dependency_overrides.clear()


def test_valid_png_is_stored_under_upload_dir_with_server_name(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    content = _png_bytes()

    response = _post_image(
        settings,
        filename="../../client-name.png",
        content=content,
        content_type="image/png",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["image_id"]
    assert body["content_type"] == "image/png"
    assert body["size_bytes"] == len(content)
    assert not Path(body["stored_path"]).is_absolute()
    assert "client-name" not in body["stored_path"]

    stored_file = Path(settings.upload_dir) / Path(body["stored_path"]).name
    assert stored_file.is_file()
    assert stored_file.suffix == ".png"
    assert stored_file.read_bytes() == content


def test_valid_jpeg_uses_extension_from_validated_content(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    content = _jpeg_bytes()

    response = _post_image(
        settings,
        filename="client-supplied.png",
        content=content,
        content_type="image/jpeg",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content_type"] == "image/jpeg"
    assert Path(body["stored_path"]).suffix == ".jpg"
    assert "client-supplied" not in body["stored_path"]


def test_non_image_bytes_are_rejected_by_signature_check(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    response = _post_image(
        settings,
        filename="looks-valid.png",
        content=b"this is not an image",
        content_type="image/png",
    )

    assert response.status_code == 400
    assert list(Path(settings.upload_dir).glob("*")) == []


def test_oversize_image_is_rejected_and_temp_file_is_removed(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, max_image_mb=1)

    response = _post_image(
        settings,
        filename="large.png",
        content=b"x" * (1024 * 1024 + 1),
        content_type="image/png",
    )

    assert response.status_code == 400
    assert "1 MB limit" in response.json()["detail"]
    assert list(Path(settings.upload_dir).glob("*")) == []


def test_declared_type_must_match_validated_type(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    response = _post_image(
        settings,
        filename="mismatch.jpg",
        content=_png_bytes(),
        content_type="image/jpeg",
    )

    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]
    assert list(Path(settings.upload_dir).glob("*")) == []


def test_storage_failure_does_not_expose_absolute_path(tmp_path: Path) -> None:
    blocked_path = tmp_path / "not-a-directory"
    blocked_path.write_text("occupied")
    settings = Settings(
        database_url="sqlite://",
        upload_dir=str(blocked_path),
        _env_file=None,
    )

    response = _post_image(
        settings,
        filename="valid.png",
        content=_png_bytes(),
        content_type="image/png",
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Image could not be stored."}
    assert str(tmp_path) not in response.text
