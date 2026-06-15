"""Configuration validation tests."""

import pytest
from pydantic import ValidationError

from factorylens.config import Settings


def test_database_url_is_required_without_env_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_defaults_are_local_and_non_secret() -> None:
    settings = Settings(
        database_url="sqlite://",
        _env_file=None,
    )

    assert settings.app_env == "local"
    assert settings.log_level == "INFO"
    assert settings.upload_dir == "data/uploads"
    assert settings.max_image_mb == 10
    assert settings.max_logs_mb == 5
    assert settings.max_log_rows == 100_000
    assert settings.vision_memory_bank_path == "data/memory_bank.npz"
    assert settings.anomaly_threshold == 0.3884
    assert settings.heatmap_dir == "heatmaps"


@pytest.mark.parametrize(
    "field_name",
    ["max_image_mb", "max_logs_mb", "max_log_rows"],
)
def test_upload_limits_must_be_positive(field_name: str) -> None:
    with pytest.raises(ValidationError):
        Settings(
            database_url="sqlite://",
            _env_file=None,
            **{field_name: 0},
        )


@pytest.mark.parametrize(
    "field_name",
    ["upload_dir", "vision_memory_bank_path", "heatmap_dir"],
)
def test_path_settings_must_not_be_blank(field_name: str) -> None:
    with pytest.raises(ValidationError):
        Settings(
            database_url="sqlite://",
            **{field_name: "   "},
            _env_file=None,
        )


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_anomaly_threshold_must_be_unit_interval(threshold: float) -> None:
    with pytest.raises(ValidationError):
        Settings(
            database_url="sqlite://",
            anomaly_threshold=threshold,
            _env_file=None,
        )
