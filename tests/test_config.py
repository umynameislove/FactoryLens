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
