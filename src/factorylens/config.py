"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings.

    ``database_url`` intentionally has no default so database-backed operations
    fail clearly when configuration is missing.
    """

    database_url: str
    app_env: str = "local"
    log_level: str = "INFO"
    upload_dir: str = Field(default="data/uploads", min_length=1)
    max_image_mb: int = Field(default=10, gt=0)
    max_logs_mb: int = Field(default=5, gt=0)
    max_log_rows: int = Field(default=100_000, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("upload_dir")
    @classmethod
    def validate_upload_dir(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("upload_dir must not be blank")
        return value.strip()


@lru_cache
def get_settings() -> Settings:
    """Return validated settings once per process."""

    return Settings()
