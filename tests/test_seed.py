"""SQLite tests for idempotent demo test-log seeding."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from factorylens.config import Settings
from factorylens.db import Base, TestLog
from factorylens.ingest.logs import LogValidationError
from factorylens.seed import seed_test_logs

HEADER = (
    "unit_id,timestamp,station,measure_name,measure_value,"
    "spec_low,spec_high,pass_fail\n"
)


@pytest.fixture
def db_session() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def settings() -> Settings:
    return Settings(database_url="sqlite://", _env_file=None)


def _write_csv(path: Path, *rows: str) -> str:
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return str(path)


def _row(
    unit_id: str,
    *,
    value: str = "20.1",
    pass_fail: str = "PASS",
) -> str:
    return (
        f"{unit_id},2026-06-14T08:00:00+00:00,station-1,"
        f"diameter_mm,{value},18,22,{pass_fail}\n"
    )


def _unit_ids(db: Session) -> list[str]:
    return list(db.scalars(select(TestLog.unit_id).order_by(TestLog.unit_id)))


def test_seed_valid_csv_inserts_rows(
    tmp_path: Path,
    db_session: Session,
    settings: Settings,
) -> None:
    csv_path = _write_csv(
        tmp_path / "logs.csv",
        _row("HZ-001"),
        _row("HZ-002", pass_fail="FAIL"),
    )

    result = seed_test_logs(db_session, settings, csv_path)

    assert result.rows_received == 2
    assert result.rows_ingested == 2
    assert result.rows_rejected == 0
    assert _unit_ids(db_session) == ["HZ-001", "HZ-002"]


def test_seed_twice_without_reset_does_not_duplicate_rows(
    tmp_path: Path,
    db_session: Session,
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    csv_path = _write_csv(tmp_path / "logs.csv", _row("HZ-001"))

    first = seed_test_logs(db_session, settings, csv_path)
    with caplog.at_level("INFO"):
        second = seed_test_logs(db_session, settings, csv_path)

    assert first.rows_ingested == 1
    assert second.rows_received == 0
    assert second.rows_ingested == 0
    assert second.rows_rejected == 0
    assert _unit_ids(db_session) == ["HZ-001"]
    assert "already contains data" in caplog.text


def test_seed_with_reset_replaces_existing_rows(
    tmp_path: Path,
    db_session: Session,
    settings: Settings,
) -> None:
    first_csv = _write_csv(tmp_path / "first.csv", _row("HZ-OLD"))
    replacement_csv = _write_csv(
        tmp_path / "replacement.csv",
        _row("HZ-NEW-1"),
        _row("HZ-NEW-2"),
    )
    seed_test_logs(db_session, settings, first_csv)

    result = seed_test_logs(
        db_session,
        settings,
        replacement_csv,
        reset=True,
    )

    assert result.rows_ingested == 2
    assert _unit_ids(db_session) == ["HZ-NEW-1", "HZ-NEW-2"]


def test_seed_reuses_row_validation_and_keeps_good_rows(
    tmp_path: Path,
    db_session: Session,
    settings: Settings,
) -> None:
    csv_path = _write_csv(
        tmp_path / "mixed.csv",
        _row("HZ-GOOD"),
        _row("HZ-BAD", value="not-a-number"),
    )

    result = seed_test_logs(db_session, settings, csv_path)

    assert result.rows_received == 2
    assert result.rows_ingested == 1
    assert result.rows_rejected == 1
    assert result.errors[0].row == 3
    assert "finite float" in result.errors[0].error
    assert _unit_ids(db_session) == ["HZ-GOOD"]


def test_reset_rolls_back_if_csv_contract_is_invalid(
    tmp_path: Path,
    db_session: Session,
    settings: Settings,
) -> None:
    valid_csv = _write_csv(tmp_path / "valid.csv", _row("HZ-EXISTING"))
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("unit_id,unexpected\nHZ-NEW,value\n", encoding="utf-8")
    seed_test_logs(db_session, settings, valid_csv)

    with pytest.raises(LogValidationError):
        seed_test_logs(
            db_session,
            settings,
            str(invalid_csv),
            reset=True,
        )

    assert _unit_ids(db_session) == ["HZ-EXISTING"]


def test_oversize_csv_does_not_delete_existing_rows_on_reset(
    tmp_path: Path,
    db_session: Session,
) -> None:
    settings = Settings(
        database_url="sqlite://",
        max_logs_mb=1,
        _env_file=None,
    )
    valid_csv = _write_csv(tmp_path / "valid.csv", _row("HZ-EXISTING"))
    oversized_csv = tmp_path / "oversized.csv"
    oversized_csv.write_bytes((HEADER + _row("HZ-NEW")).encode() + b"x" * 1024 * 1024)
    seed_test_logs(db_session, settings, valid_csv)

    with pytest.raises(LogValidationError, match="1 MB limit"):
        seed_test_logs(
            db_session,
            settings,
            str(oversized_csv),
            reset=True,
        )

    assert _unit_ids(db_session) == ["HZ-EXISTING"]


def test_seed_rejects_missing_file(
    tmp_path: Path,
    db_session: Session,
    settings: Settings,
) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        seed_test_logs(db_session, settings, str(missing_path))
