"""Validation and transaction tests for manufacturing log uploads."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from factorylens.config import Settings, get_settings
from factorylens.db import Base, TestLog, get_db
from factorylens.main import app

HEADER = (
    "unit_id,timestamp,station,measure_name,measure_value,"
    "spec_low,spec_high,pass_fail\n"
)
GOOD_ROW = "HZ-001,2026-06-14T08:00:00+00:00,station-1,diameter_mm,20.1,18,22,PASS\n"

client = TestClient(app)


@pytest.fixture
def sqlite_session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()


def _post_logs(
    content: bytes,
    sqlite_session_factory,
    *,
    max_logs_mb: int = 5,
    max_log_rows: int = 100_000,
):
    settings = Settings(
        database_url="sqlite://",
        max_logs_mb=max_logs_mb,
        max_log_rows=max_log_rows,
        _env_file=None,
    )

    def override_get_db() -> Iterator[Session]:
        with sqlite_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        return client.post(
            "/uploads/logs",
            files={"file": ("test-logs.csv", content, "text/csv")},
        )
    finally:
        app.dependency_overrides.clear()


def _row_count(sqlite_session_factory) -> int:
    with sqlite_session_factory() as session:
        return session.scalar(select(func.count()).select_from(TestLog)) or 0


def test_valid_csv_ingests_rows(sqlite_session_factory) -> None:
    content = (
        HEADER + GOOD_ROW + "HZ-002,2026-06-14T08:01:00+00:00,station-2,"
        "weight_g,2.8,2.2,3.4,FAIL\n"
    ).encode()

    response = _post_logs(content, sqlite_session_factory)

    assert response.status_code == 200
    assert response.json() == {
        "rows_received": 2,
        "rows_ingested": 2,
        "rows_rejected": 0,
        "errors": [],
    }
    assert _row_count(sqlite_session_factory) == 2


def test_missing_column_rejects_entire_csv(sqlite_session_factory) -> None:
    content = (
        "unit_id,timestamp,station,measure_name,measure_value,"
        "spec_low,pass_fail\n"
        "HZ-001,2026-06-14T08:00:00+00:00,station-1,"
        "diameter_mm,20.1,18,PASS\n"
    ).encode()

    response = _post_logs(content, sqlite_session_factory)

    assert response.status_code == 400
    assert "exactly the required columns" in response.json()["detail"]
    assert _row_count(sqlite_session_factory) == 0


def test_extra_column_rejects_entire_csv(sqlite_session_factory) -> None:
    content = (
        HEADER.rstrip("\n") + ",unexpected\n" + GOOD_ROW.rstrip("\n") + ",value\n"
    ).encode()

    response = _post_logs(content, sqlite_session_factory)

    assert response.status_code == 400
    assert "exactly the required columns" in response.json()["detail"]
    assert _row_count(sqlite_session_factory) == 0


def test_bad_rows_are_reported_while_good_rows_commit(
    sqlite_session_factory,
) -> None:
    content = (
        HEADER + GOOD_ROW + "HZ-002,2026-06-14T08:01:00+00:00,station-2,"
        "weight_g,not-a-number,2.2,3.4,PASS\n"
        + "HZ-003,2026-06-14T08:02:00+00:00,station-3,"
        "color_index,50,42,58,MAYBE\n"
    ).encode()

    response = _post_logs(content, sqlite_session_factory)

    assert response.status_code == 200
    body = response.json()
    assert body["rows_received"] == 3
    assert body["rows_ingested"] == 1
    assert body["rows_rejected"] == 2
    assert [error["row"] for error in body["errors"]] == [3, 4]
    assert "finite float" in body["errors"][0]["error"]
    assert "PASS or FAIL" in body["errors"][1]["error"]
    assert _row_count(sqlite_session_factory) == 1


def test_row_limit_rejects_csv_before_database_write(
    sqlite_session_factory,
) -> None:
    content = (HEADER + GOOD_ROW + GOOD_ROW).encode()

    response = _post_logs(
        content,
        sqlite_session_factory,
        max_log_rows=1,
    )

    assert response.status_code == 400
    assert "1 row limit" in response.json()["detail"]
    assert _row_count(sqlite_session_factory) == 0


def test_log_size_limit_rejects_csv_before_database_write(
    sqlite_session_factory,
) -> None:
    content = (HEADER + GOOD_ROW).encode() + b"x" * (1024 * 1024)

    response = _post_logs(
        content,
        sqlite_session_factory,
        max_logs_mb=1,
    )

    assert response.status_code == 400
    assert "1 MB limit" in response.json()["detail"]
    assert _row_count(sqlite_session_factory) == 0


def test_error_details_are_capped(sqlite_session_factory) -> None:
    bad_row = "HZ-BAD,not-a-date,station-1,diameter_mm,not-a-number,18,22,UNKNOWN\n"
    content = (HEADER + bad_row * 60).encode()

    response = _post_logs(content, sqlite_session_factory)

    assert response.status_code == 200
    body = response.json()
    assert body["rows_received"] == 60
    assert body["rows_ingested"] == 0
    assert body["rows_rejected"] == 60
    assert len(body["errors"]) == 50


def test_database_failure_does_not_expose_internal_details() -> None:
    class FailingSession:
        def add_all(self, _rows: object) -> None:
            raise SQLAlchemyError("contains-sensitive-database-details")

        def rollback(self) -> None:
            pass

    settings = Settings(database_url="sqlite://", _env_file=None)

    def override_get_db() -> Iterator[Session]:
        yield cast(Session, FailingSession())

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        response = client.post(
            "/uploads/logs",
            files={
                "file": (
                    "test-logs.csv",
                    (HEADER + GOOD_ROW).encode(),
                    "text/csv",
                )
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {"detail": "Logs could not be ingested."}
    assert "sensitive" not in response.text
