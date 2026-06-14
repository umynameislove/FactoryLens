"""Bounded CSV parsing and transactional ingestion for manufacturing test logs."""

from __future__ import annotations

import csv
import io
import math
from datetime import datetime
from typing import BinaryIO

from fastapi import UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from factorylens.config import Settings
from factorylens.db import TestLog
from factorylens.schemas import LogIngestError, UploadLogsResponse

REQUIRED_COLUMNS = (
    "unit_id",
    "timestamp",
    "station",
    "measure_name",
    "measure_value",
    "spec_low",
    "spec_high",
    "pass_fail",
)

_CHUNK_SIZE = 64 * 1024
_BYTES_PER_MB = 1024 * 1024
_MAX_ERRORS = 50
_MAX_TEXT_LENGTH = 128


class LogValidationError(ValueError):
    """Raised when a CSV cannot be safely interpreted under the log contract."""


def parse_and_ingest_logs(
    file: UploadFile,
    db: Session,
    settings: Settings,
) -> UploadLogsResponse:
    """Validate a bounded CSV and insert all valid rows in one transaction."""

    payload = read_bounded_log_bytes(file.file, settings.max_logs_mb)
    return ingest_log_bytes(payload, db, settings)


def ingest_log_bytes(
    payload: bytes,
    db: Session,
    settings: Settings,
) -> UploadLogsResponse:
    """Validate CSV bytes and insert all valid rows in one transaction."""

    _validate_payload_size(payload, settings.max_logs_mb)
    reader = _make_reader(payload)
    _validate_header(reader)

    rows_received = 0
    valid_logs: list[TestLog] = []
    errors: list[LogIngestError] = []

    try:
        for row in reader:
            rows_received += 1
            if rows_received > settings.max_log_rows:
                raise LogValidationError(
                    f"CSV exceeds the {settings.max_log_rows} row limit."
                )

            line_number = reader.line_num
            try:
                valid_logs.append(_parse_row(row))
            except ValueError as exc:
                if len(errors) < _MAX_ERRORS:
                    errors.append(LogIngestError(row=line_number, error=str(exc)))
    except csv.Error as exc:
        raise LogValidationError("CSV is malformed.") from exc

    if valid_logs:
        try:
            db.add_all(valid_logs)
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            raise

    rows_ingested = len(valid_logs)
    return UploadLogsResponse(
        rows_received=rows_received,
        rows_ingested=rows_ingested,
        rows_rejected=rows_received - rows_ingested,
        errors=errors,
    )


def read_bounded_log_bytes(stream: BinaryIO, max_size_mb: int) -> bytes:
    """Read a binary CSV stream without exceeding the configured byte cap."""

    max_bytes = max_size_mb * _BYTES_PER_MB
    payload = io.BytesIO()
    size_bytes = 0

    while chunk := stream.read(_CHUNK_SIZE):
        size_bytes += len(chunk)
        if size_bytes > max_bytes:
            raise LogValidationError(f"CSV exceeds the {max_size_mb} MB limit.")
        payload.write(chunk)

    return payload.getvalue()


def _validate_payload_size(payload: bytes, max_size_mb: int) -> None:
    if len(payload) > max_size_mb * _BYTES_PER_MB:
        raise LogValidationError(f"CSV exceeds the {max_size_mb} MB limit.")


def _make_reader(payload: bytes) -> csv.DictReader:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LogValidationError("CSV must be UTF-8 encoded.") from exc
    return csv.DictReader(io.StringIO(text, newline=""), strict=True)


def _validate_header(reader: csv.DictReader) -> None:
    try:
        fieldnames = reader.fieldnames
    except csv.Error as exc:
        raise LogValidationError("CSV header is malformed.") from exc

    if (
        fieldnames is None
        or len(fieldnames) != len(REQUIRED_COLUMNS)
        or len(set(fieldnames)) != len(fieldnames)
        or set(fieldnames) != set(REQUIRED_COLUMNS)
    ):
        raise LogValidationError(
            "CSV header must contain exactly the required columns."
        )


def _parse_row(row: dict[str | None, str | list[str] | None]) -> TestLog:
    values: dict[str, str] = {}
    for key, value in row.items():
        if key is None or not isinstance(value, str):
            raise ValueError("Row does not match the CSV header.")
        values[key] = value.strip()

    unit_id = _required_text(values["unit_id"], "unit_id")
    station = _required_text(values["station"], "station")
    measure_name = _required_text(values["measure_name"], "measure_name")
    timestamp = _parse_timestamp(values["timestamp"])
    measure_value = _parse_float(values["measure_value"], "measure_value")
    spec_low = _parse_optional_float(values["spec_low"], "spec_low")
    spec_high = _parse_optional_float(values["spec_high"], "spec_high")
    pass_fail = values["pass_fail"]

    if pass_fail not in {"PASS", "FAIL"}:
        raise ValueError("pass_fail must be PASS or FAIL.")

    return TestLog(
        unit_id=unit_id,
        timestamp=timestamp,
        station=station,
        measure_name=measure_name,
        measure_value=measure_value,
        spec_low=spec_low,
        spec_high=spec_high,
        pass_fail=pass_fail,
    )


def _required_text(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must not be empty.")
    if "\x00" in value:
        raise ValueError(f"{field_name} contains an unsupported character.")
    if len(value) > _MAX_TEXT_LENGTH:
        raise ValueError(
            f"{field_name} exceeds the {_MAX_TEXT_LENGTH} character limit."
        )
    return value


def _parse_timestamp(value: str) -> datetime:
    if not value:
        raise ValueError("timestamp must not be empty.")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601 parseable.") from exc


def _parse_float(value: str, field_name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a finite float.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite float.")
    return parsed


def _parse_optional_float(value: str, field_name: str) -> float | None:
    if not value:
        return None
    return _parse_float(value, field_name)
