"""CLI for idempotently seeding committed manufacturing test-log data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from factorylens.config import Settings, get_settings
from factorylens.db import SessionLocal, TestLog, init_db
from factorylens.ingest.logs import (
    LogValidationError,
    ingest_log_bytes,
    read_bounded_log_bytes,
)
from factorylens.schemas import UploadLogsResponse

DEFAULT_LOG_CSV = "assets/sample_logs/hazelnut_test_logs.csv"

logger = logging.getLogger(__name__)


def seed_test_logs(
    db: Session,
    settings: Settings,
    csv_path: str,
    reset: bool = False,
) -> UploadLogsResponse:
    """Seed TestLog rows once, or replace them when reset is explicitly enabled."""

    resolved_path = _resolve_csv_path(csv_path)
    existing_row = db.scalar(select(TestLog.id).limit(1))
    if existing_row is not None and not reset:
        logger.info("Skipping seed because test_logs already contains data.")
        return UploadLogsResponse(
            rows_received=0,
            rows_ingested=0,
            rows_rejected=0,
            errors=[],
        )

    with resolved_path.open("rb") as source:
        payload = read_bounded_log_bytes(source, settings.max_logs_mb)

    if not reset:
        return ingest_log_bytes(payload, db, settings)

    try:
        db.execute(delete(TestLog))
        result = ingest_log_bytes(payload, db, settings)
        if result.rows_ingested == 0:
            db.commit()
        return result
    except (LogValidationError, SQLAlchemyError):
        db.rollback()
        raise


def _resolve_csv_path(csv_path: str) -> Path:
    requested_path = Path(csv_path).expanduser()
    try:
        resolved_path = requested_path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Seed CSV does not exist: {csv_path}") from None

    if not resolved_path.is_file():
        raise ValueError(f"Seed CSV path is not a file: {csv_path}")
    return resolved_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed FactoryLens manufacturing test logs."
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_LOG_CSV,
        help="CSV path following the FactoryLens test-log contract.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing test logs before seeding.",
    )
    args = parser.parse_args()

    try:
        settings = get_settings()
        configured_log_level = getattr(
            logging,
            settings.log_level.upper(),
            logging.INFO,
        )
        if not isinstance(configured_log_level, int):
            configured_log_level = logging.INFO
        logging.basicConfig(
            level=configured_log_level,
        )
        init_db()
        with SessionLocal() as db:
            result = seed_test_logs(
                db,
                settings,
                csv_path=args.csv,
                reset=args.reset,
            )
    except (FileNotFoundError, ValueError, LogValidationError) as exc:
        parser.exit(status=1, message=f"Seed failed: {exc}\n")
    except (OSError, SQLAlchemyError) as exc:
        logger.error("Seed failed (%s).", type(exc).__name__)
        parser.exit(status=1, message="Seed failed.\n")

    print(
        "Seed summary: "
        f"received={result.rows_received} "
        f"ingested={result.rows_ingested} "
        f"rejected={result.rows_rejected}"
    )


if __name__ == "__main__":
    main()
