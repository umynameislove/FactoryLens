"""SQLite unit tests for the driver-agnostic TestLog model."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from factorylens.db.base import Base
from factorylens.db.models import TestLog


def make_log(*, unit_id: str, pass_fail: str) -> TestLog:
    return TestLog(
        unit_id=unit_id,
        timestamp=datetime.now(timezone.utc),
        station="inspection_01",
        measure_name="diameter_mm",
        measure_value=20.0 if pass_fail == "PASS" else 24.0,
        spec_low=19.0,
        spec_high=21.0,
        pass_fail=pass_fail,
    )


def test_test_log_persists_pass_and_fail_rows() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add_all(
            [
                make_log(unit_id="HZ-0001", pass_fail="PASS"),
                make_log(unit_id="HZ-0002", pass_fail="FAIL"),
            ]
        )
        session.commit()

        rows = session.scalars(
            select(TestLog).order_by(TestLog.unit_id)
        ).all()

    assert [row.pass_fail for row in rows] == ["PASS", "FAIL"]
    assert rows[0].station == "inspection_01"
    assert rows[1].measure_value == 24.0
    assert all(row.id is not None for row in rows)
    assert all(row.created_at is not None for row in rows)

    engine.dispose()


def test_test_log_rejects_unknown_pass_fail_value() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(make_log(unit_id="HZ-0003", pass_fail="UNKNOWN"))
        with pytest.raises(IntegrityError):
            session.commit()

    engine.dispose()
