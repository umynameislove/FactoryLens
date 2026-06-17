from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from factorylens.db import Base, TestLog
from factorylens.schemas import QueryTestLogsInput
from factorylens.tools.query_logs import query_test_logs


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session.add_all([
        TestLog(unit_id="u1", timestamp=now, station="s1", measure_name="diameter_mm",
                measure_value=5.0, spec_low=4.0, spec_high=6.0, pass_fail="PASS"),
        TestLog(unit_id="u1", timestamp=now, station="s1", measure_name="weight_g",
                measure_value=9.0, spec_low=10.0, spec_high=12.0, pass_fail="FAIL"),
        TestLog(unit_id="u2", timestamp=now, station="s2", measure_name="diameter_mm",
                measure_value=7.0, spec_low=4.0, spec_high=6.0, pass_fail="FAIL"),
    ])
    session.commit()
    yield session
    session.close()


def test_failed_only_returns_only_fail(db):
    res = query_test_logs(QueryTestLogsInput(failed_only=True), db)
    assert res.row_count == 2
    assert all(r.pass_fail == "FAIL" for r in res.rows)
    assert "weight_g" in res.failed_measures


# BẠN TỰ VIẾT thêm:
# - test_filter_unit_id: lọc unit_id="u1" -> đúng 2 dòng.
# - test_limit_capped: thêm nhiều dòng, limit=1 -> chỉ 1 dòng.
# - test_bad_pass_fail_rejected: QueryTestLogsInput(pass_fail="MAYBE")
#   -> mong đợi raise pydantic.ValidationError (dùng pytest.raises).

def test_filter_unit_id(db):
    res = query_test_logs(QueryTestLogsInput(unit_id = "u1"), db)
    assert res.row_count == 2
    assert all(r.unit_id == "u1" for r in res.rows)

def test_limit_capped(db):
    res = query_test_logs(QueryTestLogsInput(limit = 1), db)
    assert res.row_count == 1

def test_bad_pass_fail_rejected():
    with pytest.raises(ValidationError):
        QueryTestLogsInput(pass_fail="MAYBE")