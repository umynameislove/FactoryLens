"""Read-only, allow-listed query tool over the TestLog table."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from factorylens.db import TestLog
from factorylens.schemas import QueryTestLogsInput, TestLogResult, TestLogRow

_MAX_LIMIT = 1000


def query_test_logs(inp: QueryTestLogsInput, db: Session) -> TestLogResult:
    # (1) Dựng SELECT an toàn — chỉ where trên cột cho phép, dùng bound params.
    stmt = select(TestLog)
    if inp.unit_id is not None:
        stmt = stmt.where(TestLog.unit_id == inp.unit_id)
    if inp.station is not None:
        stmt = stmt.where(TestLog.station == inp.station)
    if inp.measure_name is not None:
        stmt = stmt.where(TestLog.measure_name == inp.measure_name)
    if inp.pass_fail is not None:
        stmt = stmt.where(TestLog.pass_fail == inp.pass_fail)
    if inp.failed_only:
        stmt = stmt.where(TestLog.pass_fail == "FAIL")

    safe_limit = min(max(inp.limit, 1), _MAX_LIMIT)
    stmt = stmt.order_by(TestLog.id).limit(safe_limit)

    # generated_sql CHỈ để hiển thị minh bạch (không phải để chạy).
    generated_sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))

    # (2) Chạy query, bọc lỗi an toàn (không lộ traceback/DSN).
    try:
        records = db.execute(stmt).scalars().all()
    except SQLAlchemyError:
        return TestLogResult(
            generated_sql=generated_sql,
            rows=[],
            row_count=0,
            failed_measures=[],
            summary="query failed",
            warnings=["database error"],
        )

    # (3) Map ORM object -> TestLogRow (Pydantic).
    rows = [
        TestLogRow(
            unit_id=r.unit_id,
            station=r.station,
            measure_name=r.measure_name,
            measure_value=r.measure_value,
            spec_low=r.spec_low,
            spec_high=r.spec_high,
            pass_fail=r.pass_fail,
            timestamp=r.timestamp.isoformat(),
        )
        for r in records
    ]

    fail_set = set()
    for row in rows:
        if row.pass_fail == "FAIL":
            fail_set.add(row.measure_name)

    failed_measures = sorted(fail_set)

    row_count = len(rows)    
    if row_count == 0:
        summary = "No matching test logs"
    else:
        fail_count = sum(1 for r in rows if r.pass_fail == "FAIL")
        if fail_count == 0:
            summary = f"Found {row_count} rows; all PASS."
        else:
            measure_text = ", ".join(failed_measures)
            summary = (
                f"Found {row_count} rows; {fail_count} FAIL "
                f"across measures: {measure_text}."
            )

    return TestLogResult(
        generated_sql=generated_sql,
        rows=rows,
        row_count=len(rows),
        failed_measures=failed_measures,
        summary=summary,
    )