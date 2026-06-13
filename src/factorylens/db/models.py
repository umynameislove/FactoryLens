"""Database models for structured manufacturing test logs."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from factorylens.db.base import Base


class TestLog(Base):
    """One measured value for a tested manufacturing unit."""

    __test__ = False
    __tablename__ = "test_logs"
    __table_args__ = (
        CheckConstraint(
            "pass_fail IN ('PASS', 'FAIL')",
            name="ck_test_logs_pass_fail",
        ),
        Index(
            "ix_test_logs_unit_id_measure_name",
            "unit_id",
            "measure_name",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    unit_id: Mapped[str] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    station: Mapped[str] = mapped_column(String(128))
    measure_name: Mapped[str] = mapped_column(String(128), index=True)
    measure_value: Mapped[float] = mapped_column(Float)
    spec_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    spec_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    pass_fail: Mapped[str] = mapped_column(String(4))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
